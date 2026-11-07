"""Run a playground strategy on a tape and show what happened.

    uv run python -X utf8 playground/run.py --data data/round2.csv
    uv run python -X utf8 playground/run.py --data data/round2.csv --frac 0.25 --set '{"trend_thr": 2}'
    uv run python -X utf8 playground/run.py --data data/round1.csv --strategy strategies/mm_ml_r1tuned.py

Prints: pnl (same number the server reports), penalty, fills, volume, inventory quantiles, pnl per 10 % of time.
Writes: playground/out/<name>.html (interactive chart: mid price, pnl, inventory) and <name>_fills.csv.
Uses the official exchange settings (5 ms latency, 100-lot band, 10,000 $/s penalty).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.run import load_strategy_class  # noqa: E402
from simulator import main as sm  # noqa: E402
from simulator.models import OrderStatus, Side  # noqa: E402
from simulator.simulator import ExchangeSimulator, SimulationConfig  # noqa: E402

SAMPLE_US = 1_000_000  # chart resolution: one point per second


def slice_csv(path: Path, frac: float) -> Path:
    """First `frac` of the tape's rows, as a temporary CSV (quick experiments)."""
    lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
    n = 1 + int((len(lines) - 1) * frac)
    tmp = Path(tempfile.gettempdir()) / f'{path.stem}_p{round(frac * 100)}.csv'
    tmp.write_text(''.join(lines[:n]), encoding='utf-8')
    return tmp


def make_recorder(base):
    class Recorder(base):
        """Wraps any strategy: records mid / inventory / pnl every second and every fill."""

        def __init__(self):
            super().__init__()
            self._side = {}
            self._inv = 0
            self._cash = 0.0
            self._next = 0
            self._mid = None
            self.curve = []   # (t_s, mid, inventory, pnl)
            self.fills = []   # (t_s, side, qty, price, inventory_after)

        def send_order(self, order):
            oid = super().send_order(order)
            self._side[oid] = order.side
            return oid

        def on_order_book_change(self, snapshot, trades):
            if snapshot.bids and snapshot.asks:
                self._mid = (snapshot.bids[0].price + snapshot.asks[0].price) / 2
                while snapshot.timestamp_us >= self._next:
                    self.curve.append((self._next / 1e6, self._mid, self._inv, self._cash + self._inv * self._mid))
                    self._next += SAMPLE_US
            super().on_order_book_change(snapshot, trades)

        def on_private_order_update(self, event):
            if event.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED) and event.order_id in self._side:
                sign = 1 if self._side[event.order_id] is Side.BUY else -1
                self._inv += sign * event.filled_quantity
                self._cash -= sign * event.filled_quantity * event.fill_price
                self.fills.append((event.timestamp_us / 1e6, 'B' if sign > 0 else 'S', event.filled_quantity,
                                   event.fill_price, self._inv))
            super().on_private_order_update(event)

    return Recorder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/round2.csv')
    ap.add_argument('--strategy', default='playground/my_strategy.py')
    ap.add_argument('--set', default='{}', help='JSON overrides of PARAMS, e.g. \'{"quote_size": 30}\'')
    ap.add_argument('--frac', type=float, default=1.0, help='use only the first fraction of the tape')
    ap.add_argument('--name', default=None, help='output file name (default: strategy + data)')
    a = ap.parse_args()

    base = load_strategy_class(str(ROOT / a.strategy) if not Path(a.strategy).is_absolute() else a.strategy)
    overrides = json.loads(a.set)
    if overrides:
        base = type(base.__name__, (base,), {'PARAMS': {**base.PARAMS, **overrides}})
    data = ROOT / a.data if not Path(a.data).is_absolute() else Path(a.data)
    if a.frac < 1.0:
        data = slice_csv(data, a.frac)

    strat = make_recorder(base)()
    cfg = SimulationConfig(max_position=sm.MAX_POSITION, max_potential_position=sm.MAX_POTENTIAL_POSITION,
                           position_breach_penalty_per_second=sm.POSITION_BREACH_PENALTY_PER_SECOND,
                           order_accept_delay_ms=sm.ORDER_ACCEPT_DELAY_MS)
    t0 = time.time()
    res = ExchangeSimulator(strategy=strat, config=cfg).run(data)
    secs = time.time() - t0

    vol = sum(f[2] for f in strat.fills)
    invs = sorted(abs(c[2]) for c in strat.curve) or [0]
    q = lambda p: invs[min(len(invs) - 1, int(p * len(invs)))]  # noqa: E731
    print(f'strategy {a.strategy}  data {a.data} (frac {a.frac})  overrides {overrides}')
    print(f'PNL {res.pnl:,.0f}   penalty {res.breach_penalty:,.0f}   fills {len(strat.fills):,}   volume {vol:,} lots   '
          f'run {secs:.0f} s')
    print(f'|inventory| quantiles 50/90/99 %: {q(0.5)} / {q(0.9)} / {q(0.99)}')
    c = strat.curve
    if len(c) > 10:
        step = len(c) // 10
        parts = [c[min(i * step, len(c) - 1)][3] for i in range(11)]
        parts[-1] = res.pnl
        print('pnl per 10 % of time:', ' '.join(f'{parts[i + 1] - parts[i]:+,.0f}' for i in range(10)))

    out = ROOT / 'playground' / 'out'
    out.mkdir(parents=True, exist_ok=True)
    name = a.name or f'{Path(a.strategy).stem}_{Path(a.data).stem}'
    with open(out / f'{name}_fills.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['t_s', 'side', 'qty', 'price', 'inventory_after'])
        w.writerows(strat.fills)
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=('mid price', 'pnl ($)', 'inventory'))
        ts = [x[0] for x in c]
        fig.add_trace(go.Scatter(x=ts, y=[x[1] for x in c], name='mid'), 1, 1)
        fig.add_trace(go.Scatter(x=ts, y=[x[3] for x in c], name='pnl'), 2, 1)
        fig.add_trace(go.Scatter(x=ts, y=[x[2] for x in c], name='inventory'), 3, 1)
        fig.update_layout(height=900, title=f'{name}: pnl {res.pnl:,.0f}')
        fig.write_html(out / f'{name}.html')
        print(f'chart: {out / (name + ".html")}')
    except ImportError:
        print('plotly not installed: no chart')


if __name__ == '__main__':
    main()
