"""Fast approximate backtester for the "signal taker" family (`strategies/signal_taker.py`) on a features npz.

    uv run python -m research.taker_scan data/features/round1.npz                       # default grid
    uv run python -m research.taker_scan data/features/round1.npz --grid '{"signal": ["flow_ev10"],
        "q": [0.9, 0.99], "hold_ms": [1000, 5000], "qty": [25, 50], "slip": [0, 3]}' --top 20
    uv run python -m research.taker_scan data/features/sample.npz --verify data/sample.csv   # signal parity

Grid keys (lists): signal, threshold (absolute) or q (threshold = q-quantile of |signal| on the
two-sided rows of the training part), hold_ms, qty, slip, sign (+1 with / -1 fade), max_pos,
exit_on_reverse (0/1), cooldown_ms. Missing keys take the strategy defaults.

Model (the same state machine as SignalTakerStrategy, evaluated only at candidate rows):
  * signal per row = the offline feature (research.signals building blocks), NaN on one-sided rows;
    `--verify CSV` replays the tape with the strategy's own online code and checks equality.
  * a decision at row i (book after row i) is applied just before the first row j with
    ts_j >= ts_i + 5 ms, against the book after row j-1; the next decision is possible at row j.
  * IOC fill: walk the 5 aggregated levels of the opposite side, ASSUMED at consecutive ticks from
    the touch (the npz has no level prices; optimistic when the book has gaps), up to the slip limit;
    the remainder is cancelled. Our own past takes do not deplete the book (no missing liquidity).
  * exits: whole position, first two-sided row with ts >= entry fill time + hold; reverse signal
    exit when exit_on_reverse. Final position (and the chunk-edge marks) liquidated the same way
    into the book of the last row at/before that time.
Metrics as research.run: pnl, train pnl in --chunks equal time windows of the first 1 - holdout,
chunk wins, holdout pnl, orders, filled volume, max |I|.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from array import array
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from simulator.main import ORDER_ACCEPT_DELAY_MS

from research.features import N_LEVELS, load_features, mid_ticks
from research.signals import _imbalance, build_matrix, spearman, sum_last_events, sum_last_us, value_at_or_before

LAT_US = ORDER_ACCEPT_DELAY_MS * 1000
CACHE_ROOT = Path('logs/taker_scan/cache')
REPORTS_DIR = Path(__file__).parent / 'reports'

SIGNALS = ['imb1', 'imb2', 'imb3', 'imb5', 'micro', 'cnt_imb1', 'flow_ev1', 'flow_ev10', 'flow_ev100', 'flow_ms10',
           'flow_ms100', 'flow_ms1000', 'flow_ms10000', 'ret_ms100', 'ret_ms1000', 'ret_ms10000', 'ltp_dev',
           'last_dmid']  # fmt: skip
DEFAULT_GRID = {
    'signal': ['imb1', 'imb5', 'micro', 'cnt_imb1', 'flow_ev1', 'flow_ev10', 'flow_ev100', 'flow_ms100',
               'flow_ms1000', 'ret_ms100', 'ret_ms1000', 'ltp_dev'],
    'q': [0.8, 0.9, 0.97, 0.99],
    'hold_ms': [500, 2000, 5000, 15000],
    'qty': [10, 25, 50],
    'slip': [0, 2],
}  # fmt: skip
DEFAULTS = {'sign': 1, 'max_pos': 80, 'exit_on_reverse': 0, 'cooldown_ms': 0, 'impact': 0}


# ---------------------------------------------------------------- signals (offline, == OnlineFeatures)


def compute_signal(d: dict[str, np.ndarray], name: str, linear: dict | None = None) -> np.ndarray:
    """Per-row value of `name` exactly as the strategy computes it online (float64, NaN on one-sided rows)."""
    ts = d['ts_us']
    mid = mid_ticks(d)
    bq, aq = d['bid_qty'], d['ask_qty']
    if name.startswith('imb'):
        k = int(name[3:])
        out = _imbalance(bq[:, :k].sum(1, dtype=np.int64), aq[:, :k].sum(1, dtype=np.int64))
    elif name == 'micro':
        out = (d['ask_px'] - d['bid_px']).astype(np.float64) * _imbalance(bq[:, 0], aq[:, 0]) / 2
    elif name == 'cnt_imb1':
        out = _imbalance(d['bid_cnt1'], d['ask_cnt1'])
    elif name.startswith('flow_ev'):
        out = sum_last_events(d['trade_vol'].astype(np.float64), int(name[7:]))
    elif name.startswith('flow_ms'):
        out = sum_last_us(ts, d['trade_vol'].astype(np.float64), int(name[7:]) * 1000)
    elif name.startswith('ret_ms'):
        mid_ff = mid_ticks(d, ffill=True)
        out = mid_ff - value_at_or_before(ts, mid_ff, ts - int(name[6:]) * 1000)
    elif name == 'ltp_dev':
        ltp = d['last_trade_px'].astype(np.float64)
        out = np.where(ltp > 0, ltp - mid, 0.0)
    elif name == 'last_dmid':
        mid_ff = mid_ticks(d, ffill=True)
        dmid = np.diff(mid_ff, prepend=np.nan)
        changed = np.isfinite(dmid) & (dmid != 0)
        last = np.maximum.accumulate(np.where(changed, np.arange(len(ts)), -1))
        out = np.where(last >= 0, dmid[np.maximum(last, 0)], 0.0)
    elif name == 'linear':
        if not linear:
            raise ValueError('signal "linear" needs --linear model.json (an export_linear dict)')
        X, names = build_matrix(d)
        out = np.full(len(ts), linear['intercept'], dtype=np.float64)
        for col, fname in enumerate(names):
            if fname in linear['coef']:
                x = np.clip(X[:, col].astype(np.float64), linear['lo'][fname], linear['hi'][fname])
                out += linear['coef'][fname] * (x - linear['mean'][fname]) / linear['scale'][fname]
        del X
    else:
        raise ValueError(f'unknown signal {name!r}')
    out = np.asarray(out, dtype=np.float64)
    out[np.isnan(mid)] = np.nan
    return out


# ---------------------------------------------------------------- cache shared by worker processes (mmap)


def build_cache(npz: Path, signals: list[str], linear: dict | None) -> Path:
    """Write the arrays the simulation needs as .npy (memory-mapped by the workers); reuse when fresh."""
    cache = CACHE_ROOT / npz.stem
    cache.mkdir(parents=True, exist_ok=True)
    stamp = f'{npz.stat().st_mtime_ns}-{npz.stat().st_size}'
    stamp_file = cache / 'stamp.txt'
    fresh = stamp_file.exists() and stamp_file.read_text() == stamp
    if not fresh:
        for old in cache.glob('*.npy'):
            old.unlink()
    need_base = not (cache / 'ts.npy').exists()
    need_sig = [s for s in signals if s == 'linear' or not (cache / f'sig_{s}.npy').exists()]
    if need_base or need_sig:
        d = load_features(npz)
        if need_base:
            ts = d['ts_us'].astype(np.int64)
            np.save(cache / 'ts.npy', ts)
            np.save(cache / 'bid.npy', d['bid_px'].astype(np.int64))
            np.save(cache / 'ask.npy', d['ask_px'].astype(np.int64))
            np.save(cache / 'bq.npy', np.ascontiguousarray(d['bid_qty'][:, :N_LEVELS], dtype=np.int32))
            np.save(cache / 'aq.npy', np.ascontiguousarray(d['ask_qty'][:, :N_LEVELS], dtype=np.int32))
            np.save(cache / 'act.npy', np.searchsorted(ts, ts + LAT_US, side='left').astype(np.int64))
            valid = (d['bid_px'] > 0) & (d['ask_px'] > 0)
            n = len(ts)
            nxt = np.where(valid, np.arange(n), n)
            nxt = np.minimum.accumulate(nxt[::-1])[::-1]
            np.save(cache / 'next_valid.npy', np.append(nxt, n).astype(np.int64))
            meta = {'source': Path(str(d.get('source', npz))).as_posix(), 'rows': n}
            (cache / 'meta.json').write_text(json.dumps(meta))
        for s in need_sig:
            np.save(cache / f'sig_{s}.npy', compute_signal(d, s, linear))
        del d
    stamp_file.write_text(stamp)
    return cache


_G: dict = {}


def _init_worker(cache: str) -> None:
    c = Path(cache)
    _G['cache'] = c
    for key in ('ts', 'bid', 'ask', 'bq', 'aq', 'act', 'next_valid'):
        _G[key] = np.load(c / f'{key}.npy', mmap_mode='r')
    _G['sig'] = {}
    _G['cand'] = {}


def _signal(name: str) -> np.ndarray:
    sig = _G['sig'].get(name)
    if sig is None:
        sig = np.load(_G['cache'] / f'sig_{name}.npy')
        _G['sig'] = {name: sig}  # keep one signal in memory per worker (jobs are grouped by signal)
        _G['cand'] = {}
    return sig


def _candidates(name: str, sign: int, th: float) -> tuple[array, array]:
    """Sorted row indices where sign * signal > th (long) and < -th (short), as compact int64 arrays for bisect."""
    key = (name, sign, th)
    hit = _G['cand'].get(key)
    if hit is None:
        s = _signal(name)
        if sign < 0:
            s = -s
        with np.errstate(invalid='ignore'):
            hit = (array('q', np.flatnonzero(s > th).astype(np.int64).tobytes()),
                   array('q', np.flatnonzero(s < -th).astype(np.int64).tobytes()))  # fmt: skip
        if len(_G['cand']) >= 4:  # jobs arrive sorted by (signal, sign, threshold): a tiny cache suffices
            _G['cand'] = {}
        _G['cand'][key] = hit
    return hit


# ---------------------------------------------------------------- the state machine


def _walk(levels: list[int], best: int, step: int, limit: int, qty: int) -> tuple[int, int]:
    """Fill `qty` against aggregated levels at consecutive ticks best, best+step, ... while within limit.

    Returns (filled quantity, notional). step = +1 for a buy (asks), -1 for a sell (bids).
    """
    if best <= 0:
        return 0, 0
    filled = 0
    notional = 0
    price = best
    for q in levels:
        if (price > limit) if step > 0 else (price < limit):
            break
        take = q if q < qty - filled else qty - filled
        if take > 0:
            filled += take
            notional += take * price
            if filled == qty:
                break
        price += step
    return filled, notional


def _prune(miss: dict[int, int], levels: list[int], best: int, step: int) -> None:
    """Shrink our missing-liquidity record to what the historical book still shows at each price.

    The historical level quantity at a price can only hide our take while it is at least that big;
    once it drops below (orders cancelled or filled historically) the record shrinks with it; prices
    better than the current best (level gone) are forgotten. Levels beyond the 5th are kept.
    """
    for price in list(miss):
        offset = (price - best) * step
        if offset < 0:
            del miss[price]
        elif offset < len(levels):
            q = levels[offset]
            if q <= 0:
                del miss[price]
            elif miss[price] > q:
                miss[price] = q


def _eff_best(levels: list[int], best: int, step: int, miss: dict[int, int]) -> int:
    """Best price the strategy sees once our own takes are removed (consecutive-tick levels)."""
    if best <= 0:
        return best
    if miss:
        _prune(miss, levels, best, step)
        price = best
        for q in levels:
            if q - miss.get(price, 0) > 0:
                return price
            price += step
        return price
    return best


def _walk_impact(levels: list[int], best: int, step: int, limit: int, qty: int, miss: dict[int, int]) -> tuple[int, int]:
    """`_walk` on the historical levels minus what we already took (missing liquidity); records the new take."""
    if best <= 0:
        return 0, 0
    if miss:
        _prune(miss, levels, best, step)
    filled = 0
    notional = 0
    price = best
    for q in levels:
        if (price > limit) if step > 0 else (price < limit):
            break
        avail = q - miss.get(price, 0)
        take = avail if avail < qty - filled else qty - filled
        if take > 0:
            filled += take
            notional += take * price
            miss[price] = miss.get(price, 0) + take
            if filled == qty:
                break
        price += step
    return filled, notional


def _liquidation(inv: int, row: int) -> float:
    """Markout of `inv` into the book of `row` (5 levels at consecutive ticks; the rest one tick beyond)."""
    if inv == 0:
        return 0.0
    if inv > 0:
        best = int(_G['bid'][row])
        filled, notional = _walk(_G['bq'][row].tolist(), best, -1, -(10**9), inv)
        return notional + (inv - filled) * max(best - N_LEVELS, 0)
    best = int(_G['ask'][row])
    filled, notional = _walk(_G['aq'][row].tolist(), best, 1, 10**12, -inv)
    return -notional - (-inv - filled) * (best + N_LEVELS)


def simulate(cfg: dict) -> dict:
    """Run one config; returns metrics (pnl etc.) plus the fill list summary."""
    ts, bid, ask, bq, aq, act, next_valid = (_G[k] for k in ('ts', 'bid', 'ask', 'bq', 'aq', 'act', 'next_valid'))
    n = len(ts)
    up, dn = _candidates(cfg['signal'], cfg['sign'], cfg['threshold'])
    n_up, n_dn = len(up), len(dn)
    hold = int(cfg['hold_ms'] * 1000)
    qty_cfg, slip, max_pos = int(cfg['qty']), int(cfg['slip']), int(cfg['max_pos'])
    rev = bool(cfg['exit_on_reverse'])
    cool = int(cfg['cooldown_ms'] * 1000)
    impact = bool(cfg.get('impact', 0))
    miss_a: dict[int, int] = {}  # price -> lots we took there that the historical book still shows
    miss_b: dict[int, int] = {}
    i = 0
    inv = 0
    cash = 0
    entry_t = 0
    last_entry_ts = -(10**18)
    max_abs = 0
    orders = 0
    volume = 0
    f_t: list[int] = []
    f_cash: list[int] = []
    f_inv: list[int] = []
    while i < n:
        k0 = i
        if cool and ts[i] - last_entry_ts < cool:
            k0 = max(i, int(ts.searchsorted(last_entry_ts + cool, 'left')))
        if inv == 0:
            pu = bisect_left(up, k0)
            pd = bisect_left(dn, k0)
            ku = up[pu] if pu < n_up else n
            kd = dn[pd] if pd < n_dn else n
            if ku >= n and kd >= n:
                break
            if ku <= kd:
                r, side, q = ku, 1, qty_cfg
            else:
                r, side, q = kd, -1, qty_cfg
            entry = True
        else:
            e = int(ts.searchsorted(entry_t + hold, 'left'))
            r = int(next_valid[max(e, i)])
            side = -1 if inv > 0 else 1
            q = abs(inv)
            entry = False
            if rev:
                arr, m = (dn, n_dn) if inv > 0 else (up, n_up)
                p = bisect_left(arr, i)
                kr = arr[p] if p < m else n
                if kr < r:
                    r = kr
            if abs(inv) < max_pos:
                arr, m = (up, n_up) if inv > 0 else (dn, n_dn)
                p = bisect_left(arr, k0)
                ka = arr[p] if p < m else n
                if ka < r:
                    r, side, q, entry = ka, (1 if inv > 0 else -1), min(qty_cfg, max_pos - abs(inv)), True
            if r >= n:
                break
        if entry:
            last_entry_ts = int(ts[r])
        j = int(act[r])
        b = j - 1
        orders += 1
        if not impact:
            if side > 0:
                filled, notional = _walk(aq[b].tolist(), int(ask[b]), 1, int(ask[r]) + slip, q)
            else:
                filled, notional = _walk(bq[b].tolist(), int(bid[b]), -1, max(1, int(bid[r]) - slip), q)
        elif side > 0:
            limit = _eff_best(aq[r].tolist(), int(ask[r]), 1, miss_a) + slip
            filled, notional = _walk_impact(aq[b].tolist(), int(ask[b]), 1, limit, q, miss_a)
        else:
            limit = max(1, _eff_best(bq[r].tolist(), int(bid[r]), -1, miss_b) - slip)
            filled, notional = _walk_impact(bq[b].tolist(), int(bid[b]), -1, limit, q, miss_b)
        if filled:
            t_act = int(ts[r]) + LAT_US
            was_flat = inv == 0
            inv += side * filled
            cash -= side * notional
            volume += filled
            if was_flat:
                entry_t = t_act
            if abs(inv) > max_abs:
                max_abs = abs(inv)
            f_t.append(t_act)
            f_cash.append(cash)
            f_inv.append(inv)
        i = j
    final = cash + _liquidation(inv, n - 1)
    return {'pnl': final, 'orders': orders, 'volume': volume, 'max_abs': max_abs, 'end_inv': inv,
            'f_t': f_t, 'f_cash': f_cash, 'f_inv': f_inv}  # fmt: skip


def curve_metrics(res: dict, chunks: int, holdout: float) -> dict:
    """PnL at chunk edges (cash + liquidation of I into the book at that time), as research.run.chunk_metrics."""
    ts = _G['ts']
    start, end = int(ts[0]), int(ts[-1])
    hold_start = start + (1.0 - holdout) * (end - start)
    edges = np.linspace(start, hold_start, chunks + 1)
    f_t = np.asarray(res['f_t'], dtype=np.int64)
    values = []
    for edge in edges:
        k = int(np.searchsorted(f_t, edge, side='right')) - 1
        cash = res['f_cash'][k] if k >= 0 else 0
        inv = res['f_inv'][k] if k >= 0 else 0
        row = max(int(ts.searchsorted(edge, 'right')) - 1, 0)
        values.append(cash + _liquidation(inv, row))
    values = np.array(values)
    per = np.diff(values)
    return {
        'train_pnl': round(float(values[-1] - values[0]), 2),
        'hold_pnl': round(float(res['pnl'] - values[-1]), 2),
        'chunk_wins': int((per > 0).sum()),
        'chunk_min': round(float(per.min()), 2),
    }


def run_cfg(job: tuple[dict, int, float]) -> dict:
    cfg, chunks, holdout = job
    t0 = time.perf_counter()
    res = simulate(cfg)
    out = {**cfg, 'pnl': round(float(res['pnl']), 2), 'orders': res['orders'], 'volume': res['volume'],
           'max_abs': res['max_abs'], 'end_inv': res['end_inv'], **curve_metrics(res, chunks, holdout)}  # fmt: skip
    out['secs'] = round(time.perf_counter() - t0, 3)
    return out


# ---------------------------------------------------------------- grid / thresholds / CLI


def expand(grid: dict) -> list[dict]:
    keys = list(grid)
    configs = []
    for values in itertools.product(*(grid[k] for k in keys)):
        cfg = {**DEFAULTS, **dict(zip(keys, values, strict=True))}
        configs.append(cfg)
    return configs


def resolve_thresholds(configs: list[dict], cache: Path, holdout: float) -> None:
    """Turn `q` into an absolute threshold: the q-quantile of |signal| on two-sided training rows."""
    memo: dict[tuple[str, float], float] = {}
    ts = np.load(cache / 'ts.npy', mmap_mode='r')
    n_train = int(np.searchsorted(ts, ts[0] + (1 - holdout) * (ts[-1] - ts[0]), 'right'))
    by_signal: dict[str, np.ndarray] = {}
    for cfg in configs:
        if 'q' not in cfg:
            cfg['threshold'] = float(cfg.get('threshold', 0.0))
            continue
        key = (cfg['signal'], float(cfg['q']))
        if key not in memo:
            if cfg['signal'] not in by_signal:
                by_signal.clear()
                s = np.load(cache / f'sig_{cfg["signal"]}.npy', mmap_mode='r')[:n_train]
                a = np.abs(s[np.isfinite(s)])
                by_signal[cfg['signal']] = a[:: max(1, len(a) // 2_000_000)]
            memo[key] = float(np.quantile(by_signal[cfg['signal']], key[1]))
        cfg['threshold'] = memo[key]


def strategy_overrides(cfg: dict) -> dict:
    """Class-constant overrides for strategies/signal_taker.py that reproduce `cfg`."""
    out = {'SIGNAL': cfg['signal'], 'THRESHOLD': cfg['threshold'], 'HOLD_US': int(cfg['hold_ms'] * 1000),
           'QTY': int(cfg['qty']), 'SLIP_TICKS': int(cfg['slip'])}  # fmt: skip
    if cfg['sign'] != 1:
        out['SIGN'] = int(cfg['sign'])
    if cfg['max_pos'] != 80:
        out['MAX_POS'] = int(cfg['max_pos'])
    if cfg['exit_on_reverse']:
        out['EXIT_ON_REVERSE'] = True
    if cfg['cooldown_ms']:
        out['COOLDOWN_US'] = int(cfg['cooldown_ms'] * 1000)
    return out


def run_command(cfg: dict, source: str) -> str:
    return (f"uv run python -m research.run --strategy strategies/signal_taker.py --data {source} "
            f"--set '{json.dumps(strategy_overrides(cfg))}' --no-ledger")  # fmt: skip


def verify(csv_path: Path, cache: Path, signals: list[str], max_events: int | None = None) -> dict[str, int]:
    """Replay the tape with the strategy's own (non-trading) signal code and compare every row with the cache."""
    from research.run import load_strategy_class, with_params
    from simulator.simulator import ExchangeSimulator, SimulationConfig
    from simulator.strategy import BaseStrategy

    cls = load_strategy_class('strategies/signal_taker.py')
    probes = []
    for name in signals:
        probe = with_params(cls, {'SIGNAL': name})()
        probe._ioc = lambda *args: None  # noqa: SLF001 - never trade
        probes.append(probe)
    values: list[list[float]] = [[] for _ in signals]

    class _Probe(BaseStrategy):
        def on_order_book_change(self, snapshot, trades) -> None:  # noqa: ANN001
            if max_events is None or len(values[0]) < max_events:
                for k, probe in enumerate(probes):
                    probe.on_order_book_change(snapshot, trades)
                    values[k].append(probe.sig if snapshot.bids and snapshot.asks else math.nan)

    ExchangeSimulator(_Probe(), SimulationConfig()).run(csv_path)
    mismatches = {}
    for name, online in zip(signals, values, strict=True):
        online = np.array(online)
        offline = np.load(cache / f'sig_{name}.npy')[: len(online)]
        both_nan = np.isnan(online) & np.isnan(offline)
        bad = ~both_nan & ~(online == offline)
        mismatches[name] = int(bad.sum())
        if bad.any():
            k = int(np.flatnonzero(bad)[0])
            print(f'  {name}: {bad.sum()} mismatches, first row {k}: online {online[k]} offline {offline[k]}')
    return mismatches


def write_report(path: Path, source: str, rows: list[dict], top: int, n_rows: int, runtime: float, args: argparse.Namespace) -> None:
    lines = [
        f'# Signal-taker proxy scan - `{source}`',
        '',
        f'{len(rows)} configs on {n_rows:,} rows in {runtime:.1f} s ({args.workers} workers). Proxy model: decision at '
        'row i, IOC applied against the book after the last row before ts_i + 5 ms, 5 levels assumed at consecutive '
        'ticks, no self-impact. Ranked by train pnl (first '
        f'{1 - args.holdout:.0%} of time, {args.chunks} chunks); holdout = last {args.holdout:.0%}.',
        '',
        '| # | signal | sign | thr | q | hold_ms | qty | slip | rev | pnl | train | wins | chunk_min | hold | orders | vol | max|I| |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for k, r in enumerate(rows[:top], 1):
        lines.append(
            f"| {k} | {r['signal']} | {r['sign']} | {r['threshold']:.4g} | {r.get('q', '')} | {r['hold_ms']} | {r['qty']} | "
            f"{r['slip']} | {int(r['exit_on_reverse'])} | {r['pnl']:,.0f} | {r['train_pnl']:,.0f} | {r['chunk_wins']}/{args.chunks} | "
            f"{r['chunk_min']:,.0f} | {r['hold_pnl']:,.0f} | {r['orders']} | {r['volume']} | {r['max_abs']} |"
        )
    lines += ['', '## Confirm in the real simulator', '']
    for k, r in enumerate(rows[: min(top, 10)], 1):
        lines.append(f'{k}. `{run_command(r, source)}`')
    lines += ['', '## Best config per signal (by train pnl)', '', '| signal | pnl | train | wins | hold | config |', '|---|---:|---:|---:|---:|---|']
    seen = set()
    for r in rows:
        if r['signal'] in seen:
            continue
        seen.add(r['signal'])
        lines.append(f"| {r['signal']} | {r['pnl']:,.0f} | {r['train_pnl']:,.0f} | {r['chunk_wins']} | {r['hold_pnl']:,.0f} | "
                     f"`{json.dumps(strategy_overrides(r))}` |")  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Fast proxy scan of signal-taker configs on a features npz')
    parser.add_argument('npz', type=Path)
    parser.add_argument('--grid', default=None, help='JSON dict of lists (see module docstring)')
    parser.add_argument('--top', type=int, default=20)
    parser.add_argument('--chunks', type=int, default=8)
    parser.add_argument('--holdout', type=float, default=0.25)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--linear', type=Path, default=None, help='JSON file with an export_linear dict (signal "linear")')
    parser.add_argument('--verify', type=Path, default=None, help='tape CSV: check the strategy online signal == proxy signal')
    parser.add_argument('--out', type=Path, default=None, help='report path (default research/reports/<stem>_taker_scan.md)')
    parser.add_argument('--impact', action='store_true', help='model our own missing liquidity (pessimistic bound; default off)')
    parser.add_argument('--json', type=Path, default=None, help='also dump all results as JSON here')
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    grid = json.loads(args.grid) if args.grid else DEFAULT_GRID
    linear = json.loads(args.linear.read_text()) if args.linear else None
    signals = sorted(set(grid.get('signal', ['flow_ev1'])))
    cache = build_cache(args.npz, signals if not args.verify else sorted(set(signals) | set(SIGNALS)), linear)
    meta = json.loads((cache / 'meta.json').read_text())
    print(f'cache {cache} ready [{time.perf_counter() - t0:.1f}s], {meta["rows"]:,} rows')
    if args.verify:
        names = [s for s in SIGNALS]
        bad = verify(args.verify, cache, names)
        print(f'verify: mismatching rows per signal {bad}  [{time.perf_counter() - t0:.1f}s]')
        if not args.grid:
            return
    configs = expand(grid)
    if args.impact:
        for cfg in configs:
            cfg['impact'] = 1
    if 'signal' not in grid:
        for cfg in configs:
            cfg['signal'] = 'flow_ev1'
    resolve_thresholds(configs, cache, args.holdout)
    configs.sort(key=lambda c: (c['signal'], c['sign'], c['threshold']))
    jobs = [(cfg, args.chunks, args.holdout) for cfg in configs]
    t1 = time.perf_counter()
    if args.workers > 1 and len(jobs) > 1:
        # contiguous blocks per worker keep each worker on few signals (one signal array in memory)
        chunksize = max(1, math.ceil(len(jobs) / (args.workers * 4)))
        with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(str(cache),)) as pool:
            rows = list(pool.map(run_cfg, jobs, chunksize=chunksize))
    else:
        _init_worker(str(cache))
        rows = [run_cfg(job) for job in jobs]
    sim_time = time.perf_counter() - t1
    rows.sort(key=lambda r: -r['train_pnl'])
    print(f'{len(rows)} configs simulated in {sim_time:.1f}s ({sim_time / max(len(rows), 1) * args.workers:.2f} s/config/worker)')
    header = f"{'#':>3} {'signal':<12} {'sg':>2} {'thr':>9} {'hold':>6} {'qty':>3} {'slp':>3} {'rev':>3} {'pnl':>10} {'train':>10} {'wins':>4} {'hold_pnl':>9} {'orders':>6} {'maxI':>4}"
    print(header)
    for k, r in enumerate(rows[: args.top], 1):
        print(f"{k:>3} {r['signal']:<12} {r['sign']:>2} {r['threshold']:>9.4g} {r['hold_ms']:>6} {r['qty']:>3} {r['slip']:>3} "
              f"{int(r['exit_on_reverse']):>3} {r['pnl']:>10,.0f} {r['train_pnl']:>10,.0f} {r['chunk_wins']:>4} {r['hold_pnl']:>9,.0f} "
              f"{r['orders']:>6} {r['max_abs']:>4}")  # fmt: skip
    print('\nconfirm the top configs in the real simulator:')
    for r in rows[: min(args.top, 5)]:
        print('  ' + run_command(r, meta['source']))
    report = args.out or REPORTS_DIR / f'{args.npz.stem}_taker_scan.md'
    write_report(report, meta['source'], rows, args.top, meta['rows'], time.perf_counter() - t0, args)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows))
    print(f'report -> {report}  [{time.perf_counter() - t0:.1f}s total]')


if __name__ == '__main__':
    main()
