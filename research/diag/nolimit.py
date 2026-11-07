"""Upper bound: what does the strategy earn without the +-100 penalty? (capacity value diagnostic)

    uv run python -X utf8 research/diag/nolimit.py data/round1.csv strategies/exp_overcap.py '{"max_position":300}' [penalty]
Prints pnl, breach seconds (time with |I| > 100) and the penalty it WOULD have cost at 10,000 $/s.
"""
import json, sys
from pathlib import Path
from research.run import load_strategy_class
from simulator import main as sm
from simulator.simulator import ExchangeSimulator, SimulationConfig

Base = load_strategy_class(sys.argv[2])
ov = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
pen = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
class S(Base):
    PARAMS = {**Base.PARAMS, **ov}
    def __init__(self):
        super().__init__(); self.t_over = 0; self.last = None; self.n_over = 0; self.was = False
    def on_order_book_change(self, snap, trades):
        now = snap.timestamp_us
        if self.last is not None and abs(self.inv) > 100:
            self.t_over += now - self.last
        over = abs(self.inv) > 100
        if over and not self.was: self.n_over += 1
        self.was = over; self.last = now
        super().on_order_book_change(snap, trades)
cfg = SimulationConfig(max_position=sm.MAX_POSITION, max_potential_position=sm.MAX_POTENTIAL_POSITION,
                       position_breach_penalty_per_second=pen, order_accept_delay_ms=sm.ORDER_ACCEPT_DELAY_MS)
st = S()
r = ExchangeSimulator(strategy=st, config=cfg).run(Path(sys.argv[1]))
print(f'{ov} pen/s={pen:.0f}: pnl {r.pnl:,.0f}  breach episodes {st.n_over}  time over 100 {st.t_over/1e6:,.1f} s '
      f'(= {st.t_over/1e6*10000:,.0f} $ at 10k/s)  max|I| {st.max_abs_inv}')
