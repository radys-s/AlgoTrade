# Playground: your own strategy experiments

Two files, nothing else to learn:

| File | What it is |
|---|---|
| `playground/my_strategy.py` | **The strategy you edit.** Simple passive market maker + trend filter. All knobs in `PARAMS`, logic in the block marked `STRATEGY LOGIC`. Can be uploaded to the arena as is. |
| `playground/run.py` | Runner: loads a tape CSV, runs the official simulator (same result as the server), prints stats, saves a chart + fills CSV to `playground/out/`. |

## Baseline result (unchanged defaults)

| Tape | PNL | Notes |
|---|---|---|
| `data/round2.csv` | **+42,193** | 0 penalty, 7,015 fills, 149k lots, 75 s run. Our R1 champions lose −148k..−230k here. |
| `data/round1.csv` | see `docs/progress.md` | R1 champion `mm_ml_r1tuned` = 497,292 |

pnl per 10 % of the R2 tape: `-5k +2k +6k -9k -1k +7k +20k +6k +11k +7k`.

## Run it

Local (Windows, Git Bash, repo root):

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run python -X utf8 playground/run.py --data data/round2.csv                   # full tape, ~75 s
uv run python -X utf8 playground/run.py --data data/round2.csv --frac 0.25       # first 25 %, ~20 s
uv run python -X utf8 playground/run.py --set '{"trend_thr": 2, "quote_size": 60}' --name thr2
uv run python -X utf8 playground/run.py --strategy strategies/mm_ml_r1tuned.py    # any strategy file works
```

VS Code: open the repo folder, select the `.venv` interpreter (`uv sync` creates it), run `playground/run.py`
with arguments, or use the terminal commands above. Open `playground/out/<name>.html` in a browser for the chart
(mid price / pnl / inventory over time, zoomable).

Colab (needs Python 3.13 per `pyproject.toml`; if Colab has an older Python, `pip install` the deps manually and it
still runs):

```python
!git clone https://github.com/radys-s/AlgoTrade.git && cd AlgoTrade && git checkout radys
%cd AlgoTrade
!pip install pydantic numpy pandas plotly
# upload round2.csv to data/ (it is not in git), then:
!python playground/run.py --data data/round2.csv --frac 0.25
```

## What the output means

- `PNL` is the score (cash + inventory at the final price − penalty).
- `penalty` should stay 0; the baseline cannot breach (inventory + resting orders ≤ `max_position`).
- `|inventory| quantiles`: how loaded we are. Baseline sits at 60 = full almost all the time (see ideas).
- `pnl per 10 % of time`: where you win and lose. The R2 tape crashes 450 ticks around 70-80 % of the way.
- `out/<name>_fills.csv`: every fill (time, side, qty, price, inventory after) for your own analysis in pandas.

## Parameters

| Param | Default | Meaning |
|---|---|---|
| `quote_size` | 40 | lots per side at the best price |
| `max_position` | 60 | hard cap on inventory + same-side resting lots (≤ 100 enforced) |
| `trend_fast_s` / `trend_slow_s` | 5 / 60 | time constants (s) of the fast and slow EMA of the mid |
| `trend_thr` | 3.0 | ticks: if fast − slow < −thr stop bidding, > +thr stop offering |
| `skew_lots` | 0 (off) | stop quoting the side that would push \|inventory\| above this |

## Ideas worth trying (rough order of expected value)

1. **Inventory is always at the cap (60).** Unload faster: when inventory is against the trend, quote the reducing
   side 1 tick inside the spread (if the spread is > 1 tick) or use `skew_lots`.
2. **Trend horizons.** R2 has 120-s return autocorrelation +0.44: try `trend_slow_s` 120-300, `trend_thr` 1-6.
3. **Ride the trend:** in a strong trend keep the WITH-trend side and let inventory build to `max_position` on purpose.
4. **Queue position:** cancel/resend loses queue. Port "parking" from `strategies/mm_ml_r1tuned.py`
   (keep the old order with `modify_order` down instead of cancelling) — that was worth +6 % in R1.
5. **Bigger size when calm:** `quote_size` up to 100 when |trend| is small, down when it is large.
6. **Use `trades`:** big aggressive sweeps (IOC) in `on_order_book_change(snapshot, trades)` often start a move.

Rule of thumb from R1: changes that *add* resting size won, rules that *remove* it lost. R2 is different
(trending), so check both tapes: a change is real if it helps R2 without killing R1.

Submitting: copy your file over `submission/best_strategy.py` only after comparing with the README table there.
