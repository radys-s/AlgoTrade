# Signal-taker proxy scan - `data/sample_v2.csv`

24 configs on 2,000,000 rows in 10.2 s (8 workers). Proxy model: decision at row i, IOC applied against the book after the last row before ts_i + 5 ms, 5 levels assumed at consecutive ticks, no self-impact. Ranked by train pnl (first 75% of time, 8 chunks); holdout = last 25%.

| # | signal | sign | thr | q | hold_ms | qty | slip | rev | pnl | train | wins | chunk_min | hold | orders | vol | max|I| |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | flow_ms1000 | -1 | 1538 | 0.99 | 5000 | 10 | 0 | 0 | -14,519 | -10,039 | 0/8 | -2,446 | -4,480 | 1049 | 18320 | 80 |
| 2 | flow_ms1000 | 1 | 1538 | 0.99 | 5000 | 10 | 0 | 0 | -18,103 | -13,770 | 0/8 | -2,730 | -4,333 | 1068 | 18340 | 80 |
| 3 | flow_ms1000 | -1 | 1538 | 0.99 | 500 | 10 | 0 | 0 | -24,699 | -17,936 | 0/8 | -2,910 | -6,763 | 1754 | 30000 | 80 |

## Confirm in the real simulator

1. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/sample_v2.csv --set '{"SIGNAL": "flow_ms1000", "THRESHOLD": 1538.0, "HOLD_US": 5000000, "QTY": 10, "SLIP_TICKS": 0, "SIGN": -1}' --no-ledger`
2. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/sample_v2.csv --set '{"SIGNAL": "flow_ms1000", "THRESHOLD": 1538.0, "HOLD_US": 5000000, "QTY": 10, "SLIP_TICKS": 0}' --no-ledger`
3. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/sample_v2.csv --set '{"SIGNAL": "flow_ms1000", "THRESHOLD": 1538.0, "HOLD_US": 500000, "QTY": 10, "SLIP_TICKS": 0, "SIGN": -1}' --no-ledger`

## Best config per signal (by train pnl)

| signal | pnl | train | wins | hold | config |
|---|---:|---:|---:|---:|---|
| flow_ms1000 | -14,519 | -10,039 | 0 | -4,480 | `{"SIGNAL": "flow_ms1000", "THRESHOLD": 1538.0, "HOLD_US": 5000000, "QTY": 10, "SLIP_TICKS": 0, "SIGN": -1}` |
| micro | -39,707 | -30,449 | 0 | -9,258 | `{"SIGNAL": "micro", "THRESHOLD": 0.8466752507994226, "HOLD_US": 5000000, "QTY": 10, "SLIP_TICKS": 0}` |
| flow_ev10 | -76,375 | -57,210 | 0 | -19,165 | `{"SIGNAL": "flow_ev10", "THRESHOLD": 327.0, "HOLD_US": 5000000, "QTY": 10, "SLIP_TICKS": 0, "SIGN": -1}` |
