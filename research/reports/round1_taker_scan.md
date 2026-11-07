# Signal-taker proxy scan - `data/round1.csv`

1152 configs on 4,411,336 rows in 66.1 s (4 workers). Proxy model: decision at row i, IOC applied against the book after the last row before ts_i + 5 ms, 5 levels assumed at consecutive ticks, no self-impact. Ranked by train pnl (first 75% of time, 8 chunks); holdout = last 25%.

| # | signal | sign | thr | q | hold_ms | qty | slip | rev | pnl | train | wins | chunk_min | hold | orders | vol | max|I| |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ltp_dev | 1 | 1.5 | 0.99 | 15000 | 25 | 0 | 0 | 206 | 206 | 1/8 | 0 | 0 | 10 | 122 | 42 |
| 2 | ltp_dev | 1 | 1.5 | 0.99 | 15000 | 50 | 0 | 0 | 206 | 206 | 1/8 | 0 | 0 | 10 | 122 | 42 |
| 3 | ltp_dev | 1 | 1.5 | 0.99 | 15000 | 25 | 2 | 0 | 114 | 114 | 1/8 | -60 | 0 | 10 | 298 | 75 |
| 4 | ltp_dev | 1 | 1.5 | 0.99 | 15000 | 10 | 0 | 0 | 96 | 96 | 1/8 | 0 | 0 | 10 | 78 | 20 |
| 5 | ltp_dev | 1 | 1.5 | 0.99 | 15000 | 50 | 2 | 0 | 65 | 65 | 1/8 | -71 | 0 | 9 | 364 | 80 |
| 6 | ltp_dev | 1 | 1.5 | 0.99 | 15000 | 10 | 2 | 0 | 63 | 63 | 1/8 | -15 | 0 | 10 | 126 | 30 |
| 7 | ltp_dev | 1 | 1.5 | 0.99 | 5000 | 25 | 0 | 0 | 42 | 42 | 1/8 | 0 | 0 | 10 | 122 | 42 |
| 8 | ltp_dev | 1 | 1.5 | 0.99 | 5000 | 50 | 0 | 0 | 42 | 42 | 1/8 | 0 | 0 | 10 | 122 | 42 |
| 9 | ltp_dev | 1 | 1.5 | 0.99 | 5000 | 10 | 0 | 0 | 20 | 20 | 1/8 | 0 | 0 | 10 | 78 | 20 |
| 10 | ltp_dev | 1 | 1.5 | 0.99 | 5000 | 10 | 2 | 0 | -4 | -4 | 1/8 | -15 | 0 | 10 | 126 | 30 |
| 11 | ltp_dev | 1 | 1.5 | 0.99 | 2000 | 10 | 0 | 0 | -15 | -15 | 0/8 | -15 | 0 | 10 | 78 | 20 |
| 12 | ltp_dev | 1 | 1.5 | 0.99 | 2000 | 25 | 0 | 0 | -15 | -15 | 0/8 | -15 | 0 | 10 | 122 | 42 |
| 13 | ltp_dev | 1 | 1.5 | 0.99 | 2000 | 50 | 0 | 0 | -15 | -15 | 0/8 | -15 | 0 | 10 | 122 | 42 |
| 14 | ltp_dev | 1 | 1.5 | 0.99 | 500 | 10 | 0 | 0 | -35 | -35 | 0/8 | -20 | 0 | 10 | 78 | 20 |
| 15 | ltp_dev | 1 | 1.5 | 0.99 | 5000 | 25 | 2 | 0 | -42 | -42 | 1/8 | -60 | 0 | 10 | 298 | 75 |

## Confirm in the real simulator

1. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round1.csv --set '{"SIGNAL": "ltp_dev", "THRESHOLD": 1.5, "HOLD_US": 15000000, "QTY": 25, "SLIP_TICKS": 0}' --no-ledger`
2. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round1.csv --set '{"SIGNAL": "ltp_dev", "THRESHOLD": 1.5, "HOLD_US": 15000000, "QTY": 50, "SLIP_TICKS": 0}' --no-ledger`
3. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round1.csv --set '{"SIGNAL": "ltp_dev", "THRESHOLD": 1.5, "HOLD_US": 15000000, "QTY": 25, "SLIP_TICKS": 2}' --no-ledger`
4. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round1.csv --set '{"SIGNAL": "ltp_dev", "THRESHOLD": 1.5, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 0}' --no-ledger`
5. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round1.csv --set '{"SIGNAL": "ltp_dev", "THRESHOLD": 1.5, "HOLD_US": 15000000, "QTY": 50, "SLIP_TICKS": 2}' --no-ledger`
6. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round1.csv --set '{"SIGNAL": "ltp_dev", "THRESHOLD": 1.5, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 2}' --no-ledger`
7. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round1.csv --set '{"SIGNAL": "ltp_dev", "THRESHOLD": 1.5, "HOLD_US": 5000000, "QTY": 25, "SLIP_TICKS": 0}' --no-ledger`
8. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round1.csv --set '{"SIGNAL": "ltp_dev", "THRESHOLD": 1.5, "HOLD_US": 5000000, "QTY": 50, "SLIP_TICKS": 0}' --no-ledger`
9. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round1.csv --set '{"SIGNAL": "ltp_dev", "THRESHOLD": 1.5, "HOLD_US": 5000000, "QTY": 10, "SLIP_TICKS": 0}' --no-ledger`
10. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round1.csv --set '{"SIGNAL": "ltp_dev", "THRESHOLD": 1.5, "HOLD_US": 5000000, "QTY": 10, "SLIP_TICKS": 2}' --no-ledger`

## Best config per signal (by train pnl)

| signal | pnl | train | wins | hold | config |
|---|---:|---:|---:|---:|---|
| ltp_dev | 206 | 206 | 1 | 0 | `{"SIGNAL": "ltp_dev", "THRESHOLD": 1.5, "HOLD_US": 15000000, "QTY": 25, "SLIP_TICKS": 0}` |
| cnt_imb1 | -3,435 | -856 | 4 | -2,579 | `{"SIGNAL": "cnt_imb1", "THRESHOLD": 0.9523809523809523, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 0}` |
| flow_ms1000 | -9,211 | -6,838 | 0 | -2,373 | `{"SIGNAL": "flow_ms1000", "THRESHOLD": 2791.0, "HOLD_US": 2000000, "QTY": 10, "SLIP_TICKS": 0}` |
| imb1 | -14,872 | -9,547 | 1 | -5,325 | `{"SIGNAL": "imb1", "THRESHOLD": 0.997667638483965, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 0}` |
| flow_ev100 | -22,281 | -14,718 | 1 | -7,563 | `{"SIGNAL": "flow_ev100", "THRESHOLD": 1723.0, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 0}` |
| flow_ms100 | -29,003 | -21,461 | 0 | -7,542 | `{"SIGNAL": "flow_ms100", "THRESHOLD": 1286.0, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 0}` |
| ret_ms1000 | -28,982 | -24,204 | 0 | -4,778 | `{"SIGNAL": "ret_ms1000", "THRESHOLD": 2.5, "HOLD_US": 5000000, "QTY": 10, "SLIP_TICKS": 0}` |
| flow_ev10 | -36,138 | -27,043 | 0 | -9,095 | `{"SIGNAL": "flow_ev10", "THRESHOLD": 586.0, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 0}` |
| micro | -43,366 | -30,405 | 1 | -12,961 | `{"SIGNAL": "micro", "THRESHOLD": 0.4512987012987013, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 0}` |
| flow_ev1 | -47,469 | -31,295 | 0 | -16,174 | `{"SIGNAL": "flow_ev1", "THRESHOLD": 67.0, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 2}` |
| imb5 | -42,653 | -31,413 | 0 | -11,240 | `{"SIGNAL": "imb5", "THRESHOLD": 0.8743336604525075, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 0}` |
| ret_ms100 | -45,380 | -37,983 | 0 | -7,397 | `{"SIGNAL": "ret_ms100", "THRESHOLD": 1.0, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 0}` |
