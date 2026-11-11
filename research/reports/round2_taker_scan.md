# Signal-taker proxy scan - `data/round2.csv`

1152 configs on 2,193,064 rows in 16.4 s (14 workers). Proxy model: decision at row i, IOC applied against the book after the last row before ts_i + 5 ms, 5 levels assumed at consecutive ticks, no self-impact. Ranked by train pnl (first 75% of time, 8 chunks); holdout = last 25%.

| # | signal | sign | thr | q | hold_ms | qty | slip | rev | pnl | train | wins | chunk_min | hold | orders | vol | max|I| |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ret_ms100 | 1 | 1 | 0.9 | 5000 | 50 | 0 | 0 | 19,723 | 24,063 | 6/8 | -3,642 | -4,340 | 2042 | 79332 | 80 |
| 2 | ret_ms100 | 1 | 1 | 0.9 | 5000 | 25 | 0 | 0 | 19,317 | 22,750 | 5/8 | -3,562 | -3,433 | 2905 | 77592 | 80 |
| 3 | flow_ev100 | 1 | 1647 | 0.99 | 15000 | 50 | 2 | 0 | -458 | 20,241 | 6/8 | -2,303 | -20,699 | 581 | 29254 | 80 |
| 4 | ret_ms100 | 1 | 1 | 0.9 | 5000 | 10 | 0 | 0 | 18,040 | 19,591 | 4/8 | -3,520 | -1,551 | 4696 | 72614 | 80 |
| 5 | flow_ev100 | 1 | 1647 | 0.99 | 15000 | 25 | 2 | 0 | -1,327 | 19,400 | 6/8 | -2,206 | -20,727 | 922 | 29064 | 80 |
| 6 | flow_ev100 | 1 | 1647 | 0.99 | 15000 | 50 | 0 | 0 | -1,772 | 18,301 | 5/8 | -2,200 | -20,073 | 749 | 28856 | 80 |
| 7 | flow_ev100 | 1 | 1647 | 0.99 | 15000 | 25 | 0 | 0 | -1,824 | 18,294 | 5/8 | -2,075 | -20,118 | 1093 | 28682 | 80 |
| 8 | flow_ev100 | 1 | 381 | 0.8 | 15000 | 50 | 0 | 0 | 44,528 | 18,059 | 6/8 | -5,832 | 26,469 | 1387 | 61040 | 80 |
| 9 | flow_ev100 | 1 | 381 | 0.8 | 15000 | 10 | 0 | 0 | 43,942 | 17,667 | 6/8 | -5,895 | 26,275 | 3696 | 61040 | 80 |
| 10 | flow_ev100 | 1 | 381 | 0.8 | 15000 | 25 | 0 | 0 | 44,122 | 17,623 | 6/8 | -5,843 | 26,499 | 2139 | 61040 | 80 |
| 11 | flow_ms100 | 1 | 352 | 0.9 | 5000 | 50 | 0 | 0 | 4,815 | 17,560 | 5/8 | -4,686 | -12,745 | 3610 | 161600 | 80 |
| 12 | flow_ev100 | 1 | 381 | 0.8 | 15000 | 10 | 2 | 0 | 43,377 | 17,490 | 6/8 | -6,028 | 25,887 | 3460 | 61040 | 80 |
| 13 | flow_ev100 | 1 | 381 | 0.8 | 15000 | 50 | 2 | 0 | 43,140 | 17,203 | 6/8 | -6,076 | 25,937 | 1165 | 61040 | 80 |
| 14 | flow_ev10 | 1 | 603 | 0.99 | 15000 | 50 | 2 | 0 | 4,166 | 17,163 | 6/8 | -7,482 | -12,997 | 1054 | 55200 | 80 |
| 15 | flow_ev100 | 1 | 381 | 0.8 | 15000 | 25 | 2 | 0 | 43,160 | 17,124 | 6/8 | -6,052 | 26,036 | 1923 | 61040 | 80 |
| 16 | flow_ev100 | 1 | 1647 | 0.99 | 15000 | 10 | 2 | 0 | -3,594 | 16,830 | 5/8 | -2,105 | -20,424 | 1608 | 28070 | 80 |
| 17 | flow_ev100 | 1 | 1647 | 0.99 | 15000 | 10 | 0 | 0 | -3,140 | 16,755 | 5/8 | -2,020 | -19,895 | 1759 | 27806 | 80 |
| 18 | flow_ms100 | 1 | 352 | 0.9 | 5000 | 25 | 0 | 0 | 4,860 | 15,925 | 5/8 | -4,715 | -11,065 | 5591 | 161380 | 80 |
| 19 | flow_ev10 | 1 | 603 | 0.99 | 15000 | 50 | 0 | 0 | 2,635 | 15,112 | 6/8 | -8,741 | -12,477 | 1271 | 55124 | 80 |
| 20 | flow_ms100 | 1 | 352 | 0.9 | 5000 | 10 | 0 | 0 | 5,596 | 14,628 | 5/8 | -4,275 | -9,032 | 9704 | 160318 | 80 |

## Confirm in the real simulator

1. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round2.csv --set '{"SIGNAL": "ret_ms100", "THRESHOLD": 1.0, "HOLD_US": 5000000, "QTY": 50, "SLIP_TICKS": 0}' --no-ledger`
2. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round2.csv --set '{"SIGNAL": "ret_ms100", "THRESHOLD": 1.0, "HOLD_US": 5000000, "QTY": 25, "SLIP_TICKS": 0}' --no-ledger`
3. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round2.csv --set '{"SIGNAL": "flow_ev100", "THRESHOLD": 1647.0, "HOLD_US": 15000000, "QTY": 50, "SLIP_TICKS": 2}' --no-ledger`
4. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round2.csv --set '{"SIGNAL": "ret_ms100", "THRESHOLD": 1.0, "HOLD_US": 5000000, "QTY": 10, "SLIP_TICKS": 0}' --no-ledger`
5. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round2.csv --set '{"SIGNAL": "flow_ev100", "THRESHOLD": 1647.0, "HOLD_US": 15000000, "QTY": 25, "SLIP_TICKS": 2}' --no-ledger`
6. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round2.csv --set '{"SIGNAL": "flow_ev100", "THRESHOLD": 1647.0, "HOLD_US": 15000000, "QTY": 50, "SLIP_TICKS": 0}' --no-ledger`
7. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round2.csv --set '{"SIGNAL": "flow_ev100", "THRESHOLD": 1647.0, "HOLD_US": 15000000, "QTY": 25, "SLIP_TICKS": 0}' --no-ledger`
8. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round2.csv --set '{"SIGNAL": "flow_ev100", "THRESHOLD": 381.0, "HOLD_US": 15000000, "QTY": 50, "SLIP_TICKS": 0}' --no-ledger`
9. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round2.csv --set '{"SIGNAL": "flow_ev100", "THRESHOLD": 381.0, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 0}' --no-ledger`
10. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round2.csv --set '{"SIGNAL": "flow_ev100", "THRESHOLD": 381.0, "HOLD_US": 15000000, "QTY": 25, "SLIP_TICKS": 0}' --no-ledger`

## Best config per signal (by train pnl)

| signal | pnl | train | wins | hold | config |
|---|---:|---:|---:|---:|---|
| ret_ms100 | 19,723 | 24,063 | 6 | -4,340 | `{"SIGNAL": "ret_ms100", "THRESHOLD": 1.0, "HOLD_US": 5000000, "QTY": 50, "SLIP_TICKS": 0}` |
| flow_ev100 | -458 | 20,241 | 6 | -20,699 | `{"SIGNAL": "flow_ev100", "THRESHOLD": 1647.0, "HOLD_US": 15000000, "QTY": 50, "SLIP_TICKS": 2}` |
| flow_ms100 | 4,815 | 17,560 | 5 | -12,745 | `{"SIGNAL": "flow_ms100", "THRESHOLD": 352.0, "HOLD_US": 5000000, "QTY": 50, "SLIP_TICKS": 0}` |
| flow_ev10 | 4,166 | 17,163 | 6 | -12,997 | `{"SIGNAL": "flow_ev10", "THRESHOLD": 603.0, "HOLD_US": 15000000, "QTY": 50, "SLIP_TICKS": 2}` |
| flow_ms1000 | -6,818 | 11,867 | 5 | -18,685 | `{"SIGNAL": "flow_ms1000", "THRESHOLD": 2064.0, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 0}` |
| ret_ms1000 | 3,662 | 9,586 | 4 | -5,924 | `{"SIGNAL": "ret_ms1000", "THRESHOLD": 12.5, "HOLD_US": 5000000, "QTY": 25, "SLIP_TICKS": 0}` |
| imb5 | -11,831 | -1,114 | 3 | -10,717 | `{"SIGNAL": "imb5", "THRESHOLD": 0.6657199632717592, "HOLD_US": 15000000, "QTY": 50, "SLIP_TICKS": 0}` |
| micro | -6,529 | -1,194 | 4 | -5,335 | `{"SIGNAL": "micro", "THRESHOLD": 1.278082191780822, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 2}` |
| ltp_dev | -18,339 | -3,942 | 3 | -14,397 | `{"SIGNAL": "ltp_dev", "THRESHOLD": 2.0, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 2}` |
| cnt_imb1 | -4,382 | -3,991 | 2 | -391 | `{"SIGNAL": "cnt_imb1", "THRESHOLD": 0.9487179487179487, "HOLD_US": 15000000, "QTY": 25, "SLIP_TICKS": 0}` |
| imb1 | 163 | -5,146 | 3 | 5,309 | `{"SIGNAL": "imb1", "THRESHOLD": 0.9972807613868117, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 0}` |
| flow_ev1 | -24,319 | -10,531 | 4 | -13,788 | `{"SIGNAL": "flow_ev1", "THRESHOLD": 0.0, "HOLD_US": 15000000, "QTY": 25, "SLIP_TICKS": 0}` |
