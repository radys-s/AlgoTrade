# Signal-taker proxy scan - `data/round3_public.csv`

1152 configs on 2,412,805 rows in 30.1 s (6 workers). Proxy model: decision at row i, IOC applied against the book after the last row before ts_i + 5 ms, 5 levels assumed at consecutive ticks, no self-impact. Ranked by train pnl (first 75% of time, 8 chunks); holdout = last 25%.

| # | signal | sign | thr | q | hold_ms | qty | slip | rev | pnl | train | wins | chunk_min | hold | orders | vol | max|I| |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | flow_ev100 | 1 | 592 | 0.9 | 2000 | 50 | 0 | 0 | 80,911 | 65,459 | 6/8 | -1,881 | 15,452 | 6287 | 265952 | 80 |
| 2 | flow_ev100 | 1 | 592 | 0.9 | 2000 | 25 | 0 | 0 | 79,059 | 63,032 | 6/8 | -2,101 | 16,027 | 9443 | 263828 | 80 |
| 3 | ret_ms100 | 1 | 1 | 0.9 | 2000 | 25 | 0 | 0 | 74,486 | 61,093 | 7/8 | -1,993 | 13,393 | 6643 | 174828 | 80 |
| 4 | ret_ms100 | 1 | 2 | 0.97 | 2000 | 50 | 0 | 0 | 80,443 | 60,898 | 7/8 | -5,074 | 19,545 | 2294 | 86732 | 80 |
| 5 | ret_ms100 | 1 | 2 | 0.97 | 2000 | 25 | 0 | 0 | 80,904 | 60,407 | 7/8 | -4,944 | 20,497 | 3255 | 84278 | 80 |
| 6 | ret_ms100 | 1 | 1 | 0.9 | 2000 | 50 | 0 | 0 | 73,147 | 60,229 | 6/8 | -1,249 | 12,918 | 4604 | 177678 | 80 |
| 7 | flow_ev100 | 1 | 592 | 0.9 | 2000 | 10 | 0 | 0 | 75,794 | 59,413 | 6/8 | -1,940 | 16,381 | 15728 | 255130 | 80 |
| 8 | flow_ms100 | 1 | 866 | 0.97 | 2000 | 50 | 0 | 0 | 86,696 | 57,142 | 8/8 | 88 | 29,554 | 3625 | 152062 | 80 |
| 9 | flow_ms100 | 1 | 866 | 0.97 | 2000 | 25 | 0 | 0 | 86,032 | 56,388 | 8/8 | 61 | 29,644 | 5433 | 150560 | 80 |
| 10 | flow_ms100 | 1 | 387 | 0.9 | 2000 | 50 | 0 | 0 | 57,611 | 56,156 | 7/8 | -4,468 | 1,455 | 7221 | 309842 | 80 |
| 11 | flow_ev100 | 1 | 1086 | 0.97 | 2000 | 50 | 0 | 0 | 83,568 | 55,803 | 8/8 | 314 | 27,765 | 3099 | 123728 | 80 |
| 12 | ret_ms100 | 1 | 1 | 0.9 | 2000 | 10 | 0 | 0 | 69,090 | 55,364 | 7/8 | -4,231 | 13,726 | 10722 | 166856 | 80 |
| 13 | ret_ms100 | 1 | 2 | 0.97 | 2000 | 50 | 2 | 0 | 73,056 | 54,909 | 7/8 | -6,374 | 18,147 | 1693 | 86976 | 80 |
| 14 | flow_ms100 | 1 | 866 | 0.97 | 2000 | 10 | 0 | 0 | 83,570 | 54,704 | 8/8 | 225 | 28,866 | 8968 | 144690 | 80 |
| 15 | flow_ms1000 | 1 | 1502 | 0.9 | 2000 | 50 | 0 | 0 | 71,679 | 54,635 | 7/8 | -1,223 | 17,044 | 3328 | 134468 | 80 |
| 16 | ret_ms100 | 1 | 2 | 0.97 | 2000 | 25 | 2 | 0 | 72,473 | 54,327 | 7/8 | -6,283 | 18,146 | 2692 | 84606 | 80 |
| 17 | flow_ev100 | 1 | 592 | 0.9 | 2000 | 50 | 2 | 0 | 64,664 | 53,915 | 6/8 | -4,318 | 10,749 | 5049 | 266580 | 80 |
| 18 | flow_ms1000 | 1 | 1502 | 0.9 | 2000 | 25 | 0 | 0 | 70,278 | 53,358 | 7/8 | -1,313 | 16,920 | 4906 | 133576 | 80 |
| 19 | flow_ev100 | 1 | 1086 | 0.97 | 2000 | 25 | 0 | 0 | 80,716 | 52,994 | 8/8 | 434 | 27,722 | 4552 | 122026 | 80 |
| 20 | flow_ev100 | 1 | 592 | 0.9 | 2000 | 25 | 2 | 0 | 65,034 | 52,831 | 6/8 | -4,043 | 12,203 | 8298 | 264374 | 80 |

## Confirm in the real simulator

1. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round3_public.csv --set '{"SIGNAL": "flow_ev100", "THRESHOLD": 592.0, "HOLD_US": 2000000, "QTY": 50, "SLIP_TICKS": 0}' --no-ledger`
2. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round3_public.csv --set '{"SIGNAL": "flow_ev100", "THRESHOLD": 592.0, "HOLD_US": 2000000, "QTY": 25, "SLIP_TICKS": 0}' --no-ledger`
3. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round3_public.csv --set '{"SIGNAL": "ret_ms100", "THRESHOLD": 1.0, "HOLD_US": 2000000, "QTY": 25, "SLIP_TICKS": 0}' --no-ledger`
4. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round3_public.csv --set '{"SIGNAL": "ret_ms100", "THRESHOLD": 2.0, "HOLD_US": 2000000, "QTY": 50, "SLIP_TICKS": 0}' --no-ledger`
5. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round3_public.csv --set '{"SIGNAL": "ret_ms100", "THRESHOLD": 2.0, "HOLD_US": 2000000, "QTY": 25, "SLIP_TICKS": 0}' --no-ledger`
6. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round3_public.csv --set '{"SIGNAL": "ret_ms100", "THRESHOLD": 1.0, "HOLD_US": 2000000, "QTY": 50, "SLIP_TICKS": 0}' --no-ledger`
7. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round3_public.csv --set '{"SIGNAL": "flow_ev100", "THRESHOLD": 592.0, "HOLD_US": 2000000, "QTY": 10, "SLIP_TICKS": 0}' --no-ledger`
8. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round3_public.csv --set '{"SIGNAL": "flow_ms100", "THRESHOLD": 866.0, "HOLD_US": 2000000, "QTY": 50, "SLIP_TICKS": 0}' --no-ledger`
9. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round3_public.csv --set '{"SIGNAL": "flow_ms100", "THRESHOLD": 866.0, "HOLD_US": 2000000, "QTY": 25, "SLIP_TICKS": 0}' --no-ledger`
10. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/round3_public.csv --set '{"SIGNAL": "flow_ms100", "THRESHOLD": 387.0, "HOLD_US": 2000000, "QTY": 50, "SLIP_TICKS": 0}' --no-ledger`

## Best config per signal (by train pnl)

| signal | pnl | train | wins | hold | config |
|---|---:|---:|---:|---:|---|
| flow_ev100 | 80,911 | 65,459 | 6 | 15,452 | `{"SIGNAL": "flow_ev100", "THRESHOLD": 592.0, "HOLD_US": 2000000, "QTY": 50, "SLIP_TICKS": 0}` |
| ret_ms100 | 74,486 | 61,093 | 7 | 13,393 | `{"SIGNAL": "ret_ms100", "THRESHOLD": 1.0, "HOLD_US": 2000000, "QTY": 25, "SLIP_TICKS": 0}` |
| flow_ms100 | 86,696 | 57,142 | 8 | 29,554 | `{"SIGNAL": "flow_ms100", "THRESHOLD": 866.0, "HOLD_US": 2000000, "QTY": 50, "SLIP_TICKS": 0}` |
| flow_ms1000 | 71,679 | 54,635 | 7 | 17,044 | `{"SIGNAL": "flow_ms1000", "THRESHOLD": 1502.0, "HOLD_US": 2000000, "QTY": 50, "SLIP_TICKS": 0}` |
| ret_ms1000 | 33,957 | 34,064 | 7 | -107 | `{"SIGNAL": "ret_ms1000", "THRESHOLD": 5.5, "HOLD_US": 2000000, "QTY": 25, "SLIP_TICKS": 0}` |
| cnt_imb1 | 35,411 | 32,527 | 6 | 2,884 | `{"SIGNAL": "cnt_imb1", "THRESHOLD": 0.8620689655172413, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 0}` |
| flow_ev10 | 46,987 | 32,249 | 6 | 14,738 | `{"SIGNAL": "flow_ev10", "THRESHOLD": 552.0, "HOLD_US": 2000000, "QTY": 50, "SLIP_TICKS": 0}` |
| flow_ev1 | 25,847 | 20,976 | 5 | 4,871 | `{"SIGNAL": "flow_ev1", "THRESHOLD": 66.0, "HOLD_US": 15000000, "QTY": 50, "SLIP_TICKS": 2}` |
| micro | -3,645 | 11,721 | 5 | -15,366 | `{"SIGNAL": "micro", "THRESHOLD": 0.47953216374269003, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 2}` |
| imb1 | -23,469 | 5,278 | 5 | -28,747 | `{"SIGNAL": "imb1", "THRESHOLD": 0.898464431212786, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 0}` |
| imb5 | -21,292 | -10,259 | 4 | -11,033 | `{"SIGNAL": "imb5", "THRESHOLD": 0.767555323227043, "HOLD_US": 2000000, "QTY": 10, "SLIP_TICKS": 0}` |
| ltp_dev | -28,846 | -11,736 | 2 | -17,110 | `{"SIGNAL": "ltp_dev", "THRESHOLD": 0.5, "HOLD_US": 15000000, "QTY": 50, "SLIP_TICKS": 0}` |
