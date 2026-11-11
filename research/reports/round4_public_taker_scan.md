# Signal-taker proxy scan - `../AlgoTradeArena/data/round4_public.csv`

1152 configs on 2,398,520 rows in 57.0 s (3 workers). Proxy model: decision at row i, IOC applied against the book after the last row before ts_i + 5 ms, 5 levels assumed at consecutive ticks, no self-impact. Ranked by train pnl (first 75% of time, 8 chunks); holdout = last 25%.

| # | signal | sign | thr | q | hold_ms | qty | slip | rev | pnl | train | wins | chunk_min | hold | orders | vol | max|I| |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ret_ms1000 | 1 | 5 | 0.8 | 2000 | 50 | 0 | 0 | 338,683 | 272,898 | 8/8 | 6,026 | 65,785 | 3452 | 117756 | 80 |
| 2 | ret_ms1000 | 1 | 5 | 0.8 | 2000 | 25 | 0 | 0 | 337,714 | 272,079 | 8/8 | 5,917 | 65,635 | 4788 | 117466 | 80 |
| 3 | ret_ms100 | 1 | 1.5 | 0.9 | 2000 | 25 | 0 | 0 | 353,801 | 269,043 | 8/8 | 6,090 | 84,758 | 5696 | 153522 | 80 |
| 4 | ret_ms100 | 1 | 1.5 | 0.9 | 2000 | 50 | 0 | 0 | 354,093 | 268,920 | 8/8 | 2,294 | 85,173 | 3876 | 155702 | 80 |
| 5 | ret_ms1000 | 1 | 5 | 0.8 | 2000 | 25 | 2 | 0 | 329,052 | 268,793 | 8/8 | 4,697 | 60,259 | 3719 | 117976 | 80 |
| 6 | ret_ms1000 | 1 | 5 | 0.8 | 2000 | 10 | 0 | 0 | 335,310 | 268,696 | 8/8 | 5,692 | 66,614 | 7769 | 116640 | 80 |
| 7 | ret_ms1000 | 1 | 5 | 0.8 | 2000 | 50 | 2 | 0 | 327,692 | 267,783 | 8/8 | 4,838 | 59,909 | 2265 | 118386 | 80 |
| 8 | ret_ms1000 | 1 | 5 | 0.8 | 2000 | 10 | 2 | 0 | 328,570 | 266,972 | 8/8 | 4,408 | 61,598 | 6637 | 116966 | 80 |
| 9 | ret_ms100 | 1 | 1.5 | 0.9 | 2000 | 10 | 0 | 0 | 349,282 | 266,047 | 8/8 | 9,275 | 83,235 | 9261 | 146202 | 80 |
| 10 | ret_ms100 | 1 | 1.5 | 0.9 | 2000 | 50 | 2 | 0 | 346,089 | 262,052 | 8/8 | 1,122 | 84,037 | 2990 | 156280 | 80 |
| 11 | ret_ms100 | 1 | 1.5 | 0.9 | 2000 | 25 | 2 | 0 | 345,052 | 261,964 | 8/8 | 5,180 | 83,088 | 4854 | 154200 | 80 |
| 12 | ret_ms100 | 1 | 1 | 0.8 | 2000 | 10 | 0 | 0 | 333,515 | 259,032 | 8/8 | 6,613 | 74,483 | 11288 | 177778 | 80 |
| 13 | ret_ms100 | 1 | 1.5 | 0.9 | 2000 | 10 | 2 | 0 | 340,222 | 258,748 | 8/8 | 8,593 | 81,474 | 8379 | 146748 | 80 |
| 14 | ret_ms100 | 1 | 1 | 0.8 | 2000 | 25 | 0 | 0 | 331,174 | 256,278 | 8/8 | 1,032 | 74,896 | 6937 | 186104 | 80 |
| 15 | ret_ms100 | 1 | 1 | 0.8 | 2000 | 10 | 2 | 0 | 324,692 | 253,640 | 8/8 | 4,548 | 71,052 | 10190 | 178520 | 80 |
| 16 | ret_ms100 | 1 | 1 | 0.8 | 2000 | 25 | 2 | 0 | 323,076 | 252,117 | 7/8 | -580 | 70,959 | 5892 | 187000 | 80 |
| 17 | ret_ms100 | 1 | 1 | 0.8 | 2000 | 50 | 0 | 0 | 325,287 | 249,422 | 8/8 | 365 | 75,865 | 4732 | 189528 | 80 |
| 18 | ret_ms1000 | 1 | 9 | 0.9 | 2000 | 50 | 0 | 0 | 308,873 | 246,727 | 8/8 | 3,919 | 62,146 | 2174 | 76608 | 80 |
| 19 | ret_ms100 | 1 | 1 | 0.8 | 2000 | 50 | 2 | 0 | 317,687 | 246,333 | 7/8 | -841 | 71,354 | 3632 | 190360 | 80 |
| 20 | ret_ms1000 | 1 | 9 | 0.9 | 2000 | 25 | 0 | 0 | 305,716 | 244,411 | 8/8 | 3,875 | 61,305 | 3046 | 76428 | 80 |

## Confirm in the real simulator

1. `uv run python -m research.run --strategy strategies/signal_taker.py --data ../AlgoTradeArena/data/round4_public.csv --set '{"SIGNAL": "ret_ms1000", "THRESHOLD": 5.0, "HOLD_US": 2000000, "QTY": 50, "SLIP_TICKS": 0}' --no-ledger`
2. `uv run python -m research.run --strategy strategies/signal_taker.py --data ../AlgoTradeArena/data/round4_public.csv --set '{"SIGNAL": "ret_ms1000", "THRESHOLD": 5.0, "HOLD_US": 2000000, "QTY": 25, "SLIP_TICKS": 0}' --no-ledger`
3. `uv run python -m research.run --strategy strategies/signal_taker.py --data ../AlgoTradeArena/data/round4_public.csv --set '{"SIGNAL": "ret_ms100", "THRESHOLD": 1.5, "HOLD_US": 2000000, "QTY": 25, "SLIP_TICKS": 0}' --no-ledger`
4. `uv run python -m research.run --strategy strategies/signal_taker.py --data ../AlgoTradeArena/data/round4_public.csv --set '{"SIGNAL": "ret_ms100", "THRESHOLD": 1.5, "HOLD_US": 2000000, "QTY": 50, "SLIP_TICKS": 0}' --no-ledger`
5. `uv run python -m research.run --strategy strategies/signal_taker.py --data ../AlgoTradeArena/data/round4_public.csv --set '{"SIGNAL": "ret_ms1000", "THRESHOLD": 5.0, "HOLD_US": 2000000, "QTY": 25, "SLIP_TICKS": 2}' --no-ledger`
6. `uv run python -m research.run --strategy strategies/signal_taker.py --data ../AlgoTradeArena/data/round4_public.csv --set '{"SIGNAL": "ret_ms1000", "THRESHOLD": 5.0, "HOLD_US": 2000000, "QTY": 10, "SLIP_TICKS": 0}' --no-ledger`
7. `uv run python -m research.run --strategy strategies/signal_taker.py --data ../AlgoTradeArena/data/round4_public.csv --set '{"SIGNAL": "ret_ms1000", "THRESHOLD": 5.0, "HOLD_US": 2000000, "QTY": 50, "SLIP_TICKS": 2}' --no-ledger`
8. `uv run python -m research.run --strategy strategies/signal_taker.py --data ../AlgoTradeArena/data/round4_public.csv --set '{"SIGNAL": "ret_ms1000", "THRESHOLD": 5.0, "HOLD_US": 2000000, "QTY": 10, "SLIP_TICKS": 2}' --no-ledger`
9. `uv run python -m research.run --strategy strategies/signal_taker.py --data ../AlgoTradeArena/data/round4_public.csv --set '{"SIGNAL": "ret_ms100", "THRESHOLD": 1.5, "HOLD_US": 2000000, "QTY": 10, "SLIP_TICKS": 0}' --no-ledger`
10. `uv run python -m research.run --strategy strategies/signal_taker.py --data ../AlgoTradeArena/data/round4_public.csv --set '{"SIGNAL": "ret_ms100", "THRESHOLD": 1.5, "HOLD_US": 2000000, "QTY": 50, "SLIP_TICKS": 2}' --no-ledger`

## Best config per signal (by train pnl)

| signal | pnl | train | wins | hold | config |
|---|---:|---:|---:|---:|---|
| ret_ms1000 | 338,683 | 272,898 | 8 | 65,785 | `{"SIGNAL": "ret_ms1000", "THRESHOLD": 5.0, "HOLD_US": 2000000, "QTY": 50, "SLIP_TICKS": 0}` |
| ret_ms100 | 353,801 | 269,043 | 8 | 84,758 | `{"SIGNAL": "ret_ms100", "THRESHOLD": 1.5, "HOLD_US": 2000000, "QTY": 25, "SLIP_TICKS": 0}` |
| flow_ms1000 | 240,828 | 191,135 | 7 | 49,693 | `{"SIGNAL": "flow_ms1000", "THRESHOLD": 1238.0, "HOLD_US": 2000000, "QTY": 50, "SLIP_TICKS": 0}` |
| flow_ev100 | 246,194 | 182,985 | 8 | 63,209 | `{"SIGNAL": "flow_ev100", "THRESHOLD": 660.0, "HOLD_US": 2000000, "QTY": 50, "SLIP_TICKS": 0}` |
| flow_ms100 | 189,142 | 164,718 | 7 | 24,424 | `{"SIGNAL": "flow_ms100", "THRESHOLD": 417.0, "HOLD_US": 5000000, "QTY": 50, "SLIP_TICKS": 0}` |
| cnt_imb1 | 190,731 | 150,435 | 8 | 40,296 | `{"SIGNAL": "cnt_imb1", "THRESHOLD": 0.8518518518518519, "HOLD_US": 2000000, "QTY": 25, "SLIP_TICKS": 0}` |
| flow_ev10 | 189,055 | 147,622 | 7 | 41,433 | `{"SIGNAL": "flow_ev10", "THRESHOLD": 592.0, "HOLD_US": 2000000, "QTY": 50, "SLIP_TICKS": 0}` |
| imb5 | 159,893 | 132,001 | 8 | 27,892 | `{"SIGNAL": "imb5", "THRESHOLD": 0.6579326672182028, "HOLD_US": 2000000, "QTY": 25, "SLIP_TICKS": 0}` |
| flow_ev1 | 83,681 | 85,632 | 7 | -1,951 | `{"SIGNAL": "flow_ev1", "THRESHOLD": 78.0, "HOLD_US": 5000000, "QTY": 50, "SLIP_TICKS": 2}` |
| imb1 | 29,823 | 27,821 | 5 | 2,002 | `{"SIGNAL": "imb1", "THRESHOLD": 0.9607843137254902, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 2}` |
| ltp_dev | 23,093 | 23,571 | 5 | -478 | `{"SIGNAL": "ltp_dev", "THRESHOLD": 2.0, "HOLD_US": 5000000, "QTY": 50, "SLIP_TICKS": 0}` |
| micro | 7,986 | 14,061 | 3 | -6,075 | `{"SIGNAL": "micro", "THRESHOLD": 0.599750919798754, "HOLD_US": 5000000, "QTY": 10, "SLIP_TICKS": 2}` |
