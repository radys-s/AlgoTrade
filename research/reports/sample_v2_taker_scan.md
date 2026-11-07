# Signal-taker proxy scan - `data/sample_v2.csv`

1152 configs on 2,000,000 rows in 43.0 s (4 workers). Proxy model: decision at row i, IOC applied against the book after the last row before ts_i + 5 ms, 5 levels assumed at consecutive ticks, no self-impact. Ranked by train pnl (first 75% of time, 8 chunks); holdout = last 25%.

| # | signal | sign | thr | q | hold_ms | qty | slip | rev | pnl | train | wins | chunk_min | hold | orders | vol | max|I| |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ltp_dev | 1 | 1.5 | 0.97 | 500 | 10 | 0 | 0 | -1,380 | -870 | 0/8 | -320 | -510 | 107 | 1780 | 80 |
| 2 | ltp_dev | 1 | 1.5 | 0.99 | 500 | 10 | 0 | 0 | -1,380 | -870 | 0/8 | -320 | -510 | 107 | 1780 | 80 |
| 3 | ltp_dev | 1 | 1.5 | 0.97 | 5000 | 10 | 0 | 0 | -1,390 | -890 | 0/8 | -240 | -500 | 114 | 1780 | 80 |
| 4 | ltp_dev | 1 | 1.5 | 0.99 | 5000 | 10 | 0 | 0 | -1,390 | -890 | 0/8 | -240 | -500 | 114 | 1780 | 80 |
| 5 | ltp_dev | 1 | 1.5 | 0.97 | 500 | 10 | 2 | 0 | -1,457 | -947 | 0/8 | -320 | -510 | 107 | 1820 | 80 |
| 6 | ltp_dev | 1 | 1.5 | 0.99 | 500 | 10 | 2 | 0 | -1,457 | -947 | 0/8 | -320 | -510 | 107 | 1820 | 80 |
| 7 | ltp_dev | 1 | 1.5 | 0.97 | 2000 | 10 | 0 | 0 | -1,520 | -950 | 0/8 | -400 | -570 | 109 | 1780 | 80 |
| 8 | ltp_dev | 1 | 1.5 | 0.97 | 5000 | 10 | 2 | 0 | -1,503 | -950 | 0/8 | -260 | -553 | 107 | 1820 | 80 |
| 9 | ltp_dev | 1 | 1.5 | 0.99 | 2000 | 10 | 0 | 0 | -1,520 | -950 | 0/8 | -400 | -570 | 109 | 1780 | 80 |
| 10 | ltp_dev | 1 | 1.5 | 0.99 | 5000 | 10 | 2 | 0 | -1,503 | -950 | 0/8 | -260 | -553 | 107 | 1820 | 80 |
| 11 | ltp_dev | 1 | 1.5 | 0.97 | 500 | 25 | 0 | 0 | -1,653 | -1,010 | 0/8 | -320 | -643 | 69 | 2128 | 80 |
| 12 | ltp_dev | 1 | 1.5 | 0.99 | 500 | 25 | 0 | 0 | -1,653 | -1,010 | 0/8 | -320 | -643 | 69 | 2128 | 80 |
| 13 | ltp_dev | 1 | 1.5 | 0.97 | 500 | 50 | 0 | 0 | -1,736 | -1,024 | 0/8 | -320 | -712 | 57 | 2234 | 80 |
| 14 | ltp_dev | 1 | 1.5 | 0.99 | 500 | 50 | 0 | 0 | -1,736 | -1,024 | 0/8 | -320 | -712 | 57 | 2234 | 80 |
| 15 | ltp_dev | 1 | 1.5 | 0.97 | 2000 | 10 | 2 | 0 | -1,629 | -1,053 | 0/8 | -400 | -576 | 107 | 1820 | 80 |
| 16 | ltp_dev | 1 | 1.5 | 0.99 | 2000 | 10 | 2 | 0 | -1,629 | -1,053 | 0/8 | -400 | -576 | 107 | 1820 | 80 |
| 17 | ltp_dev | 1 | 1.5 | 0.97 | 15000 | 10 | 0 | 0 | -1,880 | -1,070 | 0/8 | -320 | -810 | 97 | 1620 | 80 |
| 18 | ltp_dev | 1 | 1.5 | 0.99 | 15000 | 10 | 0 | 0 | -1,880 | -1,070 | 0/8 | -320 | -810 | 97 | 1620 | 80 |
| 19 | ltp_dev | 1 | 1.5 | 0.97 | 2000 | 25 | 0 | 0 | -1,763 | -1,090 | 0/8 | -400 | -673 | 71 | 2128 | 80 |
| 20 | ltp_dev | 1 | 1.5 | 0.97 | 5000 | 25 | 0 | 0 | -1,708 | -1,090 | 0/8 | -400 | -618 | 76 | 2128 | 80 |

## Confirm in the real simulator

1. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/sample_v2.csv --set '{"SIGNAL": "ltp_dev", "THRESHOLD": 1.5, "HOLD_US": 500000, "QTY": 10, "SLIP_TICKS": 0}' --no-ledger`
2. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/sample_v2.csv --set '{"SIGNAL": "ltp_dev", "THRESHOLD": 1.5, "HOLD_US": 500000, "QTY": 10, "SLIP_TICKS": 0}' --no-ledger`
3. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/sample_v2.csv --set '{"SIGNAL": "ltp_dev", "THRESHOLD": 1.5, "HOLD_US": 5000000, "QTY": 10, "SLIP_TICKS": 0}' --no-ledger`
4. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/sample_v2.csv --set '{"SIGNAL": "ltp_dev", "THRESHOLD": 1.5, "HOLD_US": 5000000, "QTY": 10, "SLIP_TICKS": 0}' --no-ledger`
5. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/sample_v2.csv --set '{"SIGNAL": "ltp_dev", "THRESHOLD": 1.5, "HOLD_US": 500000, "QTY": 10, "SLIP_TICKS": 2}' --no-ledger`
6. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/sample_v2.csv --set '{"SIGNAL": "ltp_dev", "THRESHOLD": 1.5, "HOLD_US": 500000, "QTY": 10, "SLIP_TICKS": 2}' --no-ledger`
7. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/sample_v2.csv --set '{"SIGNAL": "ltp_dev", "THRESHOLD": 1.5, "HOLD_US": 2000000, "QTY": 10, "SLIP_TICKS": 0}' --no-ledger`
8. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/sample_v2.csv --set '{"SIGNAL": "ltp_dev", "THRESHOLD": 1.5, "HOLD_US": 5000000, "QTY": 10, "SLIP_TICKS": 2}' --no-ledger`
9. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/sample_v2.csv --set '{"SIGNAL": "ltp_dev", "THRESHOLD": 1.5, "HOLD_US": 2000000, "QTY": 10, "SLIP_TICKS": 0}' --no-ledger`
10. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/sample_v2.csv --set '{"SIGNAL": "ltp_dev", "THRESHOLD": 1.5, "HOLD_US": 5000000, "QTY": 10, "SLIP_TICKS": 2}' --no-ledger`

## Best config per signal (by train pnl)

| signal | pnl | train | wins | hold | config |
|---|---:|---:|---:|---:|---|
| ltp_dev | -1,380 | -870 | 0 | -510 | `{"SIGNAL": "ltp_dev", "THRESHOLD": 1.5, "HOLD_US": 500000, "QTY": 10, "SLIP_TICKS": 0}` |
| ret_ms1000 | -3,020 | -2,110 | 0 | -910 | `{"SIGNAL": "ret_ms1000", "THRESHOLD": 1.5, "HOLD_US": 2000000, "QTY": 10, "SLIP_TICKS": 0}` |
| imb1 | -10,441 | -7,877 | 0 | -2,564 | `{"SIGNAL": "imb1", "THRESHOLD": 0.9736842105263158, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 0}` |
| cnt_imb1 | -12,773 | -9,637 | 0 | -3,136 | `{"SIGNAL": "cnt_imb1", "THRESHOLD": 0.95, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 0}` |
| ret_ms100 | -14,593 | -11,093 | 0 | -3,500 | `{"SIGNAL": "ret_ms100", "THRESHOLD": 0.5, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 0}` |
| flow_ms1000 | -15,860 | -12,550 | 0 | -3,310 | `{"SIGNAL": "flow_ms1000", "THRESHOLD": 1538.0, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 0}` |
| flow_ev100 | -20,880 | -16,930 | 0 | -3,950 | `{"SIGNAL": "flow_ev100", "THRESHOLD": 1044.0, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 0}` |
| imb5 | -25,916 | -20,577 | 0 | -5,339 | `{"SIGNAL": "imb5", "THRESHOLD": 0.4882331457113961, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 0}` |
| micro | -35,684 | -24,749 | 0 | -10,935 | `{"SIGNAL": "micro", "THRESHOLD": 0.45127610208816704, "HOLD_US": 15000000, "QTY": 25, "SLIP_TICKS": 0}` |
| flow_ms100 | -36,984 | -27,224 | 0 | -9,760 | `{"SIGNAL": "flow_ms100", "THRESHOLD": 640.0, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 0}` |
| flow_ev10 | -47,122 | -35,000 | 0 | -12,122 | `{"SIGNAL": "flow_ev10", "THRESHOLD": 0.0, "HOLD_US": 15000000, "QTY": 10, "SLIP_TICKS": 0}` |
| flow_ev1 | -50,057 | -37,608 | 0 | -12,449 | `{"SIGNAL": "flow_ev1", "THRESHOLD": 29.0, "HOLD_US": 15000000, "QTY": 50, "SLIP_TICKS": 0}` |
