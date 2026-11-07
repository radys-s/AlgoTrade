# Signal-taker proxy scan - `data/sample.csv`

1152 configs on 53,758 rows in 5.7 s (4 workers). Proxy model: decision at row i, IOC applied against the book after the last row before ts_i + 5 ms, 5 levels assumed at consecutive ticks, no self-impact. Ranked by train pnl (first 75% of time, 8 chunks); holdout = last 25%.

| # | signal | sign | thr | q | hold_ms | qty | slip | rev | pnl | train | wins | chunk_min | hold | orders | vol | max|I| |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | flow_ev10 | 1 | 0 | 0.8 | 5000 | 50 | 2 | 0 | 154,741 | 115,500 | 8/8 | 9,471 | 39,241 | 2069 | 41356 | 80 |
| 2 | flow_ev10 | 1 | 0 | 0.8 | 5000 | 50 | 0 | 0 | 154,079 | 115,016 | 8/8 | 9,371 | 39,063 | 2071 | 41052 | 80 |
| 3 | flow_ev10 | 1 | 0 | 0.8 | 2000 | 50 | 2 | 0 | 142,144 | 109,297 | 8/8 | 10,655 | 32,847 | 2758 | 55460 | 80 |
| 4 | flow_ev10 | 1 | 0 | 0.8 | 5000 | 25 | 2 | 0 | 145,830 | 108,547 | 8/8 | 8,846 | 37,283 | 2070 | 38508 | 80 |
| 5 | flow_ev10 | 1 | 0 | 0.8 | 2000 | 50 | 0 | 0 | 141,341 | 108,500 | 8/8 | 10,488 | 32,841 | 2764 | 55072 | 80 |
| 6 | flow_ev10 | 1 | 0 | 0.8 | 5000 | 25 | 0 | 0 | 145,307 | 108,188 | 8/8 | 8,775 | 37,119 | 2072 | 38222 | 80 |
| 7 | flow_ms1000 | 1 | 10 | 0.8 | 5000 | 50 | 2 | 0 | 133,003 | 104,386 | 8/8 | 8,991 | 28,617 | 1880 | 38546 | 80 |
| 8 | flow_ms1000 | 1 | 10 | 0.8 | 5000 | 50 | 0 | 0 | 132,755 | 104,169 | 8/8 | 8,798 | 28,586 | 1894 | 38406 | 80 |
| 9 | flow_ms1000 | 1 | 10 | 0.8 | 5000 | 25 | 2 | 0 | 131,265 | 103,213 | 8/8 | 8,624 | 28,052 | 1969 | 38132 | 80 |
| 10 | flow_ms1000 | 1 | 10 | 0.8 | 5000 | 25 | 0 | 0 | 130,974 | 102,976 | 8/8 | 8,467 | 27,998 | 1984 | 38002 | 80 |
| 11 | flow_ev10 | 1 | 0 | 0.8 | 2000 | 25 | 2 | 0 | 127,937 | 98,133 | 8/8 | 9,517 | 29,804 | 2680 | 50496 | 80 |
| 12 | flow_ms1000 | 1 | 10 | 0.8 | 2000 | 50 | 2 | 0 | 129,172 | 98,094 | 8/8 | 9,366 | 31,078 | 2449 | 50350 | 80 |
| 13 | flow_ev10 | 1 | 8 | 0.9 | 5000 | 50 | 2 | 0 | 126,539 | 97,584 | 8/8 | 8,020 | 28,955 | 1691 | 34728 | 80 |
| 14 | flow_ev10 | 1 | 0 | 0.8 | 2000 | 25 | 0 | 0 | 127,135 | 97,322 | 8/8 | 9,358 | 29,813 | 2686 | 50146 | 80 |
| 15 | flow_ms1000 | 1 | 10 | 0.8 | 2000 | 50 | 0 | 0 | 128,108 | 97,136 | 8/8 | 9,053 | 30,972 | 2454 | 50022 | 80 |

## Confirm in the real simulator

1. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/sample.csv --set '{"SIGNAL": "flow_ev10", "THRESHOLD": 0.0, "HOLD_US": 5000000, "QTY": 50, "SLIP_TICKS": 2}' --no-ledger`
2. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/sample.csv --set '{"SIGNAL": "flow_ev10", "THRESHOLD": 0.0, "HOLD_US": 5000000, "QTY": 50, "SLIP_TICKS": 0}' --no-ledger`
3. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/sample.csv --set '{"SIGNAL": "flow_ev10", "THRESHOLD": 0.0, "HOLD_US": 2000000, "QTY": 50, "SLIP_TICKS": 2}' --no-ledger`
4. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/sample.csv --set '{"SIGNAL": "flow_ev10", "THRESHOLD": 0.0, "HOLD_US": 5000000, "QTY": 25, "SLIP_TICKS": 2}' --no-ledger`
5. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/sample.csv --set '{"SIGNAL": "flow_ev10", "THRESHOLD": 0.0, "HOLD_US": 2000000, "QTY": 50, "SLIP_TICKS": 0}' --no-ledger`
6. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/sample.csv --set '{"SIGNAL": "flow_ev10", "THRESHOLD": 0.0, "HOLD_US": 5000000, "QTY": 25, "SLIP_TICKS": 0}' --no-ledger`
7. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/sample.csv --set '{"SIGNAL": "flow_ms1000", "THRESHOLD": 10.0, "HOLD_US": 5000000, "QTY": 50, "SLIP_TICKS": 2}' --no-ledger`
8. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/sample.csv --set '{"SIGNAL": "flow_ms1000", "THRESHOLD": 10.0, "HOLD_US": 5000000, "QTY": 50, "SLIP_TICKS": 0}' --no-ledger`
9. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/sample.csv --set '{"SIGNAL": "flow_ms1000", "THRESHOLD": 10.0, "HOLD_US": 5000000, "QTY": 25, "SLIP_TICKS": 2}' --no-ledger`
10. `uv run python -m research.run --strategy strategies/signal_taker.py --data data/sample.csv --set '{"SIGNAL": "flow_ms1000", "THRESHOLD": 10.0, "HOLD_US": 5000000, "QTY": 25, "SLIP_TICKS": 0}' --no-ledger`

## Best config per signal (by train pnl)

| signal | pnl | train | wins | hold | config |
|---|---:|---:|---:|---:|---|
| flow_ev10 | 154,741 | 115,500 | 8 | 39,241 | `{"SIGNAL": "flow_ev10", "THRESHOLD": 0.0, "HOLD_US": 5000000, "QTY": 50, "SLIP_TICKS": 2}` |
| flow_ms1000 | 133,003 | 104,386 | 8 | 28,617 | `{"SIGNAL": "flow_ms1000", "THRESHOLD": 10.0, "HOLD_US": 5000000, "QTY": 50, "SLIP_TICKS": 2}` |
| flow_ev100 | 92,326 | 71,737 | 8 | 20,589 | `{"SIGNAL": "flow_ev100", "THRESHOLD": 19.0, "HOLD_US": 5000000, "QTY": 50, "SLIP_TICKS": 2}` |
| flow_ms100 | 74,722 | 54,557 | 8 | 20,165 | `{"SIGNAL": "flow_ms100", "THRESHOLD": 0.0, "HOLD_US": 5000000, "QTY": 50, "SLIP_TICKS": 2}` |
| ret_ms1000 | 66,977 | 48,753 | 8 | 18,224 | `{"SIGNAL": "ret_ms1000", "THRESHOLD": 5.0, "HOLD_US": 2000000, "QTY": 50, "SLIP_TICKS": 2}` |
| ret_ms100 | 51,441 | 43,593 | 8 | 7,848 | `{"SIGNAL": "ret_ms100", "THRESHOLD": 1.0, "HOLD_US": 5000000, "QTY": 50, "SLIP_TICKS": 2}` |
| flow_ev1 | 45,203 | 34,165 | 8 | 11,038 | `{"SIGNAL": "flow_ev1", "THRESHOLD": 0.0, "HOLD_US": 5000000, "QTY": 50, "SLIP_TICKS": 2}` |
| ltp_dev | 8,436 | 4,457 | 5 | 3,979 | `{"SIGNAL": "ltp_dev", "THRESHOLD": 12.0, "HOLD_US": 15000000, "QTY": 50, "SLIP_TICKS": 0}` |
| imb1 | 2,761 | 2,201 | 8 | 560 | `{"SIGNAL": "imb1", "THRESHOLD": 0.5918367346938775, "HOLD_US": 5000000, "QTY": 10, "SLIP_TICKS": 0}` |
| imb5 | 2,761 | 2,201 | 8 | 560 | `{"SIGNAL": "imb5", "THRESHOLD": 0.5918367346938775, "HOLD_US": 5000000, "QTY": 10, "SLIP_TICKS": 0}` |
| micro | 2,761 | 2,201 | 8 | 560 | `{"SIGNAL": "micro", "THRESHOLD": 0.5918367346938775, "HOLD_US": 5000000, "QTY": 10, "SLIP_TICKS": 0}` |
| cnt_imb1 | 0 | 0 | 0 | 0 | `{"SIGNAL": "cnt_imb1", "THRESHOLD": 0.0, "HOLD_US": 500000, "QTY": 10, "SLIP_TICKS": 0}` |
