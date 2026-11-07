# Kickoff sample_v2

Generated 2026-11-07 12:22 by research.round. Data `data/sample_v2.csv` (2,000,000 rows); earlier datasets: none.

## Insurance submission

**SUBMIT submission/best_strategy.py NOW as insurance: public pnl 237,306, no breach, check PASS, projected 57 s. Then make sure it is marked final.**

## Baselines (ledger tag `sample_v2-baseline`)

| strategy | dataset | pnl | train_pnl | hold_pnl | chunk_wins | fills | breach_penalty | inv_q997 | runtime_s | error |
|---|---|---|---|---|---|---|---|---|---|---|
| best_strategy | sample_v2 | 237306.0 | 182377.0 | 54929.0 | 8 | 10333 | 0.0 | 100 | 14.2 |  |
| mm | sample_v2 | 28025.0 | 20941.01 | 7083.99 | 8 | 11248 | 0.0 | 14 | 10.7 |  |
| trend_follower | sample_v2 | -531.0 | -531.0 | 0.0 | 0 | 618 | 0.0 | 1 | 8.6 |  |

## Reports

- Quick look (analyze_tape): `logs/round/sample_v2/analyze_tape.log`
- Data card: `research/reports/sample_v2.md`
- Signals 100 ms: `research/reports/sample_v2_signals_h100ms.md`
- Signals 1 s: `research/reports/sample_v2_signals_h1000ms.md`
- Taker scan: `research/reports/sample_v2_taker_scan.md`
- Market-making edge: `research/reports/sample_v2_mm_edge.md`

## Steps

| step | status | seconds |
|---|---|---|
| baseline_best_strategy | ok | 16 |
| baseline_mm | ok | 13 |
| baseline_trend_follower | ok | 11 |
| check_champion | ok | 5 |
| analyze_tape | ok | 31 |
| features | ok | 19 |
| profile | ok | 7 |
| signals_h100ms | ok | 26 |
| signals_h1000ms | ok | 25 |
| taker_scan | ok | 28 |
| mm_edge | ok | 3 |

## Next

1. Read the data card and the quick look: momentum or reversion, spread regime, which signals are stable.
2. Start research wave 1 (Workflow tool, `scriptPath: research/workflows/research_wave.js`), args:
   `{"data": "data/sample_v2.csv", "prior": [], "champion": "submission/best_strategy.py", "wave": 1, "workers": 2}`
3. Confirm the top taker-scan configs in the real simulator (commands in the taker-scan report).
