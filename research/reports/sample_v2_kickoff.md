# Kickoff sample_v2

Generated 2026-11-07 12:06 by research.round. Data `data/sample_v2.csv` (2,000,000 rows); earlier datasets: sample.

## Insurance submission

**DO NOT SUBMIT the champion yet: public pnl -2,440,965 <= 0. Positive baseline instead: mm.**

## Baselines (ledger tag `sample_v2-baseline`)

| strategy | dataset | pnl | train_pnl | hold_pnl | chunk_wins | fills | breach_penalty | inv_q997 | runtime_s | error |
|---|---|---|---|---|---|---|---|---|---|---|
| best_strategy | sample_v2 | -2440965.0 | -1842701.55 | -598263.45 | 0 | 47530 | 0.0 | 80 | 30.7 |  |
| best_strategy | sample | 51225.0 | 38343.0 | 12882.0 | 8 | 913 | 0.0 | 45 | 0.6 |  |
| mm | sample_v2 | 28025.0 | 20941.01 | 7083.99 | 8 | 11248 | 0.0 | 14 | 28.9 |  |
| mm | sample | -9056.0 | -7687.0 | -1369.0 | 0 | 424 | 0.0 | 14 | 1.3 |  |
| trend_follower | sample_v2 | -531.0 | -531.0 | 0.0 | 0 | 618 | 0.0 | 1 | 21.3 |  |
| trend_follower | sample | 1878.0 | 1463.0 | 415.0 | 8 | 305 | 0.0 | 1 | 0.5 |  |

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
| publish | ok | 369 |
| baseline_best_strategy | ok | 35 |
| baseline_mm | ok | 34 |
| baseline_trend_follower | ok | 26 |
| check_champion | ok | 16 |
| analyze_tape | ok | 77 |
| features | ok | 46 |
| profile | ok | 12 |
| signals_h100ms | ok | 58 |
| signals_h1000ms | ok | 60 |
| taker_scan | ok | 46 |
| mm_edge | ok | 5 |

## Next

1. Read the data card and the quick look: momentum or reversion, spread regime, which signals are stable.
2. Start research wave 1 (Workflow tool, `scriptPath: research/workflows/research_wave.js`), args:
   `{"data": "data/sample_v2.csv", "prior": ["data/sample.csv"], "champion": "submission/best_strategy.py", "wave": 1, "workers": 2}`
3. Confirm the top taker-scan configs in the real simulator (commands in the taker-scan report).
