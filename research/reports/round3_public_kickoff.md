# Kickoff round3_public

Generated 2026-11-08 01:07 by research.round. Data `data/round3_public.csv` (2,412,805 rows); earlier datasets: round1, round2, sample_v2.

## Insurance submission

**SUBMIT submission/best_strategy.py NOW as insurance: public pnl 272,063, no breach, check PASS, projected 202 s. Then make sure it is marked final.**

## Baselines (ledger tag `round3_public-baseline`)

| strategy | dataset | pnl | train_pnl | hold_pnl | chunk_wins | fills | breach_penalty | inv_q997 | runtime_s | error |
|---|---|---|---|---|---|---|---|---|---|---|
| best_strategy | round3_public | 272063.0 | 197503.27 | 74559.73 | 8 | 10897 | 0.0 | 100 | 76.2 |  |
| best_strategy | round1 | 443510.0 | 337011.67 | 106498.33 | 8 | 22740 | 0.0 | 100 | 109.2 |  |
| best_strategy | round2 | 349077.0 | 246729.95 | 102347.05 | 8 | 10371 | 0.0 | 100 | 59.7 |  |
| best_strategy | sample_v2 | 260409.0 | 197447.0 | 62962.0 | 8 | 4432 | 0.0 | 100 | 87.6 |  |
| mm | round3_public | -49389.0 | -37924.03 | -11464.97 | 1 | 23416 | 0.0 | 29 | 37.9 |  |
| mm | round1 | 27626.0 | 20632.99 | 6993.01 | 8 | 36733 | 0.0 | 18 | 60.8 |  |
| mm | round2 | -16797.0 | -13520.01 | -3276.99 | 3 | 22355 | 0.0 | 22 | 32.5 |  |
| mm | sample_v2 | 28025.0 | 20941.01 | 7083.99 | 8 | 11248 | 0.0 | 14 | 30.5 |  |
| trend_follower | round3_public | -195.0 | -298.0 | 103.0 | 2 | 635 | 0.0 | 1 | 66.3 |  |
| trend_follower | round1 | -410.0 | -410.0 | 0.0 | 0 | 786 | 0.0 | 1 | 87.5 |  |
| trend_follower | round2 | -391.0 | -180.0 | -211.0 | 4 | 1303 | 0.0 | 1 | 56.8 |  |
| trend_follower | sample_v2 | -531.0 | -531.0 | 0.0 | 0 | 618 | 0.0 | 1 | 56.3 |  |

## Reports

- Quick look (analyze_tape): `logs/round/round3_public/analyze_tape.log`
- Data card: `research/reports/round3_public.md`
- Signals 100 ms: `research/reports/round3_public_signals_h100ms.md`
- Signals 1 s: `research/reports/round3_public_signals_h1000ms.md`
- Taker scan: `research/reports/round3_public_taker_scan.md`
- Market-making edge: `research/reports/round3_public_mm_edge.md`

## Steps

| step | status | seconds |
|---|---|---|
| baseline_best_strategy | ok | 112 |
| baseline_mm | ok | 64 |
| baseline_trend_follower | ok | 92 |
| check_champion | ok | 22 |
| analyze_tape | ok | 77 |
| features | ok | 44 |
| profile | ok | 28 |
| signals_h100ms | ok | 133 |
| signals_h1000ms | ok | 65 |
| taker_scan | ok | 32 |
| mm_edge | ok | 4 |

## Next

1. Read the data card and the quick look: momentum or reversion, spread regime, which signals are stable.
2. Start research wave 1 (Workflow tool, `scriptPath: research/workflows/research_wave.js`), args:
   `{"data": "data/round3_public.csv", "prior": ["data/round1.csv", "data/round2.csv", "data/sample_v2.csv"], "champion": "submission/best_strategy.py", "wave": 1, "workers": 2}`
3. Confirm the top taker-scan configs in the real simulator (commands in the taker-scan report).
