# Kickoff round2

Generated 2026-11-07 19:20 by research.round. Data `data/round2.csv` (2,193,064 rows); earlier datasets: round1, round1_drift120, round1_drift40, round1_mirror, sample_v2.

## Insurance submission

**DO NOT SUBMIT the champion yet: public pnl -165,957 <= 0. No baseline is positive: wait for wave 1.**

## Baselines (ledger tag `round2-baseline`)

| strategy | dataset | pnl | train_pnl | hold_pnl | chunk_wins | fills | breach_penalty | inv_q997 | runtime_s | error |
|---|---|---|---|---|---|---|---|---|---|---|
| best_strategy | round2 | -165957.0 | -143909.06 | -22047.94 | 3 | 24257 | 0.0 | 100 | 60.9 |  |
| best_strategy | round1 | 515687.0 | 384137.39 | 131549.61 | 8 | 43055 | 0.0 | 100 | 130.0 |  |
| best_strategy | round1_drift120 | 511241.0 | 379659.39 | 131581.61 | 8 | 43057 | 0.0 | 100 | 124.9 |  |
| best_strategy | round1_drift40 | 513298.0 | 381966.39 | 131331.61 | 8 | 43036 | 0.0 | 100 | 125.6 |  |
| best_strategy | round1_mirror | 517336.0 | 385196.49 | 132139.51 | 8 | 43031 | 0.0 | 100 | 125.7 |  |
| best_strategy | sample_v2 | 260598.0 | 197132.0 | 63466.0 | 8 | 9280 | 0.0 | 100 | 76.2 |  |
| mm | round2 | -16797.0 | -13520.01 | -3276.99 | 3 | 22355 | 0.0 | 22 | 13.3 |  |
| mm | round1 | 27626.0 | 20632.99 | 6993.01 | 8 | 36733 | 0.0 | 18 | 24.6 |  |
| mm | round1_drift120 | 27724.0 | 20613.99 | 7110.01 | 8 | 36823 | 0.0 | 18 | 24.2 |  |
| mm | round1_drift40 | 27523.0 | 20450.99 | 7072.01 | 8 | 36746 | 0.0 | 18 | 25.4 |  |
| mm | round1_mirror | 27626.0 | 20632.99 | 6993.01 | 8 | 36733 | 0.0 | 18 | 25.9 |  |
| mm | sample_v2 | 28025.0 | 20941.01 | 7083.99 | 8 | 11248 | 0.0 | 14 | 12.0 |  |
| trend_follower | round2 | -391.0 | -180.0 | -211.0 | 4 | 1303 | 0.0 | 1 | 9.2 |  |
| trend_follower | round1 | -410.0 | -410.0 | 0.0 | 0 | 786 | 0.0 | 1 | 20.6 |  |
| trend_follower | round1_drift120 | -409.0 | -409.0 | 0.0 | 0 | 784 | 0.0 | 1 | 19.1 |  |
| trend_follower | round1_drift40 | -410.0 | -410.0 | 0.0 | 0 | 786 | 0.0 | 1 | 20.1 |  |
| trend_follower | round1_mirror | -410.0 | -410.0 | 0.0 | 0 | 786 | 0.0 | 1 | 18.6 |  |
| trend_follower | sample_v2 | -531.0 | -531.0 | 0.0 | 0 | 618 | 0.0 | 1 | 8.9 |  |

## Reports

- Quick look (analyze_tape): `logs/round/round2/analyze_tape.log`
- Data card: `research/reports/round2.md`
- Signals 100 ms: `research/reports/round2_signals_h100ms.md`
- Signals 1 s: `research/reports/round2_signals_h1000ms.md`
- Taker scan: `research/reports/round2_taker_scan.md`
- Market-making edge: `research/reports/round2_mm_edge.md`

## Steps

| step | status | seconds |
|---|---|---|
| baseline_best_strategy | ok | 133 |
| baseline_mm | ok | 29 |
| baseline_trend_follower | ok | 23 |
| check_champion | ok | 16 |
| analyze_tape | ok | 26 |
| features | ok | 22 |
| profile | ok | 9 |
| signals_h100ms | ok | 72 |
| signals_h1000ms | ok | 57 |
| taker_scan | ok | 19 |
| mm_edge | ok | 3 |

## Next

1. Read the data card and the quick look: momentum or reversion, spread regime, which signals are stable.
2. Start research wave 1 (Workflow tool, `scriptPath: research/workflows/research_wave.js`), args:
   `{"data": "data/round2.csv", "prior": ["data/round1.csv", "data/round1_drift120.csv", "data/round1_drift40.csv", "data/round1_mirror.csv", "data/sample_v2.csv"], "champion": "submission/best_strategy.py", "wave": 1, "workers": 2}`
3. Confirm the top taker-scan configs in the real simulator (commands in the taker-scan report).
