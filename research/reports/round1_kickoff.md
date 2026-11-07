# Kickoff round1

Generated 2026-11-07 13:06 by research.round. Data `data/round1.csv` (4,411,336 rows); earlier datasets: sample_v2.

## Insurance submission

**SUBMIT submission/best_strategy.py NOW as insurance: public pnl 321,024, no breach, check PASS, projected 45 s. Then make sure it is marked final.**

## Baselines (ledger tag `round1-baseline`)

| strategy | dataset | pnl | train_pnl | hold_pnl | chunk_wins | fills | breach_penalty | inv_q997 | runtime_s | error |
|---|---|---|---|---|---|---|---|---|---|---|
| best_strategy | round1 | 321024.0 | 238752.77 | 82271.23 | 8 | 42193 | 0.0 | 100 | 23.0 |  |
| best_strategy | sample_v2 | 237306.0 | 182377.0 | 54929.0 | 8 | 10333 | 0.0 | 100 | 14.2 |  |
| mm | round1 | 27626.0 | 20632.99 | 6993.01 | 8 | 36733 | 0.0 | 18 | 21.8 |  |
| mm | sample_v2 | 28025.0 | 20941.01 | 7083.99 | 8 | 11248 | 0.0 | 14 | 10.6 |  |
| trend_follower | round1 | -410.0 | -410.0 | 0.0 | 0 | 786 | 0.0 | 1 | 17.5 |  |
| trend_follower | sample_v2 | -531.0 | -531.0 | 0.0 | 0 | 618 | 0.0 | 1 | 8.6 |  |

## Reports

- Quick look (analyze_tape): `logs/round/round1/analyze_tape.log`
- Data card: `research/reports/round1.md`
- Signals 100 ms: `research/reports/round1_signals_h100ms.md`
- Signals 1 s: `research/reports/round1_signals_h1000ms.md`
- Taker scan: `research/reports/round1_taker_scan.md`
- Market-making edge: `research/reports/round1_mm_edge.md`

## Steps

| step | status | seconds |
|---|---|---|
| baseline_best_strategy | ok | 25 |
| baseline_mm | ok | 24 |
| baseline_trend_follower | ok | 20 |
| check_champion | ok | 5 |
| analyze_tape | ok | 53 |
| features | ok | 37 |
| profile | ok | 14 |
| signals_h100ms | ok | 36 |
| signals_h1000ms | ok | 38 |
| taker_scan | ok | 33 |
| mm_edge | ok | 4 |

## Next

1. Read the data card and the quick look: momentum or reversion, spread regime, which signals are stable.
2. Start research wave 1 (Workflow tool, `scriptPath: research/workflows/research_wave.js`), args:
   `{"data": "data/round1.csv", "prior": ["data/sample_v2.csv"], "champion": "submission/best_strategy.py", "wave": 1, "workers": 2}`
3. Confirm the top taker-scan configs in the real simulator (commands in the taker-scan report).
