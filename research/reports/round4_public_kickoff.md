# Kickoff round4_public

Generated 2026-11-08 07:19 by research.round. Data `../AlgoTradeArena/data/round4_public.csv` (2,398,520 rows); earlier datasets: none.

## Insurance submission

**DO NOT SUBMIT the champion yet: breach penalty 268; research.check did not PASS; projected runtime 2832.0 s (limit 420 s with margin). No baseline is positive: wait for wave 1.**

## Baselines (ledger tag `round4_public-baseline`)

| strategy | dataset | pnl | train_pnl | hold_pnl | chunk_wins | fills | breach_penalty | inv_q997 | runtime_s | error |
|---|---|---|---|---|---|---|---|---|---|---|
| best_strategy | round4_public | 1027483.35 | 837946.86 | 189536.49 | 8 | 6970 | 267.65 | 100 | 576.6 |  |
| mm | round4_public | -154906.0 | -120440.03 | -34465.97 | 0 | 23510 | 0.0 | 44 | 125.8 |  |
| trend_follower | round4_public | -12.0 | -12.0 | 0.0 | 0 | 24 | 0.0 | 0 | 79.3 |  |

## Reports

- Quick look (analyze_tape): `logs/round/round4_public/analyze_tape.log`
- Data card: `research/reports/round4_public.md`
- Signals 100 ms: `research/reports/round4_public_signals_h100ms.md`
- Signals 1 s: `research/reports/round4_public_signals_h1000ms.md`
- Taker scan: `research/reports/round4_public_taker_scan.md`
- Market-making edge: `research/reports/round4_public_mm_edge.md`

## Steps

| step | status | seconds |
|---|---|---|
| baseline_best_strategy | ok | 584 |
| baseline_mm | ok | 135 |
| baseline_trend_follower | ok | 88 |
| check_champion | FAILED | 154 |
| analyze_tape | ok | 114 |
| features | ok | 107 |
| profile | ok | 67 |
| signals_h100ms | ok | 435 |
| signals_h1000ms | ok | 246 |
| taker_scan | ok | 66 |
| mm_edge | ok | 10 |

## Next

1. Read the data card and the quick look: momentum or reversion, spread regime, which signals are stable.
2. Start research wave 1 (Workflow tool, `scriptPath: research/workflows/research_wave.js`), args:
   `{"data": "../AlgoTradeArena/data/round4_public.csv", "prior": [], "champion": "submission/best_strategy.py", "wave": 1, "workers": 2}`
3. Confirm the top taker-scan configs in the real simulator (commands in the taker-scan report).
