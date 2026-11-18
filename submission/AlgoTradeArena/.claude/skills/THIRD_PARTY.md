# Skills: what is ours and what is borrowed

Ours (written for this repo): `arena-market-mechanics`, `round-kickoff`, `run-experiment`, `pre-submit`, `signal-research`.

Third-party, vetted 2026-11-07 (every SKILL.md read in full, scripts and references scanned for
network, process, env and credential access, hidden text). Each folder keeps the upstream
licence as `LICENSE.upstream.md`.

| Skill | Source (commit) | Licence | Local changes |
|---|---|---|---|
| walk-forward-validation | agiprolabs/claude-trading-skills (981e1d7) | MIT | Note on the buggy inline DSR snippet; use `scripts/overfit_detector.py` |
| market-microstructure-traditional | agiprolabs/claude-trading-skills (981e1d7) | MIT | Description rewritten for limit order books instead of crypto |
| backtest-review | shakeebshaan/claude-code-quant-skills (6b39f8f) | MIT | none |
| scikit-learn | K-Dense-AI/scientific-agent-skills (a25cf9e) | MIT | Removed `allowed-tools` (pre-approved Bash), the self-citation section that told agents to fetch arxiv and add references to our code, the install commands, and `scripts/` |
| statsmodels | K-Dense-AI/scientific-agent-skills (a25cf9e) | MIT | Same removals; install block replaced by "offline only via `uv run --with statsmodels`, never in a strategy" |

Considered and skipped (crypto/DeFi/equities focus, live vendor API calls, or descriptions broad
enough to hijack unrelated tasks): the other 23 candidates from the same three repos.
