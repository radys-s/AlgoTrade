export const meta = {
  name: 'research-wave',
  description: 'One research wave for the current round: propose experiments, implement and sweep them in parallel, try to refute the winners, build a champion candidate',
  whenToUse: 'During an Algo Trade Arena round, once the round CSV is unzipped in data/',
  phases: [
    { title: 'Recon', detail: 'features, data card and baselines on the new data (args.recon only)' },
    { title: 'Hypotheses', detail: 'three lenses propose experiments, a selector picks the slate' },
    { title: 'Experiments', detail: 'each experiment in its own strategy file, swept with research.run' },
    { title: 'Skeptic', detail: 'independent agent tries to refute each promising result' },
    { title: 'Integrate', detail: 'accepted changes merged into a candidate, wave report and progress log' },
  ],
}

// args: { data: 'data/round1.csv', prior: ['data/playground.csv'], champion: 'submission/best_strategy.py',
//         wave: 1, slots: 6, workers: 2, recon: false, focus: 'optional steer from the team',
//         models: { propose, experiments, skeptic, integrate } }  // e.g. {experiments: 'sonnet'}; omitted = session model
const A = args || {}
const MODELS = A.models || {}
const withModel = (phaseKey, opts) => (MODELS[phaseKey] ? { ...opts, model: MODELS[phaseKey] } : opts)
const DATA = A.data
if (!DATA) throw new Error('args.data (the current round CSV) is required')
const PRIOR = A.prior || []
const CHAMPION = A.champion || 'submission/best_strategy.py'
const WAVE = A.wave || 1
const SLOTS = A.slots || 6
const WORKERS = A.workers || 2
const STEM = DATA.split('/').pop().replace(/\.csv$/, '')
const ALL_DATA = [DATA, ...PRIOR].join(' ')
const FOCUS = A.focus ? `\nSteer from the team for this wave: ${A.focus}\n` : ''

const CTX = `
You are part of an automated quant research team in the Qminers Algo Trade Arena 2026 hackathon.
The repo root is your current working directory (Windows; the Bash tool is Git Bash). uv must be on
PATH (per-user install in ~/.local/bin); run Python as uv run python ... from the repo root.
Rules and mechanics: docs/kontext.md, docs/workflow.md, .claude/skills/arena-market-mechanics/SKILL.md.
Current round data: ${DATA}. Earlier datasets for robustness: ${PRIOR.length ? PRIOR.join(', ') : 'none yet'}.
Champion (what is submitted now): ${CHAMPION}. Start from research/reports/${STEM}_kickoff.md, the data card
research/reports/${STEM}.md, the taker scan research/reports/${STEM}_taker_scan.md and the market-making
edge research/reports/${STEM}_mm_edge.md (whichever exist).
Tools (all write to the shared ledger in research/results/, never delete ledger files):
- uv run python -m research.taker_scan data/features/${STEM}.npz --grid '{..}'   (fast approximate scan of
  signal-taker configs; confirm winners with research.run on strategies/signal_taker.py)
- uv run python -m research.mm_edge ${DATA}   (when passive quotes at the touch make money)
- uv run python -m research.run --strategy F --data ${ALL_DATA} --grid '{"p": [..]}' --workers ${WORKERS} --tag T
- uv run python -m research.ledger --data ${STEM} --top 20      (what has been tried already, by anyone)
- uv run python -m research.compare --a ${CHAMPION} --b F [--b-set '{..}'] --data ${ALL_DATA} --workers ${WORKERS}
- uv run python -m research.check F --data ${DATA}               (sandbox rules + runtime projection)
- uv run python -m research.signals data/features/${STEM}.npz --horizon-ms H   (signal / ML research)
Hard rules:
- Strategy files: one BaseStrategy subclass; tunables either in a class dict PARAMS read via self.PARAMS[...]
  or as UPPER_CASE class constants (both work with research.run --grid); new mechanics go behind a
  parameter whose default reproduces the old behaviour.
- Sandbox: only stdlib + numpy/pandas/scipy/sklearn + simulator.models + simulator.strategy.BaseStrategy;
  no os/sys/io/socket/subprocess/threading/importlib/inspect/gc/pathlib, no open/eval/exec/getattr/type,
  no dunder attributes except __init__, no str.format. Keep per-event work tiny (up to ~9M events, 15 min, 1 CPU).
- Position invariant: |inventory| + all our live and in-flight same-side quantity <= max_position <= 100, always.
- Use at most ${WORKERS} worker processes. Only write files you are told to own. Never edit simulator/, explorer/,
  ${CHAMPION}, STATUS.md or README.md. Never commit or push. Never try to escape or probe the sandbox.
- Judge results by paired chunks, holdout and earlier datasets, not by the total on one dataset.
`

const EXPERIMENT = {
  type: 'object',
  properties: {
    slug: { type: 'string', description: 'short snake_case id' },
    title: { type: 'string' },
    hypothesis: { type: 'string', description: 'what market behaviour we exploit and why it should hold on private data' },
    change: { type: 'string', description: 'concrete code change to the champion, behind which PARAMS flag' },
    grid: { type: 'string', description: 'JSON grid for research.run, small (<= 9 combos)' },
    evidence: { type: 'string', description: 'data card / ledger facts supporting it' },
    expected_value: { type: 'number', description: '0-10, gain x probability' },
  },
  required: ['slug', 'title', 'hypothesis', 'change', 'grid', 'expected_value'],
}
const PROPOSALS = { type: 'object', properties: { experiments: { type: 'array', items: EXPERIMENT } }, required: ['experiments'] }
const RESULT = {
  type: 'object',
  properties: {
    slug: { type: 'string' },
    file: { type: 'string' },
    flag_off_matches_champion: { type: 'boolean' },
    best_overrides: { type: 'string', description: 'JSON of the best robust overrides' },
    pnl: { type: 'number' }, hold_pnl: { type: 'number' }, chunk_wins: { type: 'number' },
    champion_pnl: { type: 'number' }, champion_hold_pnl: { type: 'number' },
    promising: { type: 'boolean' },
    summary: { type: 'string', description: 'what was tried, numbers, why it did or did not work' },
  },
  required: ['slug', 'file', 'promising', 'summary'],
}
const VERDICT = {
  type: 'object',
  properties: {
    slug: { type: 'string' },
    verdict: { type: 'string', enum: ['ACCEPT', 'REJECT'] },
    overrides: { type: 'string' },
    compare_summary: { type: 'string' },
    reasons: { type: 'string' },
  },
  required: ['slug', 'verdict', 'reasons'],
}
const INTEGRATION = {
  type: 'object',
  properties: {
    candidate_file: { type: 'string' },
    verdict: { type: 'string', enum: ['ACCEPT', 'REJECT', 'NONE'] },
    pnl_gain_primary: { type: 'number' },
    submit_recommendation: { type: 'string' },
    report_file: { type: 'string' },
    summary: { type: 'string' },
  },
  required: ['verdict', 'summary', 'submit_recommendation'],
}

if (A.recon) {
  phase('Recon')
  await parallel([
    () => agent(`${CTX}
Task: prepare the new round data. If data/features/${STEM}.npz is missing run research.features on ${DATA};
then run research.profile on ${DATA} (writes research/reports/${STEM}.md). Read the report and add at its end a
section "## Opportunities" with 5-8 concrete, testable trading ideas this data suggests (cite the numbers).
You own research/reports/${STEM}.md. Return a 10-line summary of the market and the ideas.`, withModel('experiments', { label: 'recon:data-card', phase: 'Recon' })),
    () => agent(`${CTX}
Task: baselines. Run research.run (no grid) for ${CHAMPION}, strategies/mm.py and strategies/trend_follower.py
on ${ALL_DATA} with --tag w${WAVE}-baseline (skip files that do not exist). Return the table of pnl, hold_pnl,
chunk_wins, breach, fills per strategy and dataset, and 3 lines on what it means.`, withModel('experiments', { label: 'recon:baselines', phase: 'Recon' })),
  ])
}

phase('Hypotheses')
const LENSES = [
  ['microstructure', 'market making mechanics: quote placement, size, skew, queue position, requote rules, adverse selection, replay mechanics (the tape never reacts to us)'],
  ['signals', 'predictive signals and ML: book imbalance, microprice, order-flow imbalance, trade flow, walk-forward models from research.signals; how to use a forecast (skew quotes, pull a side, take with IOC)'],
  ['risk', 'risk and regimes: inventory control, unwinding, volatility/spread/activity regimes, robustness across datasets, end-of-data inventory, runtime'],
]
const proposals = await parallel(LENSES.map(([lens, scope]) => () => agent(`${CTX}${FOCUS}
You are the ${lens} researcher. Scope: ${scope}.
Read the data card research/reports/${STEM}.md, the ledger (research.ledger --data ${STEM} and without filter),
docs/progress.md, earlier wave reports in research/reports/wave_*.md, and the champion code.
Propose up to 4 experiments for wave ${WAVE} that are NOT already tried (check the ledger and progress log).
Each must be a small change to the champion behind a PARAMS flag, testable within ~30 minutes, with a grid of
<= 9 combos. Prefer ideas with a mechanism that should survive on unseen data over curve fitting.`,
  withModel('propose', { label: `propose:${lens}`, phase: 'Hypotheses', schema: PROPOSALS }))))
const pool = proposals.filter(Boolean).flatMap(p => p.experiments)
log(`${pool.length} proposals from ${proposals.filter(Boolean).length} lenses`)
if (!pool.length) return { error: 'no proposals' }

const slate = await agent(`${CTX}
You are the research lead. Pick the ${SLOTS} experiments with the highest expected value from these proposals,
merging duplicates and keeping the slate diverse (not all from one lens). Return them unchanged in shape
(you may tighten the grid). Proposals:
${JSON.stringify(pool, null, 1)}`, withModel('propose', { label: 'select-slate', phase: 'Hypotheses', schema: PROPOSALS }))
const selected = (slate && slate.experiments.length ? slate.experiments : pool).slice(0, SLOTS)
log(`slate: ${selected.map(e => e.slug).join(', ')}`)

const outcomes = await pipeline(
  selected,
  exp => agent(`${CTX}
Experiment "${exp.title}" (wave ${WAVE}).
Hypothesis: ${exp.hypothesis}
Change: ${exp.change}
Grid: ${exp.grid}
Steps:
1. Copy ${CHAMPION} to strategies/exp_w${WAVE}_${exp.slug}.py (you own only this file) and implement the change
   behind PARAMS flag(s). With the flag off it must reproduce the champion exactly: verify by running both
   with research.run on ${DATA} (same pnl).
2. research.check the file (static part must pass).
3. Sweep with research.run on ${ALL_DATA}, --tag w${WAVE}-${exp.slug}, --workers ${WORKERS}. If the first grid
   shows a trend toward an edge, extend it once in that direction.
4. Pick the most robust overrides (plateau, chunk wins, holdout, earlier datasets) — not the single max.
Set promising=true only if it beats the champion on the primary dataset in train AND holdout without hurting
earlier datasets. Report honestly, including failure.`, withModel('experiments', { label: `exp:${exp.slug}`, phase: 'Experiments', schema: RESULT })),
  (res, exp) => {
    if (!res) return null
    if (!res.promising) return { slug: exp.slug, verdict: 'REJECT', reasons: `not promising: ${res.summary}`, result: res }
    return agent(`${CTX}
You are an adversarial reviewer. Default to REJECT unless the evidence is solid.
Claim: strategies/exp_w${WAVE}_${exp.slug}.py with overrides ${res.best_overrides} beats the champion.
Experimenter's summary: ${res.summary}
Check: (1) research.compare --a ${CHAMPION} --b ${res.file} --b-set '${res.best_overrides}' on ${ALL_DATA};
(2) plateau: run research.run with the neighbouring values of each tuned parameter — is the gain still there?
(3) is the gain spread over chunks or one lucky window? (4) read the diff for look-ahead, invariant violations,
sandbox-rule violations, per-event cost (research.check with runtime). ACCEPT only if compare says ACCEPT and
(2)-(4) hold. Do not edit any file.`, withModel('skeptic', { label: `skeptic:${exp.slug}`, phase: 'Skeptic', schema: VERDICT }))
      .then(v => (v ? { ...v, result: res } : null))
  },
)
const done = outcomes.filter(Boolean)
const accepted = done.filter(o => o.verdict === 'ACCEPT')
log(`${accepted.length}/${done.length} accepted: ${accepted.map(o => o.slug).join(', ') || 'none'}`)

phase('Integrate')
const integration = await agent(`${CTX}
You are the integrator for wave ${WAVE}. Outcomes of all experiments (accepted and rejected):
${JSON.stringify(done.map(o => ({ slug: o.slug, verdict: o.verdict, overrides: o.overrides, reasons: o.reasons, result: o.result })), null, 1)}
1. If any were accepted: build strategies/candidate_w${WAVE}.py = champion + the accepted mechanisms switched on
   with their overrides baked into PARAMS (you own this file). Run research.compare vs ${CHAMPION} on ${ALL_DATA}
   and research.check with runtime. If combining hurts, try the best single accepted experiment instead.
2. Write research/reports/wave_${STEM}_${WAVE}.md: every experiment with hypothesis, numbers, verdict and why.
3. Append to docs/progress.md one section per experiment using the template at the end of that file
   (branch herzic, round ${STEM}); add rows to the timeline table. Append only, never rewrite old text.
Return whether the candidate should be submitted (the team decides and promotes it to champion by hand).`,
  withModel('integrate', { label: 'integrate', phase: 'Integrate', schema: INTEGRATION }))

return { wave: WAVE, data: DATA, accepted: accepted.map(o => o.slug), rejected: done.filter(o => o.verdict !== 'ACCEPT').map(o => o.slug), integration }
