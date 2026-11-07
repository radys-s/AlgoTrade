"""One command for the start of a round: data in, kickoff report out.

    uv run python -m research.round data/round1.csv.gz --publish        # laptop: unzip, share, analyse
    uv run python -m research.round --fetch round1                      # home PC: download, analyse

Steps (failures are reported, never fatal; tools that do not exist yet are skipped):
1. unzip the round file (the .gz is kept), and with --publish upload it to a private GitHub
   release `data-<name>` so the home PC can fetch it with --fetch;
2. lane A: baselines (`research.run`) for the champion and the reference strategies on the round
   and every earlier dataset, then the sandbox check of the champion on the round data;
3. lane B: Tomáš's quick look (`analyze_tape.py`), features, data card, walk-forward signals,
   taker scan and market-making edge;
4. `research/reports/<name>_kickoff.md`: the insurance-submission verdict, baselines, links, timings.
Lanes A and B run in parallel; step logs go to logs/round/<name>/.
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = 'radys-s/AlgoTrade'
CHAMPION = 'submission/best_strategy.py'
BASELINES = [CHAMPION, 'strategies/mm.py', 'strategies/trend_follower.py']
REPORTS = Path('research/reports')
SYNTHETIC = {'sample'}  # the starter's synthetic trend: a different world, never a robustness check
MAX_PROJECTED_S = 7 * 60  # the sandbox allows 15 min on one CPU; keep a 2x margin


def gh_executable() -> str | None:
    """The GitHub CLI, also when the per-user install is not on this shell's PATH yet."""
    found = shutil.which('gh')
    if found:
        return found
    fallback = Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'gh' / 'bin' / 'gh.exe'
    return str(fallback) if fallback.exists() else None


class Step:
    """One logged subprocess of the pipeline."""

    def __init__(self, name: str, ok: bool, seconds: float, output: str, skipped: str = '') -> None:
        self.name, self.ok, self.seconds, self.output, self.skipped = name, ok, seconds, output, skipped


def run_step(name: str, command: list[str], log_dir: Path, timeout: int = 3600) -> Step:
    """Run `command`, keep its output in log_dir/<name>.log, never raise."""
    started = time.perf_counter()
    print(f'[{datetime.now():%H:%M:%S}] start {name}', flush=True)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
        )
        output = completed.stdout + completed.stderr
        ok = completed.returncode == 0
    except (OSError, subprocess.SubprocessError) as error:
        output, ok = f'{type(error).__name__}: {error}', False
    seconds = time.perf_counter() - started
    (log_dir / f'{name}.log').write_text(output, encoding='utf-8')
    print(f'[{datetime.now():%H:%M:%S}] {"done" if ok else "FAILED"} {name} ({seconds:.0f}s)', flush=True)
    return Step(name, ok, seconds, output)


def skipped(name: str, reason: str) -> Step:
    print(f'[{datetime.now():%H:%M:%S}] skip {name}: {reason}', flush=True)
    return Step(name, True, 0.0, '', skipped=reason)


def unzip(path: Path, name: str) -> tuple[Path, Path | None]:
    """The plain CSV for the simulator (decompressed next to the .gz, which is kept) and the .gz if any."""
    if path.suffix == '.gz':
        csv_path = path.parent / f'{name}.csv'
        if not csv_path.exists() or csv_path.stat().st_mtime < path.stat().st_mtime:
            with gzip.open(path, 'rb') as source, csv_path.open('wb') as sink:
                shutil.copyfileobj(source, sink, length=1 << 22)
        return csv_path, path
    return path, None


def publish(gz_path: Path | None, csv_path: Path, name: str, log_dir: Path) -> Step:
    """Upload the compressed round file to the private release data-<name> (created if missing)."""
    gh = gh_executable()
    if gh is None:
        return skipped('publish', 'GitHub CLI not found')
    if gz_path is None:
        gz_path = csv_path.with_suffix('.csv.gz')
        with csv_path.open('rb') as source, gzip.open(gz_path, 'wb', compresslevel=6) as sink:
            shutil.copyfileobj(source, sink, length=1 << 22)
    tag = f'data-{name}'
    exists = subprocess.run([gh, 'release', 'view', tag, '-R', REPO], capture_output=True).returncode == 0
    if exists:
        return run_step('publish', [gh, 'release', 'upload', tag, str(gz_path), '-R', REPO, '--clobber'], log_dir)
    notes = f'Public backtest data for {name}, uploaded {datetime.now():%Y-%m-%d %H:%M}.'
    command = [gh, 'release', 'create', tag, str(gz_path), '-R', REPO, '--title', f'{name} data', '--notes', notes]
    return run_step('publish', command, log_dir)


def fetch(name: str) -> Path:
    """Download data-<name> from the private release into data/ and return the downloaded file."""
    gh = gh_executable()
    if gh is None:
        raise SystemExit('GitHub CLI not found: install it and run gh auth login first')
    Path('data').mkdir(exist_ok=True)
    subprocess.run([gh, 'release', 'download', f'data-{name}', '-R', REPO, '-D', 'data', '--clobber'], check=True)
    listing = subprocess.run(
        [gh, 'release', 'view', f'data-{name}', '-R', REPO, '--json', 'assets', '-q', '.assets[].name'],
        capture_output=True,
        text=True,
        check=True,
    )
    asset = listing.stdout.split()[0]
    return Path('data') / asset


def lane_a(csv_path: Path, prior: list[Path], name: str, workers: int, log_dir: Path) -> list[Step]:
    """Baselines on all datasets, then the sandbox check of the champion on the round data."""
    steps = []
    datasets = [str(csv_path), *map(str, prior)]
    for strategy in BASELINES:
        if not Path(strategy).exists():
            steps.append(skipped(f'baseline_{Path(strategy).stem}', 'file missing'))
            continue
        command = [sys.executable, '-m', 'research.run', '--strategy', strategy, '--data', *datasets,
                   '--workers', str(workers), '--tag', f'{name}-baseline']  # fmt: skip
        steps.append(run_step(f'baseline_{Path(strategy).stem}', command, log_dir))
    command = [sys.executable, '-m', 'research.check', CHAMPION, '--data', str(csv_path)]
    steps.append(run_step('check_champion', command, log_dir))
    return steps


def lane_b(csv_path: Path, name: str, log_dir: Path) -> list[Step]:
    """Quick look, features, data card, signals, taker scan, market-making edge (each needs the previous)."""
    steps = []
    stem = csv_path.stem
    npz = Path('data/features') / f'{stem}.npz'
    if Path('research/analyze_tape.py').exists():
        steps.append(run_step('analyze_tape', [sys.executable, '-X', 'utf8', 'research/analyze_tape.py', str(csv_path)], log_dir))
    steps.append(run_step('features', [sys.executable, '-m', 'research.features', str(csv_path)], log_dir))
    steps.append(run_step('profile', [sys.executable, '-m', 'research.profile', str(csv_path)], log_dir))
    for horizon in (100, 1000):
        command = [sys.executable, '-m', 'research.signals', str(npz), '--horizon-ms', str(horizon)]
        steps.append(run_step(f'signals_h{horizon}ms', command, log_dir))
    for module, argument in (('taker_scan', str(npz)), ('mm_edge', str(csv_path))):
        if Path(f'research/{module}.py').exists():
            steps.append(run_step(module, [sys.executable, '-m', f'research.{module}', argument], log_dir))
        else:
            steps.append(skipped(module, 'tool not built yet'))
    return steps


def baseline_table(steps: list[Step]) -> pd.DataFrame:
    """The baseline rows of this kickoff, read back from the ledger files the runs wrote."""
    frames = []
    for step in steps:
        match = re.search(r'ledger: (\S+\.csv)', step.output)
        if step.name.startswith('baseline_') and match and Path(match.group(1)).exists():
            frames.append(pd.read_csv(match.group(1)))
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames)
    frame['strategy'] = frame['strategy'].map(lambda path: Path(path).stem)
    return frame[['strategy', 'dataset', 'pnl', 'train_pnl', 'hold_pnl', 'chunk_wins', 'fills', 'breach_penalty',
                  'inv_q997', 'runtime_s', 'error']]  # fmt: skip


def insurance_verdict(table: pd.DataFrame, check: Step | None, round_name: str) -> str:
    """Whether the champion can be submitted right now as insurance, and why (not)."""
    if table.empty:
        return 'NO VERDICT: the baselines did not run, see the step logs.'
    rows = table[table['dataset'] == round_name]
    champion = rows[rows['strategy'] == Path(CHAMPION).stem]
    projected = re.search(r'projected\s+([\d,.]+) s', check.output if check else '')
    projected_s = float(projected.group(1).replace(',', '')) if projected else None
    passed = check is not None and '== PASS' in check.output
    if champion.empty or (isinstance(champion['error'].iloc[0], str) and champion['error'].iloc[0]):
        return 'DO NOT SUBMIT the champion: it did not run on the round data (see baseline log).'
    pnl, breach = float(champion['pnl'].iloc[0]), float(champion['breach_penalty'].iloc[0])
    problems = []
    if pnl <= 0:
        problems.append(f'public pnl {pnl:,.0f} <= 0')
    if breach > 0:
        problems.append(f'breach penalty {breach:,.0f}')
    if not passed:
        problems.append('research.check did not PASS')
    if projected_s is None or projected_s > MAX_PROJECTED_S:
        problems.append(f'projected runtime {projected_s} s (limit {MAX_PROJECTED_S} s with margin)')
    if not problems:
        return (f'SUBMIT {CHAMPION} NOW as insurance: public pnl {pnl:,.0f}, no breach, check PASS, '
                f'projected {projected_s:.0f} s. Then make sure it is marked final.')  # fmt: skip
    others = rows[(rows['strategy'] != Path(CHAMPION).stem) & (rows['pnl'] > 0) & (rows['breach_penalty'] == 0)]
    fallback = (
        f' Positive baseline instead: {others.sort_values("pnl").iloc[-1]["strategy"]}.' if not others.empty else
        ' No baseline is positive: wait for wave 1.'
    )  # fmt: skip
    return 'DO NOT SUBMIT the champion yet: ' + '; '.join(problems) + '.' + fallback


def write_report(name: str, csv_path: Path, prior: list[Path], steps: list[Step], verdict: str, table: pd.DataFrame) -> Path:
    """research/reports/<name>_kickoff.md."""
    stem = csv_path.stem
    rows = count_rows(csv_path)
    lines = [
        f'# Kickoff {name}',
        '',
        f'Generated {datetime.now():%Y-%m-%d %H:%M} by research.round. Data `{csv_path.as_posix()}` ({rows:,} rows); '
        f'earlier datasets: {", ".join(p.stem for p in prior) or "none"}.',
        '',
        '## Insurance submission',
        '',
        f'**{verdict}**',
        '',
        '## Baselines (ledger tag `' + f'{name}-baseline`)',
        '',
        markdown_table(table) if not table.empty else '_no baseline results_',
        '',
        '## Reports',
        '',
    ]
    for label, path in (
        ('Quick look (analyze_tape)', Path('logs/round') / name / 'analyze_tape.log'),
        ('Data card', REPORTS / f'{stem}.md'),
        ('Signals 100 ms', REPORTS / f'{stem}_signals_h100ms.md'),
        ('Signals 1 s', REPORTS / f'{stem}_signals_h1000ms.md'),
        ('Taker scan', REPORTS / f'{stem}_taker_scan.md'),
        ('Market-making edge', REPORTS / f'{stem}_mm_edge.md'),
    ):
        lines.append(f'- {label}: `{path.as_posix()}`' + ('' if path.exists() else ' (missing)'))
    lines += ['', '## Steps', '', '| step | status | seconds |', '|---|---|---|']
    for step in steps:
        status = f'skipped: {step.skipped}' if step.skipped else ('ok' if step.ok else 'FAILED')
        lines.append(f'| {step.name} | {status} | {step.seconds:.0f} |')
    prior_json = ', '.join(f'"{p.as_posix()}"' for p in prior)
    lines += [
        '',
        '## Next',
        '',
        '1. Read the data card and the quick look: momentum or reversion, spread regime, which signals are stable.',
        '2. Start research wave 1 (Workflow tool, `scriptPath: research/workflows/research_wave.js`), args:',
        f'   `{{"data": "{csv_path.as_posix()}", "prior": [{prior_json}], "champion": "{CHAMPION}", "wave": 1, "workers": 2}}`',
        '3. Confirm the top taker-scan configs in the real simulator (commands in the taker-scan report).',
    ]
    path = REPORTS / f'{name}_kickoff.md'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def count_rows(path: Path) -> int:
    """Data rows of a CSV (lines minus the header)."""
    with path.open('rb') as file:
        return max(sum(chunk.count(b'\n') for chunk in iter(lambda: file.read(1 << 22), b'')) - 1, 0)


def markdown_table(frame: pd.DataFrame) -> str:
    """A plain markdown table (pandas' to_markdown needs tabulate, which the project does not ship)."""
    frame = frame.fillna('')
    header = '| ' + ' | '.join(map(str, frame.columns)) + ' |'
    rule = '|' + '---|' * len(frame.columns)
    body = ['| ' + ' | '.join(str(value) for value in row) + ' |' for row in frame.itertuples(index=False)]
    return '\n'.join([header, rule, *body])


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description='Round kickoff: data in, kickoff report out')
    parser.add_argument('data', nargs='?', type=Path, help='Round file from the arena (.csv or .csv.gz)')
    parser.add_argument('--fetch', metavar='NAME', help='Download data-NAME from the private release instead')
    parser.add_argument('--name', help='Round name (default: file name without .csv/.gz)')
    parser.add_argument('--publish', action='store_true', help='Upload the round file for the home PC')
    parser.add_argument(
        '--prior', type=Path, nargs='*', help='Earlier datasets (default: every other data/*.csv except the synthetic sample)'
    )
    parser.add_argument('--workers', type=int, default=max(2, (os.cpu_count() or 8) - 6), help='Baseline workers')
    return parser.parse_args()


def main() -> None:
    """Run both lanes and write the kickoff report."""
    args = parse_args()
    started = time.perf_counter()
    if args.fetch:
        source = fetch(args.fetch)
        name = args.name or args.fetch
    elif args.data:
        source = args.data
        name = args.name or source.name.removesuffix('.gz').removesuffix('.csv')
    else:
        raise SystemExit('give the round file, or --fetch NAME')
    csv_path, gz_path = unzip(source, name)
    log_dir = Path('logs/round') / name
    log_dir.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if args.prior is None:
        prior = sorted(
            p for p in Path('data').glob('*.csv') if p.resolve() != csv_path.resolve() and p.stem not in SYNTHETIC
        )
    else:
        prior = list(args.prior)
    print(f'round {name}: {csv_path.as_posix()}, earlier datasets: {[p.stem for p in prior]}', flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        publishing = pool.submit(publish, gz_path, csv_path, name, log_dir) if args.publish else None
        a = pool.submit(lane_a, csv_path, prior, name, args.workers, log_dir)
        b = pool.submit(lane_b, csv_path, name, log_dir)
        steps = ([publishing.result()] if publishing else []) + a.result() + b.result()

    table = baseline_table(steps)
    check = next((step for step in steps if step.name == 'check_champion'), None)
    verdict = insurance_verdict(table, check, csv_path.stem)
    report = write_report(name, csv_path, prior, steps, verdict, table)
    print(f'\n{verdict}\nkickoff report: {report.as_posix()}  ({time.perf_counter() - started:.0f}s total)')


if __name__ == '__main__':
    main()
