"""Pre-upload check of a strategy file: the sandbox's static rules, then a timed smoke run.

    uv run python -m research.check strategies/foo.py [--data data/sample.csv] [--rows 200000] [--no-run]

Static findings are ERROR where README.md / docs/kontext.md state the rule and WARN where
the rule is a guess (the sandbox's banned list is explicitly incomplete). The smoke run loads
the strategy the way `simulator/main.py` does, replays the first --rows rows with the round's
latency and limits, compares it with an empty strategy on the same rows and projects the wall
time of a full round. Exit code 1 if anything is an ERROR.
"""

from __future__ import annotations

import argparse
import ast
import io
import sys
import tempfile
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from simulator.main import (
    MAX_POSITION,
    MAX_POTENTIAL_POSITION,
    ORDER_ACCEPT_DELAY_MS,
    POSITION_BREACH_PENALTY_PER_SECOND,
    load_strategy_from_file,
)
from simulator.models import OrderBookSnapshot, SimulationResult, Trade
from simulator.simulator import ExchangeSimulator, SimulationConfig
from simulator.strategy import BaseStrategy

ERROR = 'ERROR'
WARN = 'WARN'

MAX_FILE_BYTES = 1 << 20
FULL_ROUND_ROWS = 8_800_000
WARN_WALL_S = 7 * 60
MAX_WALL_S = 15 * 60
REPEATS = 2  # replays of each strategy when one empty replay takes under REPEAT_BELOW_S
REPEAT_BELOW_S = 10.0
LOG_DIR = Path(__file__).resolve().parent.parent / 'logs'

# Stated in README.md / docs/kontext.md: rejected before the run.
BANNED_MODULES: dict[str, str] = {
    'os': 'system access',
    'sys': 'interpreter access',
    'io': 'file access',
    'socket': 'network access',
    'subprocess': 'process access',
    'threading': 'threading',
    'importlib': 'dynamic imports',
    'inspect': 'introspection',
    'gc': 'introspection',
    'pathlib': 'file access',
}
# Not stated, but the kind of access the sandbox says it rejects.
RISKY_MODULES: dict[str, str] = {
    **dict.fromkeys(
        ('multiprocessing', 'concurrent', 'asyncio', 'queue', 'signal', 'atexit'),
        'concurrency / process control',
    ),
    **dict.fromkeys(
        ('ctypes', 'posix', 'nt', 'pwd', 'grp', 'fcntl', 'msvcrt', 'winreg', 'winsound', 'resource', 'mmap',
         'platform', 'sysconfig', 'shutil', 'tempfile', 'glob', 'getpass', 'pty', 'tty', 'termios', 'select',
         'selectors', 'faulthandler', 'site', 'venv', 'ensurepip'),
        'system access',
    ),
    **dict.fromkeys(
        ('pickle', 'marshal', 'shelve', 'dbm', 'sqlite3', 'fileinput', 'linecache', 'zipfile', 'tarfile', 'gzip',
         'bz2', 'lzma', 'zipimport', 'logging', 'configparser', 'netrc', 'plistlib', 'mimetypes', 'filecmp',
         'tokenize', 'tabnanny', 'py_compile', 'compileall', 'zipapp', 'mailbox', 'copyreg', 'pickletools'),
        'file access / serialisation',
    ),
    **dict.fromkeys(
        ('socketserver', 'urllib', 'http', 'ftplib', 'smtplib', 'poplib', 'imaplib', 'ssl', 'xmlrpc', 'wsgiref',
         'webbrowser'),
        'network access',
    ),
    **dict.fromkeys(
        ('builtins', 'types', 'weakref', 'traceback', 'dis', 'ast', 'code', 'codeop', 'runpy', 'pkgutil',
         'modulefinder', 'symtable', 'pyclbr', 'opcode', 'tracemalloc', 'trace', 'pdb', 'bdb', 'cProfile',
         'profile', 'pstats', 'pydoc', 'doctest', 'unittest', 'timeit'),
        'introspection / code execution',
    ),
    **dict.fromkeys(('tkinter', 'turtle', 'turtledemo', 'idlelib'), 'GUI'),
}
ALLOWED_THIRD_PARTY = frozenset({'numpy', 'pandas', 'scipy', 'sklearn'})
RISKY_SUBMODULES: dict[str, str] = {
    'numpy.ctypeslib': 'raw memory access',
    'numpy.f2py': 'compiles and runs code',
    'numpy.lib.format': 'file access',
    'pandas.io': 'file access',
    'scipy.io': 'file access',
    'scipy.datasets': 'network download',
    'sklearn.datasets': 'file / network access',
}

BANNED_NAMES: dict[str, str] = {
    'open': 'file access',
    'eval': 'code execution',
    'exec': 'code execution',
    'compile': 'code execution',
    'getattr': 'introspection',
    'type': 'introspection',
    '__import__': 'dynamic import',
}
RISKY_NAMES: dict[str, str] = {
    **dict.fromkeys(('setattr', 'delattr', 'hasattr', 'globals', 'locals', 'vars', 'dir', 'help'), 'introspection'),
    'input': 'reads stdin',
    'breakpoint': 'debugger',
    'exit': 'process control',
    'quit': 'process control',
}
RISKY_ATTRIBUTES: dict[str, str] = {
    **dict.fromkeys(
        ('fromfile', 'tofile', 'memmap', 'load', 'loadtxt', 'genfromtxt', 'save', 'savez', 'savez_compressed',
         'savetxt', 'dump', 'ctypeslib', 'DataSource', 'read_csv', 'read_table', 'read_fwf', 'read_pickle',
         'read_parquet', 'read_json', 'read_excel', 'read_hdf', 'read_feather', 'read_sql', 'read_html', 'read_xml',
         'read_clipboard', 'to_csv', 'to_pickle', 'to_parquet', 'to_json', 'to_excel', 'to_hdf', 'to_feather',
         'to_sql', 'to_clipboard'),
        'file access through an allowed library; probably blocked at run time or seen as circumvention',
    ),
    **dict.fromkeys(('format_map', 'Formatter'), 'same attribute-lookup risk as str.format; probably rejected'),
    **dict.fromkeys(('attrgetter', 'methodcaller'), 'a getattr equivalent; may be rejected as introspection'),
    'eval': 'evaluates a string (pandas eval); may be rejected like the eval builtin',
    **dict.fromkeys(
        ('f_globals', 'f_locals', 'f_builtins', 'f_back', 'f_code', 'gi_frame', 'gi_code', 'cr_frame', 'ag_frame',
         'tb_frame', 'co_code', 'mro'),
        'frame / class introspection, a classic sandbox escape',
    ),
    **dict.fromkeys(
        ('_ctx', '_attach_context'),
        'BaseStrategy internals; use send_order / cancel_order / modify_order / now_us',
    ),
}
BOOK_CALLBACK = 'on_order_book_change'


@dataclass(frozen=True, slots=True)
class Finding:
    """One problem: its level, the line it is on (None for the whole file or the run) and what is wrong."""

    level: str
    line: int | None
    message: str

    def render(self, path: str) -> str:
        """Format as `path:line: LEVEL message`."""
        where = path if self.line is None else f'{path}:{self.line}'
        return f'{where}: {self.level} {self.message}'


def is_dunder(name: str) -> bool:
    """Return whether `name` has the `__x__` form."""
    return len(name) > 4 and name.startswith('__') and name.endswith('__')


class RuleVisitor(ast.NodeVisitor):
    """Flag the imports, names and attributes the sandbox rejects (ERROR) or probably rejects (WARN)."""

    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.base_names: set[str] = {'BaseStrategy'}
        self.module_aliases: set[str] = set()

    def add(self, level: str, line: int, message: str) -> None:
        """Record a finding."""
        self.findings.append(Finding(level, line, message))

    def visit_Import(self, node: ast.Import) -> None:
        """Check every `import a.b [as c]`."""
        for alias in node.names:
            self.check_module(alias.name, node.lineno)
            self.check_alias(alias, node.lineno, imported_from=False)
            self.module_aliases.add(alias.asname or alias.name.split('.')[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check `from a import b`, with the simulator allowed only as `simulator.models` and `BaseStrategy`."""
        line = node.lineno
        module = node.module or ''
        if node.level:
            self.add(ERROR, line, f"relative import 'from {'.' * node.level}{module} import ...' is rejected")
            return
        if module == 'simulator.strategy':
            for alias in node.names:
                if alias.name == 'BaseStrategy':
                    self.base_names.add(alias.asname or alias.name)
                else:
                    self.add(ERROR, line, f"'from simulator.strategy import {alias.name}': only BaseStrategy is allowed")
        elif module == 'simulator':
            for alias in node.names:
                if alias.name == 'models':
                    self.add(WARN, line, "'from simulator import models' imports the simulator package itself; "
                                         "write 'from simulator.models import ...' to match the rule literally")
                else:
                    self.add(ERROR, line, f"'from simulator import {alias.name}': only simulator.models and "
                                          'simulator.strategy.BaseStrategy may be imported')
        else:
            self.check_module(module, line)
            if module.split('.')[0] in ALLOWED_THIRD_PARTY:
                for alias in node.names:
                    self.check_submodule(f'{module}.{alias.name}', line)
        for alias in node.names:
            self.check_alias(alias, line, imported_from=True)
            if alias.name == '*':
                self.add(WARN, line, f"'from {module} import *' hides which names are used; import them by name")

    def check_module(self, name: str, line: int) -> None:
        """Classify one imported module by its dotted name."""
        if name == '__future__':
            return
        parts = name.split('.')
        top = parts[0]
        if top == 'simulator':
            if name != 'simulator.models':
                self.add(ERROR, line, f"import of '{name}': only simulator.models and "
                                      'simulator.strategy.BaseStrategy may be imported')
        elif any(part.startswith('_') for part in parts):
            self.add(ERROR, line, f"import of private module '{name}' is rejected")
        elif top in BANNED_MODULES:
            self.add(ERROR, line, f"import of '{name}' is rejected ({BANNED_MODULES[top]})")
        elif top in RISKY_MODULES:
            self.add(WARN, line, f"import of '{name}' is probably rejected ({RISKY_MODULES[top]})")
        elif top in ALLOWED_THIRD_PARTY:
            self.check_submodule(name, line)
        elif top not in sys.stdlib_module_names:
            self.add(ERROR, line, f"import of '{name}': not in the standard library and not numpy / pandas / "
                                  'scipy / sklearn, so not available in the sandbox')

    def check_submodule(self, name: str, line: int) -> None:
        """Warn about the parts of the allowed libraries that reach files, the network or raw memory."""
        for prefix, reason in RISKY_SUBMODULES.items():
            if name == prefix or name.startswith(prefix + '.'):
                self.add(WARN, line, f"'{name}' is probably rejected or blocked ({reason})")

    def check_alias(self, alias: ast.alias, line: int, *, imported_from: bool) -> None:
        """Check the bound name and, in `from m import x`, the imported name (a module path is checked apart)."""
        names = (alias.name, alias.asname or '') if imported_from else (alias.asname or '',)
        for name in names:
            if is_dunder(name) and name != '__init__':
                self.add(ERROR, line, f"dunder name '{name}' is rejected (only __init__ is allowed)")
            elif name.startswith('_') and not is_dunder(name):
                self.add(WARN, line, f"private name '{name}' (private modules are rejected)")

    def visit_Name(self, node: ast.Name) -> None:
        """Banned builtins and dunder names, wherever they are used or bound."""
        name = node.id
        if name in BANNED_NAMES:
            self.add(ERROR, node.lineno, f"name '{name}' is rejected (banned builtin: {BANNED_NAMES[name]})")
        elif is_dunder(name) and name != '__init__':
            self.add(ERROR, node.lineno, f"dunder name '{name}' is rejected (only __init__ is allowed)")
        elif name in RISKY_NAMES:
            self.add(WARN, node.lineno, f"name '{name}' is probably rejected (builtin: {RISKY_NAMES[name]})")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Dunder attributes, `.format` and attributes that reach outside the strategy."""
        attr = node.attr
        if is_dunder(attr) and attr != '__init__':
            self.add(ERROR, node.lineno, f"dunder attribute '.{attr}' is rejected (only __init__ is allowed)")
        elif attr == 'format':
            self.add(ERROR, node.lineno, "'.format' is rejected as str.format (the receiver type is unknown "
                                         'statically); use an f-string')
        elif attr in RISKY_ATTRIBUTES:
            self.add(WARN, node.lineno, f"'.{attr}': {RISKY_ATTRIBUTES[attr]}")
        elif attr.startswith('_') and isinstance(node.value, ast.Name) and node.value.id in self.module_aliases:
            self.add(WARN, node.lineno, f"private name '{node.value.id}.{attr}' (private modules are rejected)")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Dunder method definitions other than `__init__`."""
        if is_dunder(node.name) and node.name != '__init__':
            self.add(WARN, node.lineno, f"defines dunder method '{node.name}': the rule rejects dunder attributes; "
                                        'a def is not an attribute access, but a checker that scans every '
                                        'identifier would reject it, so avoid it unless needed')
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Constant(self, node: ast.Constant) -> None:
        """A string that is exactly a dunder name, the usual argument of a dynamic attribute lookup."""
        if isinstance(node.value, str) and is_dunder(node.value) and node.value != '__init__':
            self.add(WARN, node.lineno, f"string '{node.value}' looks like a dynamic dunder lookup")


def base_name(node: ast.expr) -> str | None:
    """Return the class name a base-class expression refers to (`X` or `module.X`)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def check_strategy_class(tree: ast.Module, base_names: set[str]) -> list[Finding]:
    """Exactly one `BaseStrategy` subclass, constructible without arguments, overriding the book callback."""
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    strategies: dict[str, ast.ClassDef] = {}
    changed = True
    while changed:
        changed = False
        for cls in classes:
            names = {base_name(base) for base in cls.bases}
            if cls.name not in strategies and names & (base_names | set(strategies)):
                strategies[cls.name] = cls
                changed = True

    if not strategies:
        return [Finding(ERROR, None, 'no class subclasses BaseStrategy; the loader raises ValueError')]
    findings: list[Finding] = []
    if len(strategies) > 1:
        listed = ', '.join(sorted(strategies))
        findings.append(Finding(ERROR, strategies[min(strategies)].lineno,
                                f'{len(strategies)} BaseStrategy subclasses ({listed}); the rules allow one '
                                f"(the loader would instantiate '{min(strategies)}', the alphabetically first)"))

    for cls in strategies.values():
        for item in cls.body:
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and item.name == '__init__':
                findings.extend(check_init(cls.name, item))

    chosen = strategies[min(strategies)]
    methods: set[str] = set()
    lineage = {chosen.name}
    pending = [chosen]
    while pending:
        cls = pending.pop()
        methods |= {item.name for item in cls.body if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)}
        for base in cls.bases:
            if (name := base_name(base)) in strategies and name not in lineage:
                lineage.add(name)
                pending.append(strategies[name])
    if BOOK_CALLBACK not in methods:
        findings.append(Finding(WARN, chosen.lineno, f"'{chosen.name}' does not override {BOOK_CALLBACK}, "
                                                     'so it never sends an order'))
    return findings


def check_init(class_name: str, init: ast.FunctionDef | ast.AsyncFunctionDef) -> list[Finding]:
    """The loader calls `Cls()`: `__init__` may take nothing but `self`."""
    args = init.args
    positional = args.posonlyargs + args.args
    required = max(len(positional) - len(args.defaults) - 1, 0)
    required += sum(default is None for default in args.kw_defaults)
    if required:
        return [Finding(ERROR, init.lineno, f'{class_name}.__init__ requires {required} argument(s) besides self; '
                                            f'the loader calls {class_name}() with none')]
    if len(positional) > 1 or args.kwonlyargs or args.vararg or args.kwarg:
        return [Finding(WARN, init.lineno, f'{class_name}.__init__ takes optional parameters that are never passed; '
                                           'the rules ask for a constructor without arguments')]
    return []


def static_check(path: Path) -> tuple[list[Finding], bool]:
    """Apply the static rules to the strategy file; return the findings and whether it parsed."""
    findings: list[Finding] = []
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        findings.append(Finding(ERROR, None, f'file is {size:,} bytes; the limit is 1 MiB ({MAX_FILE_BYTES:,} bytes)'))
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except SyntaxError as exc:
        findings.append(Finding(ERROR, exc.lineno, f'syntax error: {exc.msg}'))
        return findings, False
    visitor = RuleVisitor()
    visitor.visit(tree)
    findings += visitor.findings
    findings += check_strategy_class(tree, visitor.base_names)
    return sorted(findings, key=lambda finding: finding.line or 0), True


@dataclass(slots=True)
class RunStats:
    """Outcome and wall-clock timing of one replay."""

    result: SimulationResult | None
    seconds: float
    half_seconds: float | None
    events: int
    sim_time_us: int
    error: str | None


def round_config() -> SimulationConfig:
    """The round's settings as `simulator/main.py` sets them, with every log disabled."""
    return SimulationConfig(
        max_position=MAX_POSITION,
        max_potential_position=MAX_POTENTIAL_POSITION,
        position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND,
        order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS,
    )


def scan_rows(source: Path, rows: int | None, target: Path | None) -> tuple[int, float]:
    """Read the header and the first `rows` data rows (all if None), copying them to `target` if given.

    Returns the number of rows and the market time they span, in seconds.
    """
    with source.open('rb') as src, (target.open('wb') if target else nullcontext()) as out:
        header = src.readline()
        time_index = header.decode().strip().split(',').index('Time')
        if out is not None:
            out.write(header)
        count = 0
        first = last = b''
        for line in islice(src, rows):
            if out is not None:
                out.write(line)
            if not count:
                first = line
            last = line
            count += 1
    if not count:
        return 0, 0.0
    return count, float(last.split(b',')[time_index]) - float(first.split(b',')[time_index])


def timed_run(strategy: BaseStrategy, backtest: Path, rows: int) -> RunStats:
    """Replay `backtest` through `strategy`, noting the wall time at the halfway event."""
    simulator = ExchangeSimulator(strategy=strategy, config=round_config())
    callback = strategy.on_order_book_change
    half = max(rows // 2, 1)
    events = 0
    half_at: float | None = None

    def counting_callback(snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        nonlocal events, half_at
        events += 1
        if events == half:
            half_at = time.perf_counter()
        callback(snapshot, trades)

    strategy.on_order_book_change = counting_callback  # type: ignore[method-assign]
    result: SimulationResult | None = None
    error: str | None = None
    start = time.perf_counter()
    try:
        result = simulator.run(backtest)
    except Exception:
        error = traceback.format_exc()
    seconds = time.perf_counter() - start
    half_seconds = None if half_at is None else half_at - start
    return RunStats(result, seconds, half_seconds, events, simulator.current_ts_us, error)


def print_traceback(text: str) -> None:
    """Print a traceback indented under the report."""
    for line in text.rstrip().splitlines():
        print(f'    | {line}')


def smoke_run(strategy_path: Path, data: Path, rows: int) -> list[Finding]:
    """Replay the first `rows` rows of `data` (all if 0) through the strategy and an empty one; print the report."""
    print(f'== smoke run: {data.as_posix()}, first {rows:,} rows' if rows else f'== smoke run: {data.as_posix()}')
    try:
        strategy = load_strategy_from_file(strategy_path)
    except Exception:
        print('  crashed         yes, while loading')
        print_traceback(traceback.format_exc())
        return [Finding(ERROR, None, 'crashed while loading the strategy (traceback above)')]

    LOG_DIR.mkdir(exist_ok=True)
    slice_path: Path | None = None
    try:
        if rows:
            with tempfile.NamedTemporaryFile(dir=LOG_DIR, prefix='_check_', suffix='.csv', delete=False) as file:
                slice_path = Path(file.name)
            count, span_s = scan_rows(data, rows, slice_path)
            backtest = slice_path
        else:
            count, span_s = scan_rows(data, None, None)
            backtest = data
        print(f'  rows            {count:,} ({span_s:,.1f} s of market time), latency {ORDER_ACCEPT_DELAY_MS} ms, '
              f'limits {MAX_POSITION}/{MAX_POTENTIAL_POSITION}')
        if not count:
            return [Finding(ERROR, None, f'{data.as_posix()} has no data rows')]
        # Empty and strategy replays alternate, the first empty one warming up the file cache and the
        # simulator; short slices get two of each and the faster of each counts.
        empties = [timed_run(BaseStrategy(), backtest, count)]
        if empties[0].error is not None:
            print('  replay failed even with an empty strategy')
            print_traceback(empties[0].error)
            return [Finding(ERROR, None, 'the data does not replay (traceback above)')]
        users: list[RunStats] = []
        for repeat in range(REPEATS if empties[0].seconds < REPEAT_BELOW_S else 1):
            users.append(timed_run(strategy if not repeat else load_strategy_from_file(strategy_path), backtest, count))
            if users[-1].error is not None:
                break
            empties.append(timed_run(BaseStrategy(), backtest, count))
    finally:
        if slice_path is not None:
            slice_path.unlink(missing_ok=True)

    first = users[0]
    if first.error is not None or first.result is None:
        print(f'  crashed         yes, at event {first.events:,} (t = {first.sim_time_us / 1e6:,.3f} s)')
        print_traceback(first.error or '')
        return [Finding(ERROR, None, f'crashed at event {first.events:,} of the replay (traceback above)')]
    result = first.result
    print('  crashed         no')
    print(f'  pnl             {result.pnl:,.2f}')
    print(f'  fills           {result.fills:,}')
    print(f'  breach_penalty  {result.breach_penalty:,.2f}')
    print(f'  inventory       {result.inventory}')
    fastest_user = min(users, key=lambda stats: stats.seconds)
    return report_timing(fastest_user, min(empties, key=lambda stats: stats.seconds), count)


def report_timing(user: RunStats, empty: RunStats, count: int) -> list[Finding]:
    """Print events/s, the strategy's own cost per event and the projected wall time of a full round."""
    findings: list[Finding] = []
    overhead_us = (user.seconds - empty.seconds) / count * 1e6
    per_event_s = user.seconds / count
    halves = ''
    if user.half_seconds is not None and empty.half_seconds is not None and count >= 2:
        half = count // 2
        first_us = (user.half_seconds - empty.half_seconds) / half * 1e6
        second_us = ((user.seconds - user.half_seconds) - (empty.seconds - empty.half_seconds)) / (count - half) * 1e6
        halves = f' (first half {first_us:.1f}, second half {second_us:.1f})'
        # A cost that grows with the run is projected from the later, slower half.
        per_event_s = max(per_event_s, (user.seconds - user.half_seconds) / (count - half))
        if second_us > 2 * max(first_us, 0.0) + 2.0:
            findings.append(Finding(WARN, None, f'per-event cost grew from {first_us:.1f} to {second_us:.1f} us within '
                                                'the slice; it may keep growing over a full round (check memory '
                                                'and run with a larger --rows)'))
    projected_s = FULL_ROUND_ROWS * per_event_s
    print(f'  events/s        {count / user.seconds:,.0f} strategy vs {count / empty.seconds:,.0f} empty '
          f'({user.seconds:.2f} s vs {empty.seconds:.2f} s)')
    print(f'  overhead        {overhead_us:.1f} us/event over the empty strategy{halves}')
    print(f'  projected       {projected_s:,.0f} s ({projected_s / 60:.1f} min) for {FULL_ROUND_ROWS:,} rows at '
          f'{per_event_s * 1e6:.1f} us/event (linear from this slice; deeper books cost more)')
    if projected_s > MAX_WALL_S:
        findings.append(Finding(ERROR, None, f'projected {projected_s / 60:.1f} min for a full round exceeds the '
                                             f'{MAX_WALL_S // 60} min sandbox limit'))
    elif projected_s > WARN_WALL_S:
        findings.append(Finding(WARN, None, f'projected {projected_s / 60:.1f} min for a full round is over '
                                            f'{WARN_WALL_S // 60} min; the sandbox ({MAX_WALL_S // 60} min) may run '
                                            'on a slower CPU'))
    return findings


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog='python -m research.check',
        description='Check a strategy file against the sandbox rules and time a smoke run',
    )
    parser.add_argument('strategy', type=Path, help='Strategy .py file to check')
    parser.add_argument('--data', type=Path, default=Path('data/sample.csv'), help='Backtest CSV for the smoke run')
    parser.add_argument('--rows', type=int, default=200_000, help='Data rows to replay; 0 for the whole file')
    parser.add_argument('--no-run', action='store_true', help='Static checks only')
    args = parser.parse_args()
    if args.rows < 0:
        parser.error('--rows must be >= 0')
    if not args.strategy.is_file():
        parser.error(f'strategy file not found: {args.strategy}')
    if not args.no_run and not args.data.is_file():
        parser.error(f'data file not found: {args.data} (or pass --no-run)')
    if args.data.suffix == '.gz':
        parser.error('the simulator reads plain CSV; decompress first: uv run python -m gzip -d <file>.csv.gz')
    return args


def main() -> int:
    """Run the checks, print the findings and return the exit code (1 if any ERROR)."""
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors='replace')
    args = parse_args()
    path: Path = args.strategy
    shown = path.as_posix()

    print(f'== static checks: {shown} ({path.stat().st_size:,} bytes)')
    findings, parsed = static_check(path)
    for finding in findings:
        print(finding.render(shown))
    if not findings:
        print('  no findings')

    if not args.no_run and parsed:
        run_findings = smoke_run(path, args.data, args.rows)
        for finding in run_findings:
            print(finding.render(shown))
        findings += run_findings

    errors = sum(finding.level == ERROR for finding in findings)
    warnings = len(findings) - errors
    print(f'== {"FAIL" if errors else "PASS"}: {shown}, {errors} error(s), {warnings} warning(s)')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
