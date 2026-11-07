"""Lokální kontrola submise podle pravidel soutěže (přibližná, server je autorita).

Použití:
    uv run python research/check_submission.py strategies/momentum_hold.py

Kontroluje: velikost <= 1 MiB, povolené importy, zakázané builtiny, dunder atributy,
str.format, počet podtříd BaseStrategy a konstruktor bez argumentů.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ALLOWED_TOP = {'numpy', 'pandas', 'scipy', 'sklearn', 'math', 'collections', 'dataclasses', 'enum', 'typing',
               'itertools', 'functools', 'heapq', 'bisect', 'statistics', 'random', 'array', 'decimal',
               'fractions', 'operator', 'abc', 'copy', 'time', 'datetime', 'json', 're'}
BANNED_MODULES = {'os', 'sys', 'io', 'socket', 'subprocess', 'threading', 'multiprocessing', 'importlib',
                  'inspect', 'gc', 'pathlib', 'shutil', 'ctypes', 'builtins', 'pickle', 'marshal', 'signal'}
SIMULATOR_OK = {'simulator.models', 'simulator.strategy'}
BANNED_NAMES = {'open', 'eval', 'exec', 'compile', 'getattr', 'setattr', 'delattr', 'type', '__import__',
                'globals', 'locals', 'vars', 'input', 'breakpoint'}


def check(path: Path) -> list[str]:
    errs: list[str] = []
    src = path.read_text(encoding='utf-8')
    if len(src.encode()) > 1024 * 1024:
        errs.append('soubor > 1 MiB')
    tree = ast.parse(src)
    subclasses = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split('.')[0]
                if top in BANNED_MODULES or top == 'simulator' or top.startswith('_'):
                    errs.append(f'ř. {node.lineno}: import {a.name}')
                elif top not in ALLOWED_TOP:
                    errs.append(f'ř. {node.lineno}: import {a.name} (neověřený modul, zvaž)')
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            top = mod.split('.')[0]
            if mod == '__future__':
                continue
            if top == 'simulator':
                if mod not in SIMULATOR_OK:
                    errs.append(f'ř. {node.lineno}: from {mod}')
                elif mod == 'simulator.strategy' and any(a.name != 'BaseStrategy' for a in node.names):
                    errs.append(f'ř. {node.lineno}: ze simulator.strategy jen BaseStrategy')
            elif top in BANNED_MODULES or top.startswith('_'):
                errs.append(f'ř. {node.lineno}: from {mod}')
            elif top not in ALLOWED_TOP:
                errs.append(f'ř. {node.lineno}: from {mod} (neověřený modul, zvaž)')
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            errs.append(f'ř. {node.lineno}: zakázaný builtin {node.id}')
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith('__') and node.attr.endswith('__') and node.attr != '__init__':
                errs.append(f'ř. {node.lineno}: dunder atribut {node.attr}')
            if node.attr == 'format':
                errs.append(f'ř. {node.lineno}: .format (str.format je zakázané)')
        elif isinstance(node, ast.ClassDef):
            if any(isinstance(b, ast.Name) and b.id == 'BaseStrategy' for b in node.bases):
                subclasses.append(node)
    if not subclasses:
        errs.append('chybí podtřída BaseStrategy')
    for cls in subclasses:
        for item in cls.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                a = item.args
                if len(a.args) > 1 or a.vararg or a.kwonlyargs or a.kwarg:
                    errs.append(f'{cls.name}.__init__ má argumenty')
    if len(subclasses) > 1:
        errs.append(f'víc podtříd BaseStrategy ({", ".join(c.name for c in subclasses)}), použije se abecedně první')
    return errs


def main() -> None:
    bad = False
    for p in sys.argv[1:]:
        errs = check(Path(p))
        print(f'{p}: ' + ('OK' if not errs else 'PROBLÉMY'))
        for e in errs:
            print(f'  - {e}')
        bad |= bool(errs)
    raise SystemExit(1 if bad else 0)


if __name__ == '__main__':
    main()
