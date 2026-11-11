"""Bake mlp_fit --save pickles into a copy of an s14-family strategy.
usage: python -m research.diag.bake_s14 SRC DST [--taker PKL] [--quote PKL]  (quote -> MLPQ[500])"""
import argparse, pickle
from pathlib import Path


def r(v):
    return round(float(v), 6)


def lst(m, s, layers):
    return ([r(v) for v in m], [r(v) for v in s], [([[r(v) for v in row] for row in w], [r(v) for v in b]) for w, b in layers])


p = argparse.ArgumentParser()
p.add_argument('src'); p.add_argument('dst')
p.add_argument('--taker', default=''); p.add_argument('--quote', default='')
a = p.parse_args()
src = Path(a.src).read_text(encoding='utf-8')
if a.taker:
    m, s, L = lst(*pickle.loads(Path(a.taker).read_bytes()))
    i = src.index('MLP_MEAN ='); j = src.index(']\n', src.index('MLP_LAYERS =')) + 2
    body = f'MLP_MEAN = {m}\nMLP_SCALE = {s}\nMLP_LAYERS = [\n' + ''.join(f'    ({w}, {b}),\n' for w, b in L) + ']\n'
    src = src[:i] + body + src[j:]
if a.quote:
    q = lst(*pickle.loads(Path(a.quote).read_bytes()))
    i = src.index('MLPQ[500] ='); j = src.index('\n', i) + 1
    src = src[:i] + f'MLPQ[500] = {tuple(q)!r}\n' + src[j:]
Path(a.dst).write_text(src, encoding='utf-8')
print('wrote', a.dst)
