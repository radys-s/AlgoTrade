"""GRU taker predictor (research/diag/seq_r3.py, unchanged model and tokens) trained on an AVERAGED target: the mean of the
robust-mid move at t + 5 ms + {100, 200, 300, 400, 500} ms instead of the single 200 ms point. Taker oracles on mm_r3_gru
(research/diag/oracle_pred_r3.py): this averaged target is worth 1,066k on round3 vs 1,004k for the 200 ms point.

    $USERPROFILE/torch-env/Scripts/python -X utf8 research/diag/seqt_r3.py build
    $USERPROFILE/torch-env/Scripts/python -X utf8 research/diag/seqt_r3.py train --T 128 --hidden 16 --lite 1 --save logs/seqt.pt
    $USERPROFILE/torch-env/Scripts/python -X utf8 research/diag/seq_r3.py export logs/seqt.pt logs/seqt_weights.py
Rows, split and features are those of data/features/mlp2_r3.npz; only the target column (kept under the key y200) differs.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, '.')
SRC = 'data/features/mlp2_r3.npz'
OUT = 'data/features/mlp2tw_r3.npz'
HS = (100, 200, 300, 400, 500)


def build():
    from research.diag import pred_rfit_r3
    from research.signals import LATENCY_US, value_before
    z = dict(np.load(SRC))
    for name in ('round3', 'round3_mirror'):
        ts, cols, rm, mid_ff = pred_rfit_r3.cols_for(name)
        acc = np.zeros(len(ts))
        for H in HS:
            e = value_before(ts, rm, ts + LATENCY_US + H * 1000)
            acc += e - mid_ff
        tw = acc / len(HS)
        tw[ts + LATENCY_US + HS[-1] * 1000 > ts[-1]] = np.nan
        for tag in ('tr', 'ho'):
            idx = z[f'{name}_{tag}_idx']
            y = tw[idx]
            old = z[f'{name}_{tag}_y200']
            bad = ~np.isfinite(y)
            y[bad] = old[bad]
            print(name, tag, 'corr(tw, y200)', round(float(np.corrcoef(y, old)[0, 1]), 4), 'std tw', round(float(y.std()), 3),
                  'std y200', round(float(old.std()), 3), 'filled', int(bad.sum()), flush=True)
            z[f'{name}_{tag}_y200'] = y.astype(np.float32)
    np.savez(OUT, **z)
    print('wrote', OUT)


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build()
    else:
        from research.diag import seq_r3
        ap = argparse.ArgumentParser()
        ap.add_argument('cmd')
        ap.add_argument('--T', type=int, default=128); ap.add_argument('--hidden', type=int, default=16)
        ap.add_argument('--seq', type=int, default=1); ap.add_argument('--epochs', type=int, default=6)
        ap.add_argument('--batch', type=int, default=2048); ap.add_argument('--lr', type=float, default=3e-3)
        ap.add_argument('--seed', type=int, default=0); ap.add_argument('--wd', type=float, default=1e-3)
        ap.add_argument('--out', default='logs/seqt_ho.npy'); ap.add_argument('--save', default='')
        ap.add_argument('--lite', type=int, default=1)
        ap.add_argument('--npz', default=OUT)
        a = ap.parse_args()
        seq_r3.LITE = bool(a.lite)
        seq_r3.NPZ = a.npz
        seq_r3.main(a)
