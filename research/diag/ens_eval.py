import pickle, numpy as np, sys
sys.path.insert(0, '.')
from research.diag.mlp_fit import load, split, forward
M = '../mlpdata/'
for nm in ('round4_public', 'round4_mirror'):
    X, y = load(M + nm + '.npz', 200, 26); (_, _), (Xh, yh) = split(X, y, 0.75)
    ps = [forward(Xh, *pickle.loads(open(M + f'ens_s{s}.pkl', 'rb').read())) for s in range(4)]
    single = [np.corrcoef(p, yh)[0, 1] for p in ps]
    print(nm, 'single', ' '.join(f'{c:.4f}' for c in single), '| avg2', f'{np.corrcoef(np.mean(ps[:2],0), yh)[0,1]:.4f}', '| avg4', f'{np.corrcoef(np.mean(ps,0), yh)[0,1]:.4f}')
