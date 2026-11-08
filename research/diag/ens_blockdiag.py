"""Average K mlp_fit pickles (same input scaler) into ONE block-diagonal MLP pickle: hidden layers concatenated,
inner weights block-diagonal, output weights / K. Exactly the mean of the K nets.
usage: python research/diag/ens_blockdiag.py OUT.pkl IN1.pkl IN2.pkl ..."""
import pickle, sys
import numpy as np
out, ins = sys.argv[1], sys.argv[2:]
nets = [pickle.loads(open(p, 'rb').read()) for p in ins]
K = len(nets); mean, scale = np.asarray(nets[0][0]), np.asarray(nets[0][1])
for m, s, _ in nets:
    assert np.allclose(m, mean, atol=1e-4) and np.allclose(s, scale, atol=1e-4), 'scalers differ'
L = []
for li in range(len(nets[0][2])):
    Ws = [np.asarray(n[2][li][0]) for n in nets]; bs = [np.asarray(n[2][li][1]) for n in nets]
    if li == 0:
        W = np.hstack(Ws); b = np.concatenate(bs)
    elif li == len(nets[0][2]) - 1:
        W = np.vstack(Ws) / K; b = sum(bs) / K
    else:
        r = sum(w.shape[0] for w in Ws); c = sum(w.shape[1] for w in Ws); W = np.zeros((r, c)); i = j = 0
        for w in Ws:
            W[i:i + w.shape[0], j:j + w.shape[1]] = w; i += w.shape[0]; j += w.shape[1]
        b = np.concatenate(bs)
    L.append((W, b))
open(out, 'wb').write(pickle.dumps((mean, scale, L)))
print('wrote', out, [w.shape for w, _ in L])
