"""Dump the s12 predictor inputs at every book event, for offline MLP training with the exact in-strategy features.

    uv run python -m research.diag.mlp_dump strategies/honza_r3_s12.py data/round3_public.csv out.npz

Columns of X: the 22 PRED_NAMES features (same order as the MLP in s12), then 4 extras:
fair_sig, log1p(market qty at bb-1), log1p(market qty at ba+1), imbalance of those level-2 quantities.
Also saved: t (us), mid2 (bb+ba), rm2 (robust mid x2, last mid2 with spread <= 3), ridge (raw ridge, ticks).
The strategy trades normally while dumping; features use only market-owned liquidity, so the trading does not leak in.
"""

from __future__ import annotations

import sys
from array import array
from math import exp, log1p
from pathlib import Path

import numpy as np

from research.run import load_strategy_class, simulation_config
from simulator.simulator import ExchangeSimulator

NF = 26


def make_dumper(cls):
    class Dumper(cls):
        def __init__(self) -> None:
            super().__init__()
            self._buf = array('d')
            self._meta = array('d')

        def _pred_update(self, now, bids, asks, bb, ba):
            super()._pred_update(now, bids, asks, bb, ba)
            qb = nb = 0
            qb2 = qa2 = 0
            for v in bids:
                if v.owner != 'market':
                    continue
                if v.price == bb:
                    qb += v.quantity
                    nb += 1
                elif v.price == bb - 1:
                    qb2 += v.quantity
                elif v.price < bb - 1:
                    break
            qa = na = 0
            for v in asks:
                if v.owner != 'market':
                    continue
                if v.price == ba:
                    qa += v.quantity
                    na += 1
                elif v.price == ba + 1:
                    qa2 += v.quantity
                elif v.price > ba + 1:
                    break
            spread = ba - bb
            mid2 = bb + ba
            rm2 = self.pr_rm2 if self.pr_rm2 > 0 else mid2
            pg = self.pf_sig
            f70 = pg[2]
            af = f70 if f70 >= 0.0 else -f70
            fw = self.pf_w[2] * exp(-(now - self.pf_t) / 70000.0) if self.pf_seen else 0.0
            s100 = self.pr_s100
            ltv = self.pr_ltv
            ltp = self.pr_ltp
            ltr_sign = 1.0 if ltv > 0.0 else (-1.0 if ltv < 0.0 else 0.0)
            ltr_lsz = ltr_sign * log1p(ltv if ltv > 0.0 else -ltv)
            sg100 = 1.0 if s100 > 0.0 else (-1.0 if s100 < 0.0 else 0.0)
            if qb + qa > 0:
                imb1 = (qb - qa) / (qb + qa)
                cnt1 = (nb - na) / (nb + na)
            else:
                imb1 = cnt1 = 0.0
            imb2 = (qb2 - qa2) / (qb2 + qa2) if qb2 + qa2 > 0 else 0.0
            self._buf.extend((
                pg[0], pg[1], f70, pg[3], pg[4], f70 * log1p(fw), f70 * af, sg100 * af, f70 * spread, spread,
                imb1, cnt1, log1p(qb), log1p(qa), ltr_sign, ltr_lsz, (ltp - mid2 * 0.5) if ltp > 0 else 0.0,
                (mid2 - rm2) * 0.5, self.pr_ev10, s100, self.pr_s1000,
                log1p((now - self.pr_ttr) / 1000.0) if self.pr_ttr >= 0 else 10.0,
                self.fair_sig, log1p(qb2), log1p(qa2), imb2,
            ))
            self._meta.extend((now, mid2, rm2))

    return Dumper


def main(strategy: str, data: str, out: str) -> None:
    cls = make_dumper(load_strategy_class(strategy))
    st = cls()
    sim = ExchangeSimulator(strategy=st, config=simulation_config())
    res = sim.run(Path(data))
    X = np.frombuffer(st._buf, dtype=np.float64).reshape(-1, NF).astype(np.float32)
    meta = np.frombuffer(st._meta, dtype=np.float64).reshape(-1, 3)
    np.savez(out, X=X, t=meta[:, 0], mid2=meta[:, 1], rm2=meta[:, 2])
    print(f'{data}: pnl {res.pnl:,.0f}, rows {len(X):,} -> {out}')


if __name__ == '__main__':
    main(*sys.argv[1:4])
