"""Fill-EV dataset and model (H20): predict the 1 s markout of OUR fill from state a decision can see.

usage:
  uv run python -X utf8 -m research.fill_ev build <run_dir> <features.npz> <out.npz>
  uv run python -X utf8 -m research.fill_ev eval  <r1.npz> <v2.npz> <report_fragment.md>
  uv run python -X utf8 -m research.fill_ev fit   <r1.npz> <v2.npz>      # prints the literal model for the strategy

One row per fill (rows of research.fill_attribution.attribute: timestamp, taker order, our order), lot-weighted.
Features are read at the last tape event BEFORE the fill row (the state a quote decision could have seen,
5 ms latency aside). ahead = market lots matched ahead of us in the same taker order at our price, i.e. the
queue ahead in the snapshot just before the sweep.
"""
import sys
import numpy as np
import pandas as pd

from research.fill_attribution import attribute
from research.signals import build_matrix

FEATS = ["pred_s", "spread", "spread2", "imb_s", "ahead_log", "front", "thick_log", "behind_log", "off",
         "absI", "reduce", "invdir", "flow100_s", "flow1000_s", "ret100_s", "dt_mid_log", "age_log"]
PRIVATE = ("ahead_log", "front", "thick_log", "behind_log", "off", "absI", "reduce", "invdir", "age_log")


def build(run, feat, out):
    g, act, cap, span = attribute(run, feat)
    f = np.load(feat)
    ts = f["ts_us"]
    n = len(ts)
    i0 = np.clip(np.searchsorted(ts, g.timestamp_us.values, side="left") - 1, 0, n - 1)
    X, names = build_matrix({k: f[k] for k in f.files if k not in ("source", "public_depth", "build_seconds")}, rows=i0)
    col = {k: X[:, j].astype(np.float64) for j, k in enumerate(names)}
    from strategies.mm_ml import MarketMaker
    m = MarketMaker.LINEAR_MODEL
    pred = np.full(len(g), m["intercept"])
    for nm, c in m["coef"].items():
        v = np.clip(col[nm], m["lo"][nm], m["hi"][nm])
        pred += c * (v - m["mean"][nm]) / m["scale"][nm]
    sg = g.sign.values.astype(float)
    px = g.price.values.astype(int)
    bidpx, askpx = f["bid_px"][i0].astype(int), f["ask_px"][i0].astype(int)
    off = np.where(sg > 0, bidpx - px, px - askpx)  # 0 = at the same-side touch, >0 = behind it
    kk = np.clip(off, 0, 4)
    bq, aq = f["bid_qty"][i0], f["ask_qty"][i0]
    ar = np.arange(len(g))
    thick = np.where(sg > 0, bq[ar, kk], aq[ar, kk]).astype(float)
    thick = np.where((off < 0) | (off > 4), 0, thick)
    t = pd.read_csv(f"{run}/trades.csv", dtype={"taker_order_id": str, "maker_order_id": str})
    t["mkq"] = np.where(t.maker_owner == "market", t.quantity, 0)
    t["ahead"] = t.groupby(["taker_order_id", "price"]).mkq.cumsum() - t.mkq
    ah = t[t.maker_owner == "user"].groupby(["timestamp_us", "taker_order_id", "maker_order_id"]).ahead.min()
    ahead = pd.MultiIndex.from_frame(g[["timestamp_us", "taker_order_id", "maker_order_id"]]).map(ah.to_dict().get).to_numpy(dtype=float)
    ahead = np.nan_to_num(ahead, nan=0.0)
    behind = np.maximum(thick - ahead, 0)
    inv = g.inv_before.values.astype(float)
    d = {
        "chunk": (i0 * 8 // n).astype(np.int8), "ts": g.timestamp_us.values, "qty": g.qty.values.astype(float), "sign": sg,
        "mo100": g.mo_100ms.values, "mo1s": g.mo_1s.values, "mo5s": g.mo_5s.values, "mo30s": g.mo_30s.values,
        "pred_s": pred * sg, "spread": col["spread"], "spread2": (col["spread"] >= 2).astype(float),
        "imb_s": col["imb1"] * sg, "ahead_log": np.log1p(ahead), "front": (ahead == 0).astype(float),
        "thick_log": np.log1p(thick), "behind_log": np.log1p(behind), "off": np.clip(off, -1, 4).astype(float),
        "absI": np.abs(inv) / 100.0, "reduce": ((inv * sg) < 0).astype(float), "invdir": np.clip(inv * sg / 100.0, -1, 1),
        "flow100_s": np.clip(col["flow_ms100"] * sg, -300, 300), "flow1000_s": np.clip(col["flow_ms1000"] * sg, -1000, 1000),
        "ret100_s": np.clip(col["ret_ms100"] * sg, -5, 5), "dt_mid_log": col["dt_mid_log"],
        "age_log": np.log1p(np.maximum(g.rest_ms.values, 0)),
    }
    ok = np.isfinite(np.column_stack([d[k] for k in FEATS + ["mo1s"]])).all(1)
    d = {k: v[ok] for k, v in d.items()}
    np.savez(out, **d)
    print(f"{run}: {ok.sum()} fills (of {len(g)}), lots {d['qty'].sum():.0f}, mo1s/lot {(d['qty']*d['mo1s']).sum()/d['qty'].sum():.3f}")


def wcorr(a, b, w):
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    c = np.average((a - ma) * (b - mb), weights=w)
    return c / np.sqrt(np.average((a - ma) ** 2, weights=w) * np.average((b - mb) ** 2, weights=w) + 1e-18)


def spearman(a, b):
    return np.corrcoef(pd.Series(a).rank().values, pd.Series(b).rank().values)[0, 1]


class Ridge:
    def __init__(self, alpha=10.0):
        self.alpha = alpha

    def fit(self, X, y, w):
        self.lo, self.hi = np.quantile(X, 0.002, axis=0), np.quantile(X, 0.998, axis=0)
        Xc = np.clip(X, self.lo, self.hi)
        self.mean = np.average(Xc, axis=0, weights=w)
        self.scale = np.sqrt(np.average((Xc - self.mean) ** 2, axis=0, weights=w)) + 1e-9
        Z = (Xc - self.mean) / self.scale
        self.ym = np.average(y, weights=w)
        sw = np.sqrt(w / w.mean())
        Zs = Z * sw[:, None]
        self.coef = np.linalg.solve(Zs.T @ Zs + self.alpha * np.eye(Z.shape[1]), Zs.T @ ((y - self.ym) * sw))
        return self

    def predict(self, X):
        return ((np.clip(X, self.lo, self.hi) - self.mean) / self.scale) @ self.coef + self.ym


def load(p):
    d = np.load(p)
    return {k: d[k] for k in d.files}


def walk(d, feats, label, kind):
    """Walk-forward over 8 chunks: train on chunks < k (fills whose label window reaches chunk k dropped)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = np.column_stack([d[k] for k in feats])
    y, w, ch = d[label], d["qty"], d["chunk"]
    oof = np.full(len(y), np.nan)
    for k in range(1, 8):
        te = ch == k
        t0 = d["ts"][te].min()
        tr = (ch < k) & (d["ts"] + 6_000_000 < t0)
        if kind == "ridge":
            m = Ridge(10.0).fit(X[tr], y[tr], w[tr])
        else:
            m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                                              l2_regularization=1.0, random_state=0)
            m.fit(X[tr], y[tr], sample_weight=w[tr])
        oof[te] = m.predict(X[te])
    return oof


def stats(d, oof, label="mo1s"):
    ok = np.isfinite(oof)
    y, w, p = d[label][ok], d["qty"][ok], oof[ok]
    base = np.average(y, weights=w)
    r2 = 1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - base) ** 2)
    per = [spearman(p[d["chunk"][ok] == k], y[d["chunk"][ok] == k]) for k in range(1, 8)]
    return dict(r2=r2, ic_w=wcorr(p, y, w), ic_rank=spearman(p, y), ic_pos=sum(x > 0 for x in per), ic_min=min(per))


def deciles(d, ev, label="mo1s", q=10):
    ok = np.isfinite(ev)
    y, w, p = d[label][ok], d["qty"][ok], ev[ok]
    o = np.argsort(p)
    b = np.minimum((np.cumsum(w[o]) / w.sum() * q).astype(int), q - 1)
    rows = []
    for i in range(q):
        s = o[b == i]
        rows.append((i + 1, p[s].mean(), (w[s] * y[s]).sum() / w[s].sum(), w[s].sum(), (w[s] * y[s]).sum()))
    return rows


def md_dec(title, rows):
    out = [f"**{title}**\n", "| decile | mean EV | realised mo/lot | lots | $ at 1 s |", "|---|---|---|---|---|"]
    out += [f"| {a} | {b:.3f} | {c:.3f} | {dd:,.0f} | {e:,.0f} |" for a, b, c, dd, e in rows]
    return "\n".join(out) + "\n"


def evaluate(p1, p2, frag):
    d1, d2 = load(p1), load(p2)
    nopriv = [f for f in FEATS if f not in PRIVATE]
    out = ["| dataset | fills | lots | model | features | R2 (1 s, OOF) | IC lot-w | rank IC | chunks IC>0 (of 7) | min chunk IC |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for nm, d in (("round1", d1), ("sample_v2", d2)):
        for lab, fs, kind in (("pred only", ["pred_s"], "ridge"), ("market feats (no private)", nopriv, "ridge"),
                              ("full", FEATS, "ridge"), ("full", FEATS, "hgb")):
            oof = walk(d, fs, "mo1s", kind)
            s = stats(d, oof)
            res[(nm, lab, kind)] = oof
            out.append(f"| {nm} | {len(d['qty'])} | {d['qty'].sum():,.0f} | {kind} | {lab} | {s['r2']:.4f} | {s['ic_w']:.3f} | {s['ic_rank']:.3f} | {s['ic_pos']} | {s['ic_min']:.3f} |")
    out.append("")
    for nm, d in (("round1", d1), ("sample_v2", d2)):
        out.append(md_dec(f"{nm}: deciles by out-of-fold full-ridge EV (chunks 1-7)", deciles(d, res[(nm, 'full', 'ridge')])))
        out.append(md_dec(f"{nm}: deciles by pred_s alone (OOF ridge on pred)", deciles(d, res[(nm, 'pred only', 'ridge')])))
        out.append(md_dec(f"{nm}: deciles by out-of-fold HGB EV", deciles(d, res[(nm, 'full', 'hgb')])))
    out += ["**Cross-fit** (fit on all fills of one dataset, score the other; ridge, full features)\n",
            "| train -> test | R2 | IC lot-w | rank IC |", "|---|---|---|---|"]
    for a, b, nb in ((d1, d2, "round1 -> sample_v2"), (d2, d1, "sample_v2 -> round1")):
        m = Ridge(10.0).fit(np.column_stack([a[k] for k in FEATS]), a["mo1s"], a["qty"])
        p = m.predict(np.column_stack([b[k] for k in FEATS]))
        y, w = b["mo1s"], b["qty"]
        r2 = 1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2)
        out.append(f"| {nb} | {r2:.4f} | {wcorr(p, y, w):.3f} | {spearman(p, y):.3f} |")
        if nb.startswith("round1"):
            xs = md_dec("sample_v2 fills by the round1 model's EV (cross-fit deciles)", deciles(b, p))
    out.append("")
    out.append(xs)
    open(frag, "w", encoding="utf-8").write("\n".join(out))
    print("\n".join(out))


def fit(p1, p2, feats=None):
    """Fit the ridge on the pooled? no: on round1 only, print the literal for the strategy."""
    d = load(p1)
    feats = feats or [f for f in FEATS if f != "age_log"]
    m = Ridge(10.0).fit(np.column_stack([d[k] for k in feats]), d["mo1s"], d["qty"])
    ev = m.predict(np.column_stack([d[k] for k in feats]))
    qs = np.quantile(np.repeat(ev, 1), [0.1, 0.2, 0.3])
    lots_q = []
    o = np.argsort(ev)
    cw = np.cumsum(d["qty"][o]) / d["qty"].sum()
    for q in (0.1, 0.2, 0.3):
        lots_q.append(float(ev[o][np.searchsorted(cw, q)]))
    print("EV_MODEL = " + repr({"intercept": float(m.ym), "feats": feats, "lo": m.lo.round(6).tolist(), "hi": m.hi.round(6).tolist(),
                                "mean": m.mean.round(6).tolist(), "scale": m.scale.round(6).tolist(), "coef": m.coef.round(6).tolist()}))
    print("lot-weighted EV quantiles 10/20/30 %:", [round(x, 4) for x in lots_q], " fill-count quantiles:", qs.round(4))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "fit":
        fit(sys.argv[2], sys.argv[3])
    else:
        evaluate(sys.argv[2], sys.argv[3], sys.argv[4])
