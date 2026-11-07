"""Fill attribution for a full-logs run: where does maker PnL come from, which fills lose.

usage: uv run python -X utf8 -m research.fill_attribution logs/<tag>/000_round1 [features.npz] [label]
Fill = group of trade rows sharing (timestamp, taker order, our maker order).
Mid path comes from data/features/*.npz (market-only tape book, forward-filled when a side is empty).
Fresh/back proxy: queue_ahead = market qty resting at our price (features book) just before our
order was applied; 0 = we improved / opened the level (front), large = joined behind a queue.
"""
import sys
import numpy as np
import pandas as pd

H = {"100ms": 100_000, "1s": 1_000_000, "5s": 5_000_000, "30s": 30_000_000}


def load(run, feat):
    f = np.load(feat)
    ts = f["ts_us"]
    bp = f["bid_px"].astype(float)
    ap = f["ask_px"].astype(float)
    ok = (bp > 0) & (ap > 0)
    mid = pd.Series(np.where(ok, (bp + ap) / 2, np.nan)).ffill().bfill().to_numpy()
    spr = pd.Series(np.where(ok, ap - bp, np.nan)).ffill().bfill().to_numpy()
    bq, aq = f["bid_qty"][:, 0].astype(float), f["ask_qty"][:, 0].astype(float)
    imb = (bq - aq) / np.maximum(bq + aq, 1)
    ev = pd.read_csv(f"{run}/events.csv", usecols=["timestamp_us", "source", "operation", "order_id", "side", "price", "quantity"])
    ev = ev[ev.source == "user"]
    tr = pd.read_csv(f"{run}/trades.csv", dtype={"taker_order_id": str, "maker_order_id": str})
    # inventory timeline needs ALL our trades (we are also a taker sometimes), file order
    ut = tr[(tr.maker_owner == "user") | (tr.taker_owner == "user")].copy()
    ut["sg"] = np.where(ut.maker_owner == "user", np.where(ut.taker_side == "A", 1, -1), np.where(ut.taker_side == "B", 1, -1))
    ut["cum"] = (ut.sg * ut.quantity).cumsum()
    tr = ut[ut.maker_owner == "user"]
    return f, ts, mid, spr, imb, ev, tr, ut


def attribute(run, feat):
    f, ts, mid, spr, imb, ev, tr, ut = load(run, feat)
    adds = ev[ev.operation == "add"].set_index("order_id")
    canc = ev[ev.operation == "cancel"].set_index("order_id")["timestamp_us"]
    g = tr.groupby(["timestamp_us", "taker_order_id", "maker_order_id"], sort=True).agg(
        price=("price", "first"), qty=("quantity", "sum"), taker_side=("taker_side", "first"), cum=("cum", "last")).reset_index()
    g = g.sort_values("timestamp_us", kind="stable").reset_index(drop=True)
    g["sign"] = np.where(g.taker_side == "A", 1, -1)  # taker sells -> we buy
    g["taker_total"] = g.taker_order_id.map(tr.groupby("taker_order_id").quantity.sum())
    g["inv_before"] = g.cum - g.sign * g.qty  # includes our own taker trades
    a = adds.loc[g.maker_order_id]
    g["add_ts"] = a.timestamp_us.values
    g["rest_ms"] = (g.timestamp_us - g.add_ts) / 1000
    i0 = np.clip(np.searchsorted(ts, g.timestamp_us.values, side="left") - 1, 0, len(ts) - 1)
    g["spread"] = spr[i0]
    g["imb"] = imb[i0] * g.sign  # + = book leans the way we trade
    g["mid0"] = mid[i0]
    g["edge"] = g.sign * (g.mid0 - g.price)
    for k, h in H.items():
        j = np.clip(np.searchsorted(ts, g.timestamp_us.values + h, side="right") - 1, 0, len(ts) - 1)
        g["mo_" + k] = g.sign * (mid[j] - g.price)
    ia = np.clip(np.searchsorted(ts, a.timestamp_us.values, side="left") - 1, 0, len(ts) - 1)
    side = a.side.values
    px = a.price.values.astype(int)
    bidpx, askpx, bq, aq = f["bid_px"], f["ask_px"], f["bid_qty"], f["ask_qty"]
    qa = np.zeros(len(g))
    for n in range(len(g)):
        i = ia[n]
        if side[n] == "B":
            k = bidpx[i] - px[n]
            qa[n] = bq[i, k] if (0 <= k < 5 and bidpx[i] > 0) else (0 if px[n] > bidpx[i] else 1e6)
        else:
            k = px[n] - askpx[i]
            qa[n] = aq[i, k] if (0 <= k < 5 and askpx[i] > 0) else (0 if (px[n] < askpx[i] or askpx[i] == 0) else 1e6)
    g["queue_ahead"] = qa
    g["side"] = np.where(g.sign > 0, "buy", "sell")
    span = (ts[-1] - ts[0]) / 1e6
    last = g.groupby("maker_order_id").timestamp_us.max()
    fq = g.groupby("maker_order_id").qty.sum()
    act = {}
    for s in ("B", "A"):
        aa = adds[adds.side == s]
        st = aa.timestamp_us.values
        en = []
        for oid, q in zip(aa.index, aa.quantity.values):
            if oid in canc.index:
                en.append(canc[oid])
            elif fq.get(oid, 0) >= q:
                en.append(last[oid])
            else:
                en.append(ts[-1])
        en = np.array(en)
        o = np.argsort(st, kind="stable")
        st, en = st[o], en[o]
        tot, cs, ce = 0, None, None
        for x, y in zip(st, en):
            if ce is None or x > ce:
                if ce is not None:
                    tot += ce - cs
                cs, ce = x, y
            else:
                ce = max(ce, y)
        if ce is not None:
            tot += ce - cs
        act[s] = tot / 1e6 / span
    inv_after = ut.cum.values
    t = ut.timestamp_us.values
    dur = np.diff(np.append(t, ts[-1])) / 1e6
    cap = dur[np.abs(inv_after) >= 100].sum() / span
    return g, act, cap, span


def tab(g, key):
    d = g.assign(usd1=g.qty * g.mo_1s, usdE=g.qty * g.edge, usd30=g.qty * g.mo_30s)
    r = d.groupby(key, observed=True).agg(fills=("qty", "size"), lots=("qty", "sum"), edge_usd=("usdE", "sum"),
                                          mo1s_usd=("usd1", "sum"), mo30s_usd=("usd30", "sum"))
    r["mo1s_per_lot"] = r.mo1s_usd / r.lots
    r["mo30s_per_lot"] = r.mo30s_usd / r.lots
    r = r.round(2).reset_index()
    r.iloc[:, 0] = r.iloc[:, 0].astype(str)
    rows = ["| " + " | ".join(map(str, r.columns)) + " |", "|" + "---|" * len(r.columns)]
    rows += ["| " + " | ".join(str(x) for x in row) + " |" for row in r.itertuples(index=False)]
    return chr(10).join(rows)


def report(run, feat, label):
    g, act, cap, span = attribute(run, feat)
    out = [f"## {label}\n",
           f"fills {len(g)}, lots {g.qty.sum()}, tape span {span:.0f}s. $ columns: edge = qty*sign*(mid_before-price); "
           "mo1s/mo30s = qty*markout vs mid 1s/30s after the fill (1 tick = $1 per lot). Sums are proxies, not the exact score.\n"]
    g["spread_b"] = np.where(g.spread <= 1, "1 tick", "2+ ticks")
    out += ["### (a) by spread at fill\n", tab(g, "spread_b"), ""]
    g["rest_b"] = pd.cut(g.rest_ms, [-1, 50, 500, 5000, 1e12], labels=["<50ms", "50-500ms", "0.5-5s", ">5s"])
    out += ["### (b) by resting time of the order\n", tab(g, "rest_b"), ""]
    g["qa_b"] = pd.cut(g.queue_ahead, [-1, 0, 50, 200, 1e5, 1e9], labels=["0 (front/improver)", "1-50", "51-200", "201+", "far/off-book"])
    out += ["### (b2) fresh vs back proxy: market qty ahead of us when the order was applied\n", tab(g, "qa_b"), ""]
    g["inv_b"] = pd.cut(g.inv_before.abs(), [-1, 49, 89, 1e9], labels=["|I|<50", "50-89", ">=90"])
    out += ["### (c) by inventory before fill\n", tab(g, "inv_b"), "", "by side\n", tab(g, "side"), ""]
    g["inv_dir"] = np.where(g.inv_before == 0, "flat", np.where(np.sign(g.inv_before) * g.sign > 0, "adds to position", "reduces position"))
    out += ["by direction vs inventory\n", tab(g, "inv_dir"), ""]
    g["size_b"] = np.where(g.taker_total >= 100, "taker >=100 lots (sweep)", "taker <100 lots")
    g["fill_b"] = pd.cut(g.qty, [0, 10, 50, 99, 1e9], labels=["1-10", "11-50", "51-99", "100+"])
    out += ["### (d) by our fill size, and by total taker size\n", tab(g, "fill_b"), "", tab(g, "size_b"), ""]
    g["imb_b"] = pd.cut(g.imb, [-1.01, -0.5, -0.1, 0.1, 0.5, 1.01], labels=["<-.5 (book against)", "-.5..-.1", "flat", ".1..5", ">.5 (book with)"])
    out += ["### L1 imbalance before fill (signed: + = book leans the way we trade, i.e. we are on the heavy side)\n", tab(g, "imb_b"), ""]
    m = {k: (g.qty * g["mo_" + k]).sum() / g.qty.sum() for k in H}
    out += ["### (e) lot-weighted mean markout (ticks)\n", "| h | " + " | ".join(H) + " | edge at fill |", "|---|" + "---|" * (len(H) + 1),
            "| markout | " + " | ".join(f"{m[k]:.3f}" for k in H) + f" | {(g.qty * g.edge).sum() / g.qty.sum():.3f} |", ""]
    out += ["### (f) time",
            f"- fraction of tape time with a live bid: {act['B']:.3f}; live ask: {act['A']:.3f}",
            f"- fraction of time at |I| = 100: {cap:.3f}", ""]
    return "\n".join(out), g


def followup(run, feat):
    """2-D imbalance x queue_ahead table, and decomposition of the time a side has no live quote."""
    g, act, cap, span = attribute(run, feat)
    out = ["## Follow-up\n", "### 1. Imbalance at fill x queue_ahead (market qty at our price when the order was applied, 5 ms after send)\n",
           "Cells: lots / markout per lot at 1 s / total $ at 1 s. Imbalance signed (+ = book leans the way we trade).\n"]
    ib = pd.cut(g.imb, [-1.01, -0.5, 0.1, 0.5, 1.01], labels=["<-0.5", "-0.5..0.1", "0.1..0.5", ">0.5"])
    qb = pd.cut(g.queue_ahead, [-1, 0, 50, 200, 1e9], labels=["0", "1-50", "51-200", "201+"])
    d = g.assign(ib=ib, qb=qb, u=g.qty * g.mo_1s)
    rows = ["| imbalance x queue_ahead | " + " | ".join(qb.cat.categories) + " | all |", "|---|" + "---|" * 5]
    for i in ib.cat.categories:
        cells = []
        for q in list(qb.cat.categories) + [None]:
            s = d[(d.ib == i) & ((d.qb == q) if q is not None else True)]
            cells.append(f"{int(s.qty.sum()):,} / {s.u.sum() / max(s.qty.sum(), 1):.2f} / {s.u.sum():,.0f}")
        rows.append(f"| {i} | " + " | ".join(cells) + " |")
    out += rows + [""]

    ev = pd.read_csv(f"{run}/events.csv", usecols=["timestamp_us", "source", "operation", "order_id", "side", "price", "quantity"])
    ev = ev[ev.source == "user"]
    f = np.load(feat)
    t0, t1 = int(f["ts_us"][0]), int(f["ts_us"][-1])
    adds = ev[ev.operation == "add"].set_index("order_id")
    canc = ev[ev.operation == "cancel"].set_index("order_id")["timestamp_us"]
    sent = ev[ev.operation.isin(["add_sent", "cancel_sent"])].copy()
    sside = adds.side.to_dict()
    sent["s"] = np.where(sent.operation == "add_sent", sent.side, sent.order_id.map(sside))
    fq = g.groupby("maker_order_id").qty.sum()
    lastf = g.groupby("maker_order_id").timestamp_us.max()
    ut = load(run, feat)[-1]
    inv_after = ut.cum.values
    ft = ut.timestamp_us.values
    res = {}
    ex = {}
    for s, nm in (("B", "bid"), ("A", "ask")):
        aa = adds[adds.side == s]
        st = aa.timestamp_us.values
        en = np.array([canc[o] if o in canc.index else (lastf[o] if fq.get(o, 0) >= q else t1) for o, q in zip(aa.index, aa.quantity.values)])
        ss = sent[sent.s == s].timestamp_us.values
        # inventory intervals where the side has zero free capacity
        capmask = inv_after >= 100 if s == "B" else inv_after <= -100
        cs = ft[capmask]
        ce = np.append(ft, t1)[1:][capmask]
        sets = {"live": (st, en), "cap": (cs, ce), "infl": (ss, ss + 5000)}
        pts = np.unique(np.concatenate([[t0, t1]] + [np.clip(np.concatenate(v), t0, t1) for v in sets.values()]))
        flags = {}
        for k, (a_, b_) in sets.items():
            d_ = np.zeros(len(pts) + 1)
            np.add.at(d_, np.searchsorted(pts, np.clip(a_, t0, t1)), 1)
            np.add.at(d_, np.searchsorted(pts, np.clip(b_, t0, t1)), -1)
            flags[k] = np.cumsum(d_)[:-1] > 0
        dur = np.diff(pts, append=t1).astype(float)
        live, capf, infl = flags["live"], flags["cap"], flags["infl"]
        tot = dur[(pts >= t0) & (pts < t1)].sum()
        nl = ~live
        A = dur[nl & capf].sum(); B = dur[nl & ~capf & infl].sum(); C = dur[nl & ~capf & ~infl].sum()
        res[nm] = (live_share := dur[live].sum() / tot, A / tot, B / tot, C / tot)
        # examples of (c) gaps > 50 ms
        cgap = nl & ~capf & ~infl
        # merge consecutive segments of the no-live & not-cap state (in-flight allowed inside) -> gaps
        gaps, i, n = [], 0, len(pts)
        nolive_nocap = nl & ~capf
        while i < n:
            if nolive_nocap[i]:
                j = i
                while j + 1 < n and nolive_nocap[j + 1]:
                    j += 1
                gaps.append((pts[i], pts[j + 1] if j + 1 < n else t1))
                i = j + 1
            else:
                i += 1
        big = [x for x in gaps if x[1] - x[0] > 50_000]
        res[nm] += (len(gaps), len(big), np.mean([x[1] - x[0] for x in gaps]) / 1000 if gaps else 0,
                    np.sum([x[1] - x[0] for x in big]) / 1e6 / (tot / 1e6) if big else 0)
        ex[nm] = big
    out += ["### 2. Why a side has no live quote (share of tape time, 12,000 s)\n",
            "| side | live | (a) at cap (free capacity 0) | (b) not capped, send in flight (<5 ms after a send on that side) | (c) other | gaps (no live, not capped) | of which >50 ms | mean gap ms | share of time in >50 ms gaps |", "|---|" + "---|" * 8]
    for nm, r in res.items():
        out.append(f"| {nm} | {r[0]:.3f} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} | {r[4]} | {r[5]} | {r[6]:.1f} | {r[7]:.3f} |")
    out += ["", "(b) counts only time while a send is in flight; (c) is the remainder. Gaps merge consecutive not-capped no-quote stretches.\n"]
    # examples
    rng = np.random.default_rng(1)
    for nm in ("bid", "ask"):
        big = ex[nm]
        if not big:
            continue
        pick = [big[k] for k in sorted(rng.choice(len(big), min(3, len(big)), replace=False))] if len(big) > 3 else big
        for (a_, b_) in pick:
            iv = int(np.searchsorted(ft, a_, side="right")) - 1
            inv = int(inv_after[iv]) if iv >= 0 else 0
            w = ev[(ev.timestamp_us >= a_ - 30_000) & (ev.timestamp_us <= b_ + 10_000)]
            w = w[(w.side == ("B" if nm == "bid" else "A")) | w.side.isna()]
            out.append(f"Example, {nm} side empty {a_ / 1e6:.3f}s to {b_ / 1e6:.3f}s ({(b_ - a_) / 1000:.0f} ms), inventory about {inv}. User events (this side or cancels):\n")
            out.append("```")
            out += [f"{r.timestamp_us:>12} {r.operation:<14} {r.order_id} {r.side if isinstance(r.side, str) else ''} {r.price if r.price == r.price else ''} {r.quantity if r.quantity == r.quantity else ''}" for r in w.head(14).itertuples()]
            out.append("```\n")
    return "\n".join(out)


def trace(run, feat, a_s, b_s):
    """Raw chronological user trace from a_s-0.3 s to b_s+0.05 s (seconds)."""
    f = np.load(feat)
    ts, bp, ap = f["ts_us"], f["bid_px"], f["ask_px"]
    ev = pd.read_csv(f"{run}/events.csv", usecols=["timestamp_us", "source", "operation", "order_id", "side", "price", "quantity"])
    ev = ev[ev.source == "user"]
    tr = pd.read_csv(f"{run}/trades.csv", dtype={"taker_order_id": str, "maker_order_id": str})
    tr = tr[(tr.maker_owner == "user") | (tr.taker_owner == "user")].copy()
    tr["sg"] = np.where(tr.maker_owner == "user", np.where(tr.taker_side == "A", 1, -1), np.where(tr.taker_side == "B", 1, -1))
    tr["inv"] = (tr.sg * tr.quantity).cumsum()
    lo, hi = int((a_s - 0.3) * 1e6), int((b_s + 0.05) * 1e6)
    rows = []
    for r in ev[(ev.timestamp_us >= lo) & (ev.timestamp_us <= hi)].itertuples():
        rows.append((r.timestamp_us, 0, f"{r.operation:<13} {r.order_id} {r.side if isinstance(r.side, str) else '-'} px={r.price if r.price == r.price else '-'} qty={r.quantity if r.quantity == r.quantity else '-'}"))
    for r in tr[(tr.timestamp_us >= lo) & (tr.timestamp_us <= hi)].itertuples():
        rows.append((r.timestamp_us, 1, f"FILL{'(taker)' if r.taker_owner == 'user' else '        '} {r.maker_order_id if r.taker_owner != 'user' else r.taker_order_id} {'B' if r.sg > 0 else 'A'} px={r.price} qty={r.quantity} taker={r.taker_order_id}  inventory_after={r.inv}"))
    rows.sort()
    out = [f"Trace {lo / 1e6:.3f}-{hi / 1e6:.3f} s (gap {a_s}-{b_s} s). market bid/ask = tape book after the last tape row at or before the time.", "```"]
    for t, _, txt in rows:
        k = max(int(np.searchsorted(ts, t, side="right")) - 1, 0)
        out.append(f"{t / 1e6:12.6f} bid={bp[k]} ask={ap[k]}  {txt}")
    ga, gb = int(a_s * 1e6), int(b_s * 1e6)
    n = int(np.searchsorted(ts, gb, side="right") - np.searchsorted(ts, ga, side="left"))
    out += [f"tape rows inside the gap: {n}", "```", ""]
    return chr(10).join(out)


if __name__ == "__main__":
    run = sys.argv[1]
    feat = sys.argv[2] if len(sys.argv) > 2 else "data/features/round1.npz"
    label = sys.argv[3] if len(sys.argv) > 3 else run
    if label == "trace":
        print(trace(run, feat, float(sys.argv[4]), float(sys.argv[5])))
        sys.exit()
    print(followup(run, feat) if label == "followup" else report(run, feat, label)[0])


