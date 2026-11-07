"""When does passive liquidity provision make money on this tape? Touch-maker markouts and filters.

    uv run python -m research.mm_edge data/round1.csv                      # -> research/reports/round1_mm_edge.md
    uv run python -m research.mm_edge data/sample.csv --run-dir logs/mmedge-check/000_sample   # + strategy check

Every tape row whose trades hit the touch (a market taker) is a fill of a hypothetical maker
resting at that touch. The maker's side is s = +1 (bid, bought) when the taker sold, -1 when it
bought; its fill price p0 is the touch just before the row. With mid(T) = mid after all rows with
ts <= T (features `mid_ticks(ffill=True)`, market-only book):

    markout(h)   = s * (mid(t + h) - p0)           net edge per lot in ticks (= realized half-spread)
    effective    = s * (mid_before - p0)           half-spread captured vs the pre-fill mid
    impact(h)    = -s * (mid(t + h) - mid_before)  adverse move (taker-signed mid change)
    effective    = realized(h) + impact(h)

Conditions are taken from the book a strategy could still act on: the last row with
ts <= t - lag (default lag = 5 ms order latency, `--lag-us 0` = the row right before the fill),
never later than the row before the fill. "Imbalance" is the L1 queue imbalance from the maker
side's own view, s * (qb - qa) / (qb + qa) (> 0 = the maker's own queue is the heavier one).
"Flow toward maker" is the signed traded volume in the window, signed so that > 0 means recent
takers hit the same side as this fill. Taker size and sweep/depletion are known only after the
fill (descriptive, not filters). Front-of-queue maker = every fill counts; a back-of-queue joiner
is filled only when the taker depletes the level, so its edge is the "depleted" bucket.

t-stats treat fills as independent; fills cluster in bursts, so read them as optimistic.
Stability: the tape is split into equal-event chunks (8 by default); "k/8" = chunks where the
bucket's mean edge has the same sign as its overall mean (chunks with < 5 fills skipped).

With `--run-dir` (a `research.run --full-logs` run directory), the strategy's own maker fills from
trades.csv are marked out against the same mid series, as a sanity check of the touch-maker numbers.
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from research.features import EV_LIMIT, ensure_features, load_features, mid_ticks
from research.signals import LATENCY_US, REPORTS_DIR

HORIZONS_MS = (100, 1000, 5000, 30000)
QUEUE_AHEAD = (0, 5, 20, 50, 100, 200, 500, 1000)
MIN_CHUNK_FILLS = 5
MIN_BUCKET_SHARE = 0.01
MIN_RULE_SHARE = 0.05


# ---------------------------------------------------------------- formatting


def _table(header: list[str], rows: list[list[object]]) -> list[str]:
    out = ['| ' + ' | '.join(header) + ' |', '|---|' + '---:|' * (len(header) - 1)]
    out += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return out


def _f(x: float, spec: str = '+.3f') -> str:
    return '-' if x != x else format(x, spec)


def _pct(x: float) -> str:
    return '-' if x != x else f'{100 * x:.1f}%'


def _tstat(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float('nan')
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / math.sqrt(len(x)))) if sd > 0 else float('nan')


def _mean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float('nan')


# ---------------------------------------------------------------- fills and features


def mid_at_or_before(ts: np.ndarray, mid: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """mid after the last row with ts <= t_query; NaN past the end of the tape."""
    j = np.searchsorted(ts, t_query, side='right') - 1
    out = mid[np.maximum(j, 0)].copy()
    out[(j < 0) | (t_query > ts[-1])] = np.nan
    return out


def touch_fills(d: dict[str, np.ndarray], lag_us: int, n_chunks: int) -> dict[str, np.ndarray]:
    """One record per taker row: maker side, fill price, markouts and the pre-fill conditions."""
    ts = d['ts_us']
    n = len(ts)
    bid, ask = d['bid_px'].astype(np.int64), d['ask_px'].astype(np.int64)
    bq, aq = d['bid_qty'][:, 0].astype(np.int64), d['ask_qty'][:, 0].astype(np.int64)
    bc, ac = d['bid_cnt1'].astype(np.int64), d['ask_cnt1'].astype(np.int64)
    tv = d['trade_vol'].astype(np.int64)
    mid = mid_ticks(d, ffill=True)

    i = np.flatnonzero(tv != 0)
    i = i[i >= 1]
    p = i - 1
    i = i[(bid[p] > 0) & (ask[p] > 0)]
    p = i - 1
    taker = np.sign(tv[i])  # +1 = buyer-initiated (lifted the ask)
    s = -taker  # maker side: +1 bid (bought), -1 ask (sold)
    is_bid = s > 0
    p0 = np.where(is_bid, bid[p], ask[p])
    t = ts[i]
    mid0 = (bid[p] + ask[p]) / 2.0

    f: dict[str, np.ndarray] = {'row': i, 't': t, 's': s, 'p0': p0, 'qty': np.abs(tv[i])}
    f['eff_half'] = s * (mid0 - p0)
    for h in HORIZONS_MS:
        m = mid_at_or_before(ts, mid, t + h * 1000)
        f[f'mk{h}'] = s * (m - p0)
        f[f'imp{h}'] = -s * (m - mid0)

    # Post-fill (descriptive): did the taker eat the whole touch?
    own_after = np.where(is_bid, bid[i], ask[i])
    touch_q = np.where(is_bid, bq[p], aq[p])
    depleted = np.where(is_bid, (own_after < p0) | (own_after == 0), own_after > p0)
    beyond = f['qty'] > touch_q
    f['sweep'] = np.where(beyond, 2, np.where(depleted, 1, 0))
    f['touch_cnt_before'] = np.where(is_bid, bc[p], ac[p])
    f['touch_q'] = touch_q

    # Pre-fill state: last row with ts <= t - lag, and never later than the row before the fill.
    t_s = t - lag_us
    j = np.minimum(np.searchsorted(ts, t_s, side='right') - 1, p)
    ok = j >= 0
    j = np.maximum(j, 0)
    t_state = np.where(lag_us > 0, t_s, ts[j])
    spread = (ask[j] - bid[j]).astype(np.float64)
    spread[(bid[j] <= 0) | (ask[j] <= 0) | ~ok] = np.nan
    f['spread'] = spread
    qb, qa = bq[j].astype(np.float64), aq[j].astype(np.float64)
    tot = qb + qa
    with np.errstate(invalid='ignore', divide='ignore'):
        f['imb'] = np.where(tot > 0, s * (qb - qa) / tot, np.nan)
    f['own_q'] = np.where(is_bid, qb, qa)
    f['own_q'][~ok] = np.nan
    c = np.zeros(n + 1)
    np.cumsum(tv, out=c[1:])
    cnt = np.arange(n + 1, dtype=np.float64)
    for w_ms in (100, 1000):
        first = np.minimum(np.searchsorted(ts, t_state - w_ms * 1000, side='right'), j + 1)
        flow = c[j + 1] - c[first]
        f[f'flow{w_ms}'] = np.where(ok, taker * flow, np.nan)
        if w_ms == 1000:
            f['intensity'] = np.where(ok, cnt[j + 1] - cnt[first], np.nan)
    f['chunk'] = np.minimum(i * n_chunks // n, n_chunks - 1)
    return f


# ---------------------------------------------------------------- bucketing


def quantile_buckets(x: np.ndarray, k: int = 5, digits: int = 2) -> tuple[np.ndarray, list[str]]:
    """Codes 0..m-1 by (deduplicated) quantile edges of x; -1 for NaN."""
    fin = np.isfinite(x)
    if not fin.any():
        return np.full(len(x), -1), []
    edges = np.unique(np.round(np.quantile(x[fin], np.linspace(0, 1, k + 1)[1:-1]), digits))
    codes = np.searchsorted(edges, x, side='right')
    codes[~fin] = -1
    labels = []
    for b in range(len(edges) + 1):
        if b == 0:
            labels.append(f'< {edges[0]:g}' if len(edges) else 'all')
        elif b == len(edges):
            labels.append(f'>= {edges[-1]:g}')
        else:
            labels.append(f'[{edges[b - 1]:g}, {edges[b]:g})')
    return codes, labels


def sign_buckets(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    codes = np.where(x < 0, 0, np.where(x > 0, 2, 1))
    codes[~np.isfinite(x)] = -1
    return codes, ['against (< 0)', 'none (0)', 'toward maker (> 0)']


def conditions(f: dict[str, np.ndarray]) -> list[tuple[str, str, np.ndarray, list[str], bool]]:
    """(key, title, codes, labels, pre_fill) for every conditioning variable; codes are ordered."""
    out = []
    spread = f['spread']
    sp_codes = np.where(np.isfinite(spread), np.clip(np.nan_to_num(spread), 1, 5) - 1, -1).astype(np.int64)
    out.append(('spread', 'spread (ticks)', sp_codes, ['1', '2', '3', '4', '>= 5'], True))
    out.append(('imb', 'L1 imbalance, maker view', *quantile_buckets(f['imb']), True))
    out.append(('own_q', 'own touch queue (lots)', *quantile_buckets(f['own_q'], digits=0), True))
    out.append(('flow100', 'flow toward maker, 100 ms', *sign_buckets(f['flow100']), True))
    out.append(('flow1000', 'flow toward maker, 1 s', *sign_buckets(f['flow1000']), True))
    out.append(('intensity', 'events in last 1 s', *quantile_buckets(f['intensity'], digits=0), True))
    out.append(('side', 'maker side', np.where(f['s'] > 0, 0, 1), ['bid (maker buys)', 'ask (maker sells)'], False))
    out.append(('qty', 'taker size (lots, post-fill)', *quantile_buckets(f['qty'].astype(np.float64), digits=0), False))
    out.append(('sweep', 'taker vs touch (post-fill)', f['sweep'], ['partial (touch remains)', 'depleted exactly', 'swept beyond'], False))
    return out


def chunk_stability(edge: np.ndarray, chunk: np.ndarray, n_chunks: int, sign: float) -> tuple[int, int]:
    """(chunks whose mean edge has the same sign as `sign`, chunks with enough fills)."""
    fin = np.isfinite(edge)
    e, ch = edge[fin], chunk[fin]
    cnt = np.bincount(ch, minlength=n_chunks)
    tot = np.bincount(ch, weights=e, minlength=n_chunks)
    valid = cnt >= MIN_CHUNK_FILLS
    means = np.where(valid, tot / np.maximum(cnt, 1), 0)
    return int(((np.sign(means) == np.sign(sign)) & valid & (means != 0)).sum()), int(valid.sum())


# ---------------------------------------------------------------- sections


def decomposition_section(f: dict[str, np.ndarray]) -> tuple[list[str], dict[str, float]]:
    facts: dict[str, float] = {}
    rows = []
    groups = [('front', 'front of queue: every touch fill', np.ones(len(f['s']), bool)),
              ('back', 'back of queue: touch depleted', f['sweep'] >= 1)]
    for key, name, m in groups:
        for h in HORIZONS_MS:
            mk = f[f'mk{h}'][m]
            imp = f[f'imp{h}'][m]
            eff = f['eff_half'][m]
            fin = np.isfinite(mk)
            w = f['qty'][m][fin].astype(np.float64)
            vw = float(np.average(mk[fin], weights=w)) if fin.any() else float('nan')
            rows.append([name if h == HORIZONS_MS[0] else '', f'{h} ms', f'{int(fin.sum()):,}',
                         _f(_mean(eff[fin])), _f(_mean(mk)), _f(_mean(imp)), _f(_tstat(mk), '+.1f'), _f(vw),
                         _pct(float((mk[fin] > 0).mean()) if fin.any() else float('nan'))])
            facts[f'{key}_{h}'] = _mean(mk)
    lines = [
        '## Touch-maker markout: effective = realized + impact',
        '',
        'Per fill event, in ticks. effective = half-spread captured vs the pre-fill mid; realized = net edge per lot '
        '(markout); impact = adverse mid move (taker-signed). vw = weighted by taker volume.',
        '',
        *_table(['maker', 'h', 'n', 'effective half', 'realized (edge)', 'impact', 't', 'edge vw', 'edge > 0'], rows),
        '',
    ]
    # Queue position: a maker with Q lots ahead at the touch is filled only by takers trading more than Q there.
    rows = []
    for q_ahead in QUEUE_AHEAD:
        m = (f['touch_q'] >= q_ahead) & (f['qty'] > q_ahead)
        if m.sum() < 10:
            continue
        facts[f'queue_{q_ahead}'] = _mean(f['mk1000'][m])
        rows.append([f'{q_ahead}', f'{int(m.sum()):,}', _pct(float(m.mean())),
                     *[_f(_mean(f[f'mk{h}'][m])) for h in HORIZONS_MS], _f(_tstat(f[f'mk1000'][m]), '+.1f')])
    lines += ['Queue position (cancels ahead ignored): a maker with Q market lots ahead at the touch is filled only when '
              'the taker trades more than Q there.', '',
              *_table(['lots ahead Q', 'fills', 'of all', *[f'edge {h} ms' for h in HORIZONS_MS], 't 1000 ms'], rows), '']
    return lines, facts


def conditional_section(
    f: dict[str, np.ndarray], conds: list, horizon_ms: int, n_chunks: int
) -> tuple[list[str], list[tuple]]:
    edge = f[f'mk{horizon_ms}']
    n_all = len(edge)
    lines = [
        f'## Net edge conditioned on what is visible before the fill (h = {horizon_ms} ms)',
        '',
        f'Edge columns = mean markout in ticks at each horizon; t and stability at {horizon_ms} ms. '
        'Rows marked (post) are not usable as filters.',
        '',
    ]
    top: list[tuple] = []
    header = ['bucket', 'n', 'share', *[f'edge {h} ms' for h in HORIZONS_MS], 't', f'stable /{n_chunks}']
    for key, title, codes, labels, pre in conds:
        rows = []
        for b, label in enumerate(labels):
            m = codes == b
            nb = int(m.sum())
            if nb == 0:
                continue
            e = edge[m]
            mean = _mean(e)
            t = _tstat(e)
            same, valid = chunk_stability(e, f['chunk'][m], n_chunks, mean) if mean == mean else (0, 0)
            rows.append([label, f'{nb:,}', _pct(nb / n_all), *[_f(_mean(f[f'mk{h}'][m])) for h in HORIZONS_MS],
                         _f(t, '+.1f'), f'{same}/{valid}'])
            if nb >= MIN_BUCKET_SHARE * n_all and t == t:
                top.append((abs(t), title + ('' if pre or key == 'side' else ' (post)'), label, nb, mean, t, same, valid))
        lines += [f'### {title}{"" if pre or key == "side" else " (post)"}', '', *_table(header, rows), '']
    top.sort(key=lambda r: -r[0])
    lines += ['### Strongest buckets (|t|)', '',
              *_table(['condition', 'bucket', 'n', 'edge', 't', 'same sign'],
                      [[c, lab, f'{nb:,}', _f(mean), _f(t, '+.1f'), f'{s}/{v}'] for _, c, lab, nb, mean, t, s, v in top[:10]]),
              '']
    return lines, top


def _rule_desc(labels: list[str], b: int, kind: str) -> str:
    """Readable threshold for 'bucket >= b' (kind 'ge') / 'bucket <= b' (kind 'le') from the bucket labels."""
    lab = labels[b]
    if labels[0].startswith('against'):  # sign buckets: against (< 0), none (0), toward maker (> 0)
        return {('ge', 1): '>= 0', ('ge', 2): '> 0', ('le', 0): '< 0', ('le', 1): '<= 0'}[(kind, b)]
    if lab.isdigit() or lab.startswith('>= ') and labels[0].isdigit():  # spread ticks: '1', '2', ..., '>= 5'
        return f'>= {b + 1}' if kind == 'ge' else f'<= {b + 1}'
    if kind == 'ge':  # lower edge of bucket b: '[a, c)' or '>= a'
        return '>= ' + (lab[1:].split(',')[0] if lab.startswith('[') else lab[3:])
    return '< ' + (lab.split(', ')[1].rstrip(')') if lab.startswith('[') else lab[2:])  # upper edge: '[a, c)' or '< c'


def best_threshold_rule(
    codes: np.ndarray, labels: list[str], edge: np.ndarray, keep: np.ndarray
) -> tuple[float, np.ndarray, str] | None:
    """Best 'bucket >= b' / 'bucket <= b' rule by total edge over the fills it keeps (within `keep`)."""
    fin = np.isfinite(edge) & keep
    base_n = int(fin.sum())
    best = None
    m_b = len(labels)
    for b in range(m_b):
        for kind in ('ge', 'le'):
            if (kind == 'ge' and b == 0) or (kind == 'le' and b == m_b - 1):
                continue
            sel = fin & ((codes >= b) if kind == 'ge' else ((codes <= b) & (codes >= 0)))
            if sel.sum() < MIN_RULE_SHARE * max(base_n, 1):
                continue
            total = float(edge[sel].sum())
            if best is None or total > best[0]:
                desc = _rule_desc(labels, b, kind)
                best = (total, sel, desc)
    return best


def rules_section(
    f: dict[str, np.ndarray], conds: list, horizon_ms: int, n_chunks: int
) -> tuple[list[str], dict[str, object]]:
    edge = f[f'mk{horizon_ms}']
    fin = np.isfinite(edge)
    base_total = float(edge[fin].sum())
    base_mean = _mean(edge)
    base_same, base_valid = chunk_stability(edge, f['chunk'], n_chunks, 1.0)
    pre = [c for c in conds if c[4]]
    found = []
    for key, title, codes, labels, _ in pre:
        r = best_threshold_rule(codes, labels, edge, fin)
        if r is None:
            continue
        total, sel, desc = r
        e = edge[sel]
        pos, valid = chunk_stability(e, f['chunk'][sel], n_chunks, 1.0)
        found.append((total, key, title, desc, sel, _mean(e), _tstat(e), float(sel.sum() / fin.sum()), pos, valid))
    found.sort(key=lambda r: -r[0])
    floor_total = max(base_total, 0.0)  # a rule must keep a positive total and beat quoting always
    useful = [r for r in found if r[0] > floor_total]

    def rule_row(name: str, sel: np.ndarray) -> list[object]:
        e = edge[sel & fin]
        pos, valid = chunk_stability(e, f['chunk'][sel & fin], n_chunks, 1.0)
        return [name, _pct(float((sel & fin).sum() / fin.sum())), _f(_mean(e)), _f(_tstat(e), '+.1f'),
                _f(float(e.sum()), '+,.0f'), f'{pos}/{valid}']

    rows = [rule_row('no filter (quote always)', fin)]
    for total, key, title, desc, sel, mean, t, share, pos, valid in found:
        rows.append(rule_row(f'{title} {desc}', sel))
    combo_sel = None
    combo_name = ''
    if useful:
        first = useful[0]
        combo_sel = first[4]
        combo_name = f'{first[2]} {first[3]}'
        best2 = None
        for key, title, codes, labels, _ in pre:
            if key == first[1]:
                continue
            r = best_threshold_rule(codes, labels, edge, combo_sel)
            if r is not None and r[0] > float(edge[combo_sel & fin].sum()) and (best2 is None or r[0] > best2[0]):
                best2 = (r[0], r[1], f'{title} {r[2]}')
        if best2 is not None:
            combo_sel = best2[1]
            combo_name = f'{combo_name} AND {best2[2]}'
            rows.append(rule_row(combo_name, combo_sel))

    lines = [
        f'## Filter rules (auto-generated, h = {horizon_ms} ms)',
        '',
        'Each pre-fill condition: the one-sided threshold (keep buckets at/above or at/below b) that maximises the '
        'total edge of the fills it keeps; then the best second condition on top of the best rule. '
        'In-sample selection: expect some shrinkage out of sample.',
        '',
        *_table(['rule: quote a side only when', 'fills kept', 'edge', 't', 'total ticks', 'chunks > 0'], rows),
        '',
    ]
    bullets = []
    for total, key, title, desc, sel, mean, t, share, pos, valid in useful[:4]:
        bullets.append(f'- quote a side only when **{title} {desc}**: edge {_f(mean)} ticks/fill on '
                       f'{_pct(share)} of fills (t {_f(t, "+.1f")}, {pos}/{valid} chunks > 0) vs {_f(base_mean)} unfiltered.')
    if not bullets:
        bullets.append('- no single pre-fill threshold keeps a positive total edge that beats quoting always.')
    lines += ['Rules:', '', *bullets, '']

    # Verdict
    def ok(e: np.ndarray, sel: np.ndarray) -> tuple[bool, float, float, int, int]:
        ee = e[sel & fin]
        pos, valid = chunk_stability(ee, f['chunk'][sel & fin], n_chunks, 1.0)
        mean, t = _mean(ee), _tstat(ee)
        enough = valid >= math.ceil(0.75 * n_chunks)  # thin subsets (few fills per chunk) cannot qualify
        return (mean > 0 and t >= 2 and enough and pos >= math.ceil(0.75 * valid)), mean, t, pos, valid

    v_base = ok(edge, fin)
    if v_base[0]:
        verdict = (f'VIABLE unconditionally: a front-of-queue touch maker earns {_f(v_base[1])} ticks/fill at '
                   f'{horizon_ms} ms (t {_f(v_base[2], "+.1f")}, {v_base[3]}/{v_base[4]} chunks > 0).')
    elif combo_sel is not None and ok(edge, combo_sel)[0]:
        v = ok(edge, combo_sel)
        verdict = (f'VIABLE ONLY WITH FILTER: {combo_name} -> {_f(v[1])} ticks/fill (t {_f(v[2], "+.1f")}, '
                   f'{v[3]}/{v[4]} chunks > 0) vs {_f(base_mean)} unfiltered.')
    elif useful and ok(edge, useful[0][4])[0]:
        v = ok(edge, useful[0][4])
        verdict = (f'VIABLE ONLY WITH FILTER: {useful[0][2]} {useful[0][3]} -> {_f(v[1])} ticks/fill '
                   f'(t {_f(v[2], "+.1f")}, {v[3]}/{v[4]} chunks > 0) vs {_f(base_mean)} unfiltered.')
    else:
        best = max(useful, key=lambda r: r[5]) if useful else None
        verdict = (f'NOT VIABLE at the touch: unfiltered edge {_f(base_mean)} ticks/fill at {horizon_ms} ms'
                   + (f', best filter ({best[2]} {best[3]}) {_f(best[5])} on {_pct(best[7])} of fills' if best else '')
                   + f'; no rule is positive with t >= 2 in >= 3/4 of chunks (each chunk >= {MIN_CHUNK_FILLS} fills).')
    lines += [f'**Verdict:** {verdict}', '']
    return lines, {'verdict': verdict, 'base_mean': base_mean, 'base_chunks': f'{base_same}/{base_valid}',
                   'rules': bullets}


def simulator_section(d: dict[str, np.ndarray], f: dict[str, np.ndarray]) -> list[str]:
    ts = d['ts_us']
    n = len(ts)
    dt = np.diff(ts, append=ts[-1]).astype(np.float64)
    two = (d['bid_px'] > 0) & (d['ask_px'] > 0)
    wt = dt * two
    rows = []
    for name, cnt, qty in (('bid', d['bid_cnt1'], d['bid_qty'][:, 0]), ('ask', d['ask_cnt1'], d['ask_qty'][:, 0])):
        single = cnt == 1
        rows.append([name, _pct(float(single[two].mean())), _pct(float((wt * single).sum() / max(wt.sum(), 1))),
                     f'{np.median(cnt[two]):.0f}', f'{np.median(qty[two]):.0f}'])
    lines = [
        '## Simulator specifics: queues at the touch',
        '',
        *_table(['side', 'single-order touch (events)', 'single-order touch (time)', 'median orders at touch',
                 'median lots at touch'], rows),
        '',
    ]
    # Queue ahead of a newly joined limit order at the existing touch.
    i = np.arange(1, n)
    p = i - 1
    ev_type, ev_side, ev_px = d['ev_type'][i], d['ev_side'][i], d['ev_px'][i]
    rows = []
    qs = (0.25, 0.5, 0.75, 0.9)
    for name, sgn, px, qty, cnt in (('bid', 1, d['bid_px'], d['bid_qty'][:, 0], d['bid_cnt1']),
                                    ('ask', -1, d['ask_px'], d['ask_qty'][:, 0], d['ask_cnt1'])):
        m = (ev_type == EV_LIMIT) & (ev_side == sgn) & (ev_px == px[p]) & (px[p] > 0)
        if m.sum() == 0:
            continue
        lots = qty[p][m]
        ords = cnt[p][m]
        rows.append([name, f'{int(m.sum()):,}', ' / '.join(f'{np.quantile(lots, q):.0f}' for q in qs),
                     ' / '.join(f'{np.quantile(ords, q):.0f}' for q in qs)])
    lines += ['Queue ahead of a newly joined order (market limit inserts that join the existing touch price):', '',
              *_table(['side', 'joins', 'lots ahead p25/p50/p75/p90', 'orders ahead p25/p50/p75/p90'], rows), '']
    # Depletion of the touch.
    dur_min = max((ts[-1] - ts[0]) / 60e6, 1e-9)
    dep = f['sweep'] >= 1
    rows = []
    for name, sgn, px in (('bid', 1, d['bid_px']), ('ask', -1, d['ask_px'])):
        before, after = px[p], px[i]
        retreat = (before > 0) & ((after < before) if sgn > 0 else ((after > before) | (after == 0)))
        by_trade = retreat & (d['trade_vol'][i] * sgn < 0)
        by_cancel = retreat & (d['ev_type'][i] == 2)
        side_fills = f['s'] == sgn
        rows.append([name, f'{int(side_fills.sum()):,}', _pct(float(dep[side_fills].mean()) if side_fills.any() else float('nan')),
                     f'{by_trade.sum() / dur_min:.1f}', f'{int(retreat.sum()):,}', _pct(float(by_trade.sum() / max(retreat.sum(), 1))),
                     _pct(float(by_cancel.sum() / max(retreat.sum(), 1)))])
    lines += ['Touch depletion (a joiner at the back of the queue is filled only when the taker empties the level):', '',
              *_table(['side', 'taker fills', 'fills that deplete the touch', 'depletions / min', 'touch retreats',
                       'retreat by trade', 'retreat by cancel'], rows), '']
    single_fill = f['touch_cnt_before'] == 1
    lines += [f'At taker fills the touch held a single order before the fill in {_pct(float(single_fill.mean()))} of cases.', '']
    return lines


def strategy_check_section(d: dict[str, np.ndarray], run_dir: Path, f: dict[str, np.ndarray]) -> tuple[list[str], dict[str, float]]:
    """Mark out the strategy's own maker fills (trades.csv of a --full-logs run) against the same mid."""
    tr = pd.read_csv(run_dir / 'trades.csv')
    mine = tr[tr.maker_owner == 'user']
    taker_fills = tr[tr.taker_owner == 'user']
    ts = d['ts_us']
    mid = mid_ticks(d, ffill=True)
    s = np.where(mine.taker_side.to_numpy() == 'A', 1, -1)  # taker sold -> we bought
    t = mine.timestamp_us.to_numpy()
    px = mine.price.to_numpy().astype(np.float64)
    q = mine.quantity.to_numpy().astype(np.float64)
    j = np.searchsorted(ts, t, side='left') - 1  # state before the fill's row
    mid0 = mid[np.maximum(j, 0)]
    rows = []
    facts = {}
    for h in HORIZONS_MS:
        m = mid_at_or_before(ts, mid, t + h * 1000)
        mk = s * (m - px)
        fin = np.isfinite(mk)
        vw = float(np.average(mk[fin], weights=q[fin])) if fin.any() else float('nan')
        facts[f'strat_{h}'] = vw
        rows.append([f'{h} ms', f'{int(fin.sum()):,}', _f(_mean(s * (mid0 - px))), _f(_mean(mk)), _f(vw),
                     _f(_mean(-s * (m - mid0))), _f(_mean(f[f'mk{h}'])), _f(_mean(f[f'mk{h}'][f['sweep'] >= 1]))])
    lines = [
        f'## Strategy check: own maker fills from `{run_dir}`',
        '',
        f'{len(mine):,} maker fills ({int(mine.quantity.sum()):,} lots), {len(taker_fills):,} taker fills. '
        'Markout vs the features (market-only) mid; compare with the touch-maker columns.',
        '',
        *_table(['h', 'fills', 'effective half', 'edge / fill', 'edge / lot', 'impact', 'tool: front', 'tool: back (depleted)'], rows),
        '',
    ]
    return lines, facts


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Touch-maker markouts, adverse selection and filter rules')
    parser.add_argument('csv', type=Path, help='tape CSV')
    parser.add_argument('--npz', type=Path, default=None, help='features npz (default data/features/<stem>.npz)')
    parser.add_argument('--out', type=Path, default=None, help='report (default research/reports/<stem>_mm_edge.md)')
    parser.add_argument('--horizon-ms', type=int, default=1000, choices=HORIZONS_MS, help='primary horizon for filters')
    parser.add_argument('--lag-us', type=int, default=LATENCY_US, help='condition on the book this long before the fill')
    parser.add_argument('--chunks', type=int, default=8, help='equal-event chunks for stability')
    parser.add_argument('--run-dir', type=Path, default=None, help='research.run --full-logs run dir to cross-check')
    args = parser.parse_args(argv)

    start = time.perf_counter()
    npz = ensure_features(args.csv, args.npz)
    d = load_features(npz)
    t_load = time.perf_counter() - start
    f = touch_fills(d, args.lag_us, args.chunks)
    conds = conditions(f)
    ts = d['ts_us']
    dur_s = (ts[-1] - ts[0]) / 1e6
    n_fill = len(f['s'])

    dec_lines, dec = decomposition_section(f)
    cond_lines, top = conditional_section(f, conds, args.horizon_ms, args.chunks)
    rule_lines, rules = rules_section(f, conds, args.horizon_ms, args.chunks)
    sim_lines = simulator_section(d, f)
    chk_lines, chk = strategy_check_section(d, args.run_dir, f) if args.run_dir else ([], {})

    h = args.horizon_ms
    lines = [
        f'# Market-making edge - `{args.csv}`',
        '',
        f'Generated {datetime.now():%Y-%m-%d %H:%M} by `research.mm_edge` from `{npz}`. Prices in ticks. '
        f'{len(ts):,} events over {dur_s:,.0f} s; {n_fill:,} taker fills of the touch ({n_fill / max(dur_s, 1e-9) * 60:,.1f}/min). '
        f'Conditions from the book {args.lag_us} us before each fill.',
        '',
        '## Summary',
        '',
        f'- **{rules["verdict"]}**',
        f'- front-of-queue touch maker edge: ' + ', '.join(f'{x} ms {_f(dec[f"front_{x}"])}' for x in HORIZONS_MS) + ' ticks/fill',
        ('- back-of-queue joiner (filled only on depletion): ' + ', '.join(f'{x} ms {_f(dec[f"back_{x}"])}' for x in HORIZONS_MS)
         if dec['back_1000'] == dec['back_1000'] else '- back-of-queue joiner: no taker depleted the touch on this tape (a joiner at the back is never filled)'),
        f'- unfiltered edge at {h} ms positive in {rules["base_chunks"]} chunks',
        '- queue position, edge at 1000 ms with Q lots ahead: '
        + ', '.join(f'Q={q} {_f(dec[f"queue_{q}"])}' for q in QUEUE_AHEAD if f'queue_{q}' in dec),
        *rules['rules'],
        *(['- strategy own maker fills, edge/lot: ' + ', '.join(f'{x} ms {_f(chk[f"strat_{x}"])}' for x in HORIZONS_MS)] if chk else []),
        '',
        *dec_lines,
        '',
        *rule_lines,
        *cond_lines,
        *sim_lines,
        *chk_lines,
        'Method: see the `research/mm_edge.py` docstring. t-stats assume independent fills (optimistic in bursts).',
        '',
    ]
    out = args.out or REPORTS_DIR / f'{args.csv.stem}_mm_edge.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines), encoding='utf-8')

    print(f'{args.csv}: {len(ts):,} events, {n_fill:,} touch fills; front edge ' +
          ', '.join(f'{x}ms {_f(dec[f"front_{x}"])}' for x in HORIZONS_MS) + '; back(depleted) ' +
          ', '.join(f'{x}ms {_f(dec[f"back_{x}"])}' for x in HORIZONS_MS))
    print(rules['verdict'])
    for b in rules['rules']:
        print(b)
    if chk:
        print('strategy own maker fills edge/lot: ' + ', '.join(f'{x}ms {_f(chk[f"strat_{x}"])}' for x in HORIZONS_MS))
    print(f'report -> {out}  [{time.perf_counter() - start:.1f}s, load {t_load:.1f}s]')


if __name__ == '__main__':
    main()
