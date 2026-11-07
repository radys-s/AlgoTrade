"""Data card of one round's tape: what kind of market is this?

    uv run python -m research.profile data/round1.csv        # -> research/reports/round1.md

Tape section from the CSV (pandas); book section from the features npz (built with
`research.features` first when missing). Signal ICs use the same time alignment as
`research.signals`: signal at event t (rows <= t), label = mid change from t + 5 ms to t + 5 ms + h.
Each IC is shown per equal-event chunk (8 by default), so stability is visible, not just the mean.
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from research.features import (
    EV_CANCEL,
    EV_IOC,
    EV_LIMIT,
    cancel_source_rows,
    ensure_features,
    load_features,
    mid_ticks,
    read_tape,
    tape_timestamps_us,
)
from research.signals import LATENCY_US, REPORTS_DIR, future_mid_change, pearson, simple_signals, value_at_or_before

IC_HORIZONS_MS = (10, 100, 1000, 10000)
VOL_HORIZONS_MS = (10, 100, 1000, 10000, 60000)
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
MIN_VOL_SAMPLES = 30


def _pct(x: float) -> str:
    return f'{100 * x:.1f}%'


def _num(x: float, digits: int = 3) -> str:
    if x != x:
        return 'nan'
    if abs(x) >= 100 or x == int(x):
        return f'{x:,.0f}'
    return f'{x:.{digits}g}'


def _table(header: list[str], rows: list[list[object]]) -> list[str]:
    """Markdown table lines (first column left-aligned, the rest right-aligned)."""
    out = ['| ' + ' | '.join(header) + ' |', '|---|' + '---:|' * (len(header) - 1)]
    out += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return out


def _quantile_row(name: str, values: np.ndarray) -> list[object]:
    if len(values) == 0:
        return [name, 0, *(['-'] * (len(QUANTILES) + 2))]
    qs = np.quantile(values, QUANTILES)
    return [name, f'{len(values):,}', _num(float(values.mean())), *(_num(float(q)) for q in qs), _num(float(values.max()))]


def _autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 1:
        return float('nan')
    return pearson(x[:-lag].astype(np.float64), x[lag:].astype(np.float64))


# ---------------------------------------------------------------- tape


def tape_section(csv_path: Path, d: dict[str, np.ndarray], facts: dict[str, float]) -> list[str]:
    """Rows, timing, order mix, sizes, placement, lifetimes and activity over time."""
    tape = read_tape(csv_path)
    ts = tape_timestamps_us(tape)
    n = len(tape)
    ev_type, ev_side, ev_qty, ev_px = d['ev_type'], d['ev_side'], d['ev_qty'], d['ev_px']
    duration_s = (ts[-1] - ts[0]) / 1e6
    is_limit, is_ioc, is_cancel = ev_type == EV_LIMIT, ev_type == EV_IOC, ev_type == EV_CANCEL
    source = cancel_source_rows(tape)
    del tape
    facts.update(duration_s=duration_s, ioc_share=float(is_ioc.mean()), events_per_s=n / max(duration_s, 1e-9))

    lines = ['## Tape', '']
    same_ts = float(np.mean(np.diff(ts) == 0)) if n > 1 else 0.0
    lines += _table(
        ['metric', 'value'],
        [
            ['rows', f'{n:,}'],
            ['duration', f'{duration_s:,.1f} s ({duration_s / 3600:.2f} h), Time {ts[0] / 1e6:.3f} .. {ts[-1] / 1e6:.3f}'],
            ['events / s (mean)', _num(n / max(duration_s, 1e-9))],
            ['rows sharing the previous row\'s timestamp', _pct(same_ts)],
            ['insert limit (G) / insert IOC (F) / cancel', f'{_pct(is_limit.mean())} / {_pct(is_ioc.mean())} / {_pct(is_cancel.mean())}'],
            ['cancels of unknown ids', f'{int((is_cancel & (source < 0)).sum()):,}'],
            ['limit inserts bid / ask (count)', f'{_pct((is_limit & (ev_side > 0)).sum() / max(is_limit.sum(), 1))} / {_pct((is_limit & (ev_side < 0)).sum() / max(is_limit.sum(), 1))}'],
            ['IOC buy / sell (count)', f'{int((is_ioc & (ev_side > 0)).sum()):,} / {int((is_ioc & (ev_side < 0)).sum()):,}'],
            ['IOC buy / sell (volume)', f'{int(ev_qty[is_ioc & (ev_side > 0)].sum()):,} / {int(ev_qty[is_ioc & (ev_side < 0)].sum()):,}'],
            ['traded volume buyer- / seller-initiated', f'{int(d["trade_vol"].clip(min=0).sum()):,} / {int((-d["trade_vol"]).clip(min=0).sum()):,}'],
            ['insert price range (ticks)', f'{int(ev_px[~is_cancel].min()):,} .. {int(ev_px[~is_cancel].max()):,}'],
        ],
    )

    header = ['orders', 'n', 'mean', *(f'q{int(q * 100)}' for q in QUANTILES), 'max']
    lines += ['', '**Order sizes (lots)**', '']
    lines += _table(header, [_quantile_row('limit (G)', ev_qty[is_limit]), _quantile_row('IOC (F)', ev_qty[is_ioc])])

    # Limit placement against the book just before the insert (state after row i - 1).
    rows = np.flatnonzero(is_limit)
    rows = rows[rows > 0]
    prev = rows - 1
    side = ev_side[rows]
    px = ev_px[rows].astype(np.int64)
    own_best = np.where(side > 0, d['bid_px'][prev], d['ask_px'][prev]).astype(np.int64)
    opp_best = np.where(side > 0, d['ask_px'][prev], d['bid_px'][prev]).astype(np.int64)
    behind = np.where(side > 0, own_best - px, px - own_best)  # ticks behind the own-side touch (<0 = improves)
    crosses = (opp_best > 0) & np.where(side > 0, px >= opp_best, px <= opp_best)
    empty = own_best <= 0
    buckets = [
        ('crosses the spread (marketable)', crosses),
        ('own side empty', ~crosses & empty),
        ('inside the spread (improves)', ~crosses & ~empty & (behind < 0)),
        ('at the touch', ~crosses & ~empty & (behind == 0)),
        ('1-2 ticks behind', ~crosses & ~empty & (behind >= 1) & (behind <= 2)),
        ('3-5 ticks behind', ~crosses & ~empty & (behind >= 3) & (behind <= 5)),
        ('6-19 ticks behind', ~crosses & ~empty & (behind >= 6) & (behind <= 19)),
        ('20+ ticks behind (invisible)', ~crosses & ~empty & (behind >= 20)),
    ]
    facts['limit_at_touch'] = float(np.mean(~crosses & ~empty & (behind == 0))) if len(rows) else 0.0
    facts['limit_inside'] = float(np.mean(~crosses & ~empty & (behind < 0))) if len(rows) else 0.0
    lines += ['', '**Where limit orders are placed** (vs. the book just before the insert)', '']
    lines += _table(['placement', 'share'], [[name, _pct(mask.mean() if len(mask) else 0)] for name, mask in buckets])

    ioc_rows = np.flatnonzero(is_ioc)
    if len(ioc_rows):
        traded = d['trade_cnt'][ioc_rows] > 0
        mid = mid_ticks(d, ffill=True)
        prev_mid = mid[np.maximum(ioc_rows - 1, 0)]
        moved = np.nan_to_num(mid[ioc_rows] - prev_mid) != 0
        facts['ioc_moves_mid'] = float(moved.mean())
        lines += [
            '',
            f'IOC orders: {_pct(traded.mean())} trade anything, {_pct(moved.mean())} move the mid, '
            f'{d["trade_cnt"][ioc_rows].mean():.2f} fills per IOC on average.',
        ]

    cancels = np.flatnonzero(source >= 0)
    lifetime_ms = (ts[cancels] - ts[source[cancels]]) / 1000
    limit_cancelled = len(np.unique(source[cancels])) / max(int(is_limit.sum()), 1)
    facts['median_lifetime_ms'] = float(np.median(lifetime_ms)) if len(lifetime_ms) else float('nan')
    lines += ['', '**Order lifetime, insert -> cancel (ms)**', '']
    lines += _table(header, [_quantile_row('cancelled orders', lifetime_ms)])
    lines += [
        '',
        f'{_pct(limit_cancelled)} of limit orders are eventually cancelled (the rest fill, or live to the end); '
        f'{_pct(np.mean(lifetime_ms < 5)) if len(lifetime_ms) else "-"} of cancels come within 5 ms of the insert '
        '(faster than our latency).',
    ]

    # Activity by 5-minute buckets.
    bucket = ((ts - ts[0]) // 300_000_000).astype(np.int64)
    mid_ff = mid_ticks(d, ffill=True)
    frame = pd.DataFrame(
        {
            'bucket': bucket,
            'ioc': is_ioc,
            'fills': d['trade_cnt'],
            'volume': np.abs(d['trade_vol']),
            'mid': mid_ff,
        }
    )
    grouped = frame.groupby('bucket')
    agg = grouped.agg(
        events=('ioc', 'size'), ioc=('ioc', 'sum'), fills=('fills', 'sum'), volume=('volume', 'sum'),
        mid_lo=('mid', 'min'), mid_hi=('mid', 'max'), mid_last=('mid', 'last'),
    )  # fmt: skip
    rates = agg['events'] / 300
    facts['activity_ratio'] = float(rates.iloc[:-1].max() / max(rates.iloc[:-1].min(), 1e-9)) if len(rates) > 2 else 1.0
    lines += ['', '**Activity by 5-minute bucket**', '']
    lines += _table(
        ['minute', 'events', 'events/s', 'IOC', 'fills', 'traded vol', 'mid low', 'mid high', 'mid end'],
        [
            [
                f'{int(b) * 5}', f'{int(r.events):,}', f'{r.events / 300:,.1f}', f'{int(r.ioc):,}', f'{int(r.fills):,}',
                f'{int(r.volume):,}', _num(r.mid_lo, 6), _num(r.mid_hi, 6), _num(r.mid_last, 6),
            ]
            for b, r in agg.iterrows()
        ],
    )  # fmt: skip
    return lines


# ---------------------------------------------------------------- book


def book_section(d: dict[str, np.ndarray], facts: dict[str, float]) -> list[str]:
    """Spread, touch sizes, mid dynamics, volatility by horizon, autocorrelations."""
    ts = d['ts_us']
    two_sided = (d['bid_px'] > 0) & (d['ask_px'] > 0)
    dt = np.diff(ts, append=ts[-1]).astype(np.float64)  # time each state lasted
    spread = (d['ask_px'] - d['bid_px'])[two_sided]
    weights = dt[two_sided]
    facts['two_sided'] = float(two_sided.mean())

    lines = ['## Book', '', f'Two-sided book after {_pct(two_sided.mean())} of events.', '']
    values = [1, 2, 3, 4, 5]
    rows = []
    total_w = max(weights.sum(), 1e-9)
    for v in values:
        mask = spread == v if v < 5 else spread >= 5
        rows.append([f'{v}' if v < 5 else '5+', _pct(mask.mean() if len(spread) else 0), _pct(weights[mask].sum() / total_w)])
    lines += ['**Spread (ticks)**', '']
    lines += _table(['spread', 'share of events', 'share of time'], rows)
    facts['spread1_time'] = float(weights[spread == 1].sum() / total_w) if len(spread) else 0.0
    facts['spread_median'] = float(np.median(spread)) if len(spread) else float('nan')

    header = ['quantity', 'n', 'mean', *(f'q{int(q * 100)}' for q in QUANTILES), 'max']
    lines += ['', '**Touch size and depth (lots / orders, two-sided events)**', '']
    lines += _table(
        header,
        [
            _quantile_row('best bid qty', d['bid_qty'][two_sided, 0]),
            _quantile_row('best ask qty', d['ask_qty'][two_sided, 0]),
            _quantile_row('orders at best bid', d['bid_cnt1'][two_sided]),
            _quantile_row('orders at best ask', d['ask_cnt1'][two_sided]),
            _quantile_row('top-5 bid qty', d['bid_qty'][two_sided].sum(1)),
            _quantile_row('top-5 ask qty', d['ask_qty'][two_sided].sum(1)),
        ],
    )
    facts['touch_median'] = float(np.median(np.concatenate([d['bid_qty'][two_sided, 0], d['ask_qty'][two_sided, 0]]))) if two_sided.any() else 0.0
    ioc = d['ev_type'] == EV_IOC
    facts['ioc_size_median'] = float(np.median(d['ev_qty'][ioc])) if ioc.any() else float('nan')

    mid = mid_ticks(d, ffill=True)
    dmid = np.diff(mid)
    dmid = dmid[np.isfinite(dmid)]
    moves = dmid[dmid != 0]
    duration_s = max((ts[-1] - ts[0]) / 1e6, 1e-9)
    facts['mid_change_share'] = float(len(moves) / max(len(dmid), 1))
    facts['mid_changes_per_s'] = len(moves) / duration_s
    size = np.abs(moves)
    lines += [
        '',
        f'**Mid changes**: after {_pct(facts["mid_change_share"])} of events, {facts["mid_changes_per_s"]:.2f} per second; '
        f'size 0.5 tick {_pct(np.mean(size == 0.5) if len(size) else 0)}, 1 tick {_pct(np.mean(size == 1) if len(size) else 0)}, '
        f'1.5-2 ticks {_pct(np.mean((size >= 1.5) & (size <= 2)) if len(size) else 0)}, '
        f'more {_pct(np.mean(size > 2) if len(size) else 0)}.',
        '',
    ]

    # Calendar-time volatility on regular grids (mid of the last event at or before each grid time).
    first_valid = np.flatnonzero(np.isfinite(mid))
    rows = []
    var_1s = float('nan')
    vol_stats: dict[int, tuple[float, float, float, int]] = {}
    if len(first_valid):
        t0 = ts[first_valid[0]]
        for h in VOL_HORIZONS_MS:
            grid = np.arange(t0, ts[-1] + 1, h * 1000)
            if len(grid) < 3:
                continue
            change = np.diff(value_at_or_before(ts, mid, grid))
            vol_stats[h] = (float(change.std()), float(np.abs(change).mean()), float(np.mean(change == 0)), len(change))
            if h == 1000:
                var_1s = vol_stats[h][0] ** 2
    ratios: dict[int, float] = {}
    for h, (std, mad, zero, count) in vol_stats.items():
        ratio = std**2 / (var_1s * h / 1000) if var_1s == var_1s and var_1s > 0 else float('nan')
        if count < MIN_VOL_SAMPLES:
            ratio = float('nan')  # too few non-overlapping intervals to say anything
        ratios[h] = ratio
        vr = f'{ratio:.2f}' if ratio == ratio else f'n/a (<{MIN_VOL_SAMPLES} samples)'
        rows.append([f'{h:,} ms', f'{count:,}', f'{std:.3f}', f'{mad:.3f}', _pct(zero), f'{std / math.sqrt(h / 1000):.3f}', vr])
    facts['vol_1s'] = vol_stats.get(1000, (float('nan'),))[0]
    facts['vr_60s'] = ratios.get(VOL_HORIZONS_MS[-1], float('nan'))
    lines += ['**Mid-change volatility by horizon** (regular time grid; VR = variance ratio vs. 1 s, <1 mean reversion, >1 trending)', '']
    lines += _table(['horizon', 'samples', 'std (ticks)', 'mean abs', 'P(no change)', 'std / sqrt(s)', 'VR vs 1 s'], rows)

    # Successive mid changes (event time) and trade signs.
    signs = np.sign(moves)
    reversal = float(np.mean(signs[1:] != signs[:-1])) if len(signs) > 1 else float('nan')
    facts['mid_reversal'] = reversal
    facts['mid_ac1'] = _autocorr(moves, 1)
    trade_sign = np.sign(d['trade_vol'][d['trade_cnt'] > 0])
    trade_lags = (1, 2, 5, 10, 50)
    facts['trade_ac1'] = _autocorr(trade_sign, 1)
    lines += [
        '',
        f'**Successive mid changes** ({len(moves):,}): autocorrelation lag 1..5 = '
        + ', '.join(f'{_autocorr(moves, k):+.3f}' for k in range(1, 6))
        + f'; P(next change reverses) = {reversal:.3f} (0.5 = random walk, >0.5 bounce / mean reversion).',
        '',
        f'**Trade-sign autocorrelation** ({len(trade_sign):,} trading events): '
        + ', '.join(f'lag {k} {_autocorr(trade_sign, k):+.3f}' for k in trade_lags)
        + f'; buyer-initiated share {_pct(np.mean(trade_sign > 0) if len(trade_sign) else 0)}.',
    ]
    return lines


# ---------------------------------------------------------------- information coefficients


def ic_section(d: dict[str, np.ndarray], n_chunks: int, facts: dict[str, object]) -> list[str]:
    """Spearman / Pearson IC of the simple signals vs. future mid change, per chunk and horizon."""
    ts = d['ts_us']
    n = len(ts)
    signals = simple_signals(d)
    mid_ff = mid_ticks(d, ffill=True)
    labels = {h: future_mid_change(ts, mid_ff, h * 1000) for h in IC_HORIZONS_MS}
    valid = np.all([np.isfinite(v) for v in (*signals.values(), *labels.values())], axis=0)
    chunk = np.arange(n) * n_chunks // n

    spear: dict[tuple[str, int], list[float]] = {}
    pear: dict[tuple[str, int], list[float]] = {}
    for c in range(n_chunks):
        rows = valid & (chunk == c)
        if rows.sum() < 100:
            for name in signals:
                for h in IC_HORIZONS_MS:
                    spear.setdefault((name, h), []).append(float('nan'))
                    pear.setdefault((name, h), []).append(float('nan'))
            continue
        label_ranks = {h: rankdata(y[rows]) for h, y in labels.items()}
        for name, x in signals.items():
            x_rows = x[rows]
            x_ranks = rankdata(x_rows)
            for h, y in labels.items():
                spear.setdefault((name, h), []).append(pearson(x_ranks, label_ranks[h]))
                pear.setdefault((name, h), []).append(pearson(x_rows, y[rows]))

    def summary(values: list[float]) -> tuple[float, float, int, int]:
        arr = np.array(values)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return float('nan'), float('nan'), 0, 0
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1)) if len(arr) > 1 else float('nan')
        t = mean / (sd / math.sqrt(len(arr))) if sd and sd == sd and sd > 0 else float('nan')
        return mean, t, int(np.sum(np.sign(arr) == np.sign(mean))), len(arr)

    label_std = {h: float(np.nanstd(labels[h][valid])) for h in IC_HORIZONS_MS}
    label_nz = {h: float(np.mean(labels[h][valid] != 0)) for h in IC_HORIZONS_MS}
    lines = [
        '## Signals vs. future mid change',
        '',
        f'Label: mid(t + {LATENCY_US // 1000} ms + h) - mid(t + {LATENCY_US // 1000} ms), mid = last two-sided mid of the book '
        f'after all events before that time (what an order sent at t would meet). {int(valid.sum()):,} events with a two-sided '
        f'book and a known label, cut into {n_chunks} equal-event chunks. Signals use rows <= t only.',
        '',
        'Label std (ticks): ' + ', '.join(f'{h} ms {label_std[h]:.3f} ({_pct(label_nz[h])} non-zero)' for h in IC_HORIZONS_MS) + '.',
        '',
        'Signals: `imb_l1` = (b1-a1)/(b1+a1); `imb_top5` same over 5 price levels; `micro_minus_mid` = microprice - mid (ticks); '
        '`flow_Nev` = signed traded volume (+ = buyer-initiated) of the last N events; `flow_100ms` / `flow_1s` = over the last T.',
        '',
        '**Mean Spearman IC over chunks** (t-stat across chunks, chunks agreeing in sign / chunks)',
        '',
    ]
    rows = []
    best: tuple[float, str, int, float, int, int] | None = None
    for name in signals:
        row: list[object] = [f'`{name}`']
        for h in IC_HORIZONS_MS:
            mean, t, agree, total = summary(spear[(name, h)])
            row.append(f'{mean:+.3f} (t {t:+.1f}, {agree}/{total})' if mean == mean else 'nan')
            if mean == mean and t == t and (best is None or abs(t) > abs(best[0])):
                best = (t, name, h, mean, agree, total)
        rows.append(row)
    lines += _table(['signal', *(f'{h} ms' for h in IC_HORIZONS_MS)], rows)
    facts['best_signal'] = best
    facts['label_std'] = label_std

    lines += ['', '**Pooled-chunk mean Pearson IC**', '']
    lines += _table(
        ['signal', *(f'{h} ms' for h in IC_HORIZONS_MS)],
        [[f'`{name}`', *(f'{np.nanmean(pear[(name, h)]):+.3f}' for h in IC_HORIZONS_MS)] for name in signals],
    )
    for h in IC_HORIZONS_MS:
        lines += ['', f'**Spearman IC per chunk, h = {h} ms**', '']
        lines += _table(
            ['signal', *(str(c + 1) for c in range(n_chunks))],
            [[f'`{name}`', *(f'{v:+.3f}' if v == v else 'nan' for v in spear[(name, h)])] for name in signals],
        )
    return lines


# ---------------------------------------------------------------- observations


def observations(facts: dict[str, object]) -> list[str]:
    """3-5 auto-generated bullets from the numbers above."""
    bullets: list[str] = []
    two_sided = float(facts.get('two_sided', 1.0))
    if two_sided < 0.95:
        bullets.append(f'The book is one-sided or empty after {_pct(1 - two_sided)} of events -> guard every use of '
                       'best bid/ask, and expect the mid (last two-sided) to jump when a side reappears.')
    spread1 = float(facts.get('spread1_time', 0.0))
    if spread1 >= 0.9:
        bullets.append(
            f'Spread is 1 tick {_pct(spread1)} of the time -> you cannot improve inside the spread; a maker edge has to come '
            f'from queue position at the touch (limit orders joining the touch: {_pct(float(facts.get("limit_at_touch", 0)))}).'
        )
    else:
        bullets.append(
            f'Spread is 1 tick only {_pct(spread1)} of the time (median {facts.get("spread_median", float("nan")):g} ticks) -> '
            f'room to quote inside the spread (inserts improving the touch: {_pct(float(facts.get("limit_inside", 0)))}).'
        )
    reversal = float(facts.get('mid_reversal', float('nan')))
    if reversal == reversal:
        if reversal > 0.55:
            bullets.append(f'Mid changes mean-revert at the tick level (P(next change reverses) = {reversal:.2f}): '
                           'fade single-tick moves / expect bid-ask bounce rather than chase them.')
        elif reversal < 0.45:
            bullets.append(f'Mid changes trend at the tick level (P(reversal) = {reversal:.2f}): '
                           'a move tends to be followed by another move the same way.')
        else:
            bullets.append(f'Successive mid changes are close to a random walk (P(reversal) = {reversal:.2f}).')
    vr = float(facts.get('vr_60s', float('nan')))
    if vr == vr:
        if vr < 0.8:
            bullets.append(f'Variance ratio 60 s vs 1 s = {vr:.2f} < 1: prices mean-revert over a minute; '
                           'inventory held a while tends to come back.')
        elif vr > 1.25:
            bullets.append(f'Variance ratio 60 s vs 1 s = {vr:.2f} > 1: prices trend over a minute; '
                           'inventory against the trend is dangerous, momentum may pay.')
    best = facts.get('best_signal')
    if best:
        t, name, h, mean, agree, total = best  # type: ignore[misc]
        label_std = facts.get('label_std', {}).get(h, float('nan'))  # type: ignore[union-attr]
        bullets.append(
            f'Most reliable simple signal: `{name}` at {h} ms, mean rank IC {mean:+.3f} (t {t:+.1f}), same sign in '
            f'{agree}/{total} chunks; the whole move it predicts has std {label_std:.2f} ticks at {h} ms, vs. '
            f'{float(facts.get("spread_median", float("nan"))) / 2:g} tick(s) half-spread paid by an IOC.'
        )
    trade_ac1 = float(facts.get('trade_ac1', float('nan')))
    if trade_ac1 == trade_ac1 and abs(trade_ac1) > 0.2:
        bullets.append(f'Trade signs are {"persistent" if trade_ac1 > 0 else "alternating"} (lag-1 autocorrelation '
                       f'{trade_ac1:+.2f}): {"aggressive flow comes in runs" if trade_ac1 > 0 else "buys and sells alternate"}.')
    vol = float(facts.get('vol_1s', float('nan')))
    if vol == vol and len(bullets) < 5:
        bullets.append(f'1 s mid-change std is {vol:.2f} ticks vs a typical spread of {facts.get("spread_median", float("nan")):g} '
                       f'tick(s): {"crossing the spread to take short-horizon moves is expensive" if vol < float(facts.get("spread_median", 1)) else "moves are large relative to the spread"}.')
    ratio = float(facts.get('activity_ratio', 1.0))
    if ratio > 3 and len(bullets) < 5:
        bullets.append(f'Activity is uneven: the busiest 5-minute bucket has {ratio:.1f}x the events of the quietest.')
    return [f'- {b}' for b in bullets[:5]]


def main(argv: list[str] | None = None) -> None:
    """Write research/reports/<stem>.md for a tape CSV and print a short summary."""
    parser = argparse.ArgumentParser(description='Data card of a tape: tape stats, book stats, simple-signal ICs')
    parser.add_argument('csv', type=Path, help='tape CSV')
    parser.add_argument('--chunks', type=int, default=8, help='equal-event chunks for IC stability')
    parser.add_argument('--out', type=Path, default=None, help='report path (default research/reports/<stem>.md)')
    parser.add_argument('--npz', type=Path, default=None, help='features npz (default data/features/<stem>.npz, built if missing)')
    args = parser.parse_args(argv)

    start = time.perf_counter()
    npz_path = ensure_features(args.csv, args.npz)
    d = load_features(npz_path)
    facts: dict[str, object] = {}
    timings = {}
    t = time.perf_counter()
    tape_lines = tape_section(args.csv, d, facts)
    timings['tape'] = time.perf_counter() - t
    t = time.perf_counter()
    book_lines = book_section(d, facts)
    timings['book'] = time.perf_counter() - t
    t = time.perf_counter()
    ic_lines = ic_section(d, args.chunks, facts)
    timings['ic'] = time.perf_counter() - t
    bullets = observations(facts)

    lines = [
        f'# Data card - `{args.csv}`',
        '',
        f'Generated {datetime.now():%Y-%m-%d %H:%M} by `research.profile` from `{npz_path}` '
        f'(features built in {float(d.get("build_seconds", float("nan"))):.1f} s). Prices in ticks.',
        '',
        '## Observations',
        '',
        *bullets,
        '',
        *tape_lines,
        '',
        *book_lines,
        '',
        *ic_lines,
        '',
    ]
    out = args.out or REPORTS_DIR / f'{args.csv.stem}.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines), encoding='utf-8')

    print(f'{args.csv}: {len(d["ts_us"]):,} events, {float(facts["duration_s"]):,.0f} s, '
          f'{float(facts["events_per_s"]):,.0f} events/s, IOC {_pct(float(facts["ioc_share"]))}, '
          f'spread 1 tick {_pct(float(facts["spread1_time"]))} of the time, 1 s vol {float(facts["vol_1s"]):.2f} ticks')
    print('\n'.join(bullets))
    print(f'report -> {out}  [{time.perf_counter() - start:.1f}s: ' + ', '.join(f'{k} {v:.1f}s' for k, v in timings.items()) + ']')


if __name__ == '__main__':
    main()
