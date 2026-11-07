"""Short-horizon signal research on a features npz: feature matrix, labels, walk-forward CV, exports.

    uv run python -m research.signals data/features/round1.npz --horizon-ms 1000
    uv run python -m research.signals data/round1.csv --horizon-ms 100 --export-linear
    uv run python -m research.signals data/features/round1.npz --export-dataset data/features/round1_ml.npz
    uv run python -m research.signals data/round1.csv --parity --no-hgb    # OnlineFeatures == offline?

Time alignment (the part that matters):
    * A row is one tape event i at time t = ts_us[i]; features use rows <= i only (the state the
      strategy sees in that callback; later rows at the same timestamp are NOT included).
    * The label is the mid change from t + 5 ms (when an order sent at t reaches the book) to
      t + 5 ms + h, where "mid at T" is the last two-sided mid after all events with ts < T,
      exactly the book an order activated at T meets. Labels past the end of the tape are NaN.
    * Walk-forward: the tape is cut into equal-event chunks; fold k trains on chunks < k and
      tests on chunk k, never shuffled. Training rows whose label window reaches into the test
      chunk are purged.

Features (all in ticks or plain numbers; see FEATURE_DOC) are designed to be cheap to update
per event in a strategy, and `OnlineFeatures` is a pure-Python reference implementation that is
checked against the vectorized one (`check_online_parity`). `export_linear` prints a dict for
`predict_linear`; `export_dataset` writes X / y / timestamps for an MLP trained elsewhere (e.g.
`research/train_mlp.py` on Colab), whose weights `mlp_predict` evaluates with numpy.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from simulator.main import ORDER_ACCEPT_DELAY_MS
from simulator.models import OrderBookSnapshot, Trade

from research.features import N_LEVELS, aggregate_views, ensure_features, load_features, mid_ticks

LATENCY_US = ORDER_ACCEPT_DELAY_MS * 1_000
REPORTS_DIR = Path(__file__).parent / 'reports'

FLOW_EVENTS = (1, 10, 100)
FLOW_MS = (10, 100, 1000, 10000)
RET_MS = (100, 1000, 10000)

FEATURE_DOC: dict[str, str] = {
    'spread': 'best ask - best bid (ticks)',
    'imb1': '(b1 - a1) / (b1 + a1), b1/a1 = quantity at the best bid/ask price',
    'imb2': 'same with the quantity of the best 2 prices per side',
    'imb3': 'same with the best 3 prices per side',
    'imb5': 'same with the best 5 prices per side',
    'micro': 'microprice - mid = spread * imb1 / 2 (ticks)',
    'cnt_imb1': '(nb - na) / (nb + na), nb/na = number of orders at the best bid/ask',
    'depth1_log': 'log1p(b1 + a1)',
    'flow_ev{N}': 'signed traded volume (+ buyer-initiated) of the last N events incl. this one',
    'flow_ms{T}': 'signed traded volume of events with ts in (t - T ms, t]',
    'ntr_ms1000': 'number of fills in (t - 1 s, t]',
    'nev_ms1000': 'number of tape events in (t - 1 s, t]',
    'ret_ms{T}': 'mid now - mid after the last event with ts <= t - T ms (ticks; one-sided books keep the last mid)',
    'last_dmid': 'size of the most recent non-zero mid change (ticks, signed; 0 before the first)',
    'dt_mid_log': 'log1p(ms since the most recent mid change, or since the first event)',
    'ltp_dev': 'last trade price - mid (ticks; 0 before the first trade)',
}


def feature_names(
    flow_events: tuple[int, ...] = FLOW_EVENTS, flow_ms: tuple[int, ...] = FLOW_MS, ret_ms: tuple[int, ...] = RET_MS
) -> list[str]:
    """Return the column names of `build_matrix` for the given lookbacks."""
    return [
        'spread', 'imb1', 'imb2', 'imb3', 'imb5', 'micro', 'cnt_imb1', 'depth1_log',
        *(f'flow_ev{n}' for n in flow_events),
        *(f'flow_ms{t}' for t in flow_ms),
        'ntr_ms1000', 'nev_ms1000',
        *(f'ret_ms{t}' for t in ret_ms),
        'last_dmid', 'dt_mid_log', 'ltp_dev',
    ]  # fmt: skip


# ---------------------------------------------------------------- vectorized building blocks


def _cumsum0(x: np.ndarray) -> np.ndarray:
    """Return the cumulative sum with a leading 0 (float64): c[i+1] = x[0] + ... + x[i]."""
    out = np.zeros(len(x) + 1)
    np.cumsum(x, out=out[1:])
    return out


def sum_last_events(x: np.ndarray, n: int) -> np.ndarray:
    """Sum of x over the last `n` rows including the current one (fewer at the start)."""
    c = _cumsum0(x)
    i = np.arange(1, len(x) + 1)
    return c[i] - c[np.maximum(i - n, 0)]


def sum_last_us(ts_us: np.ndarray, x: np.ndarray, window_us: int) -> np.ndarray:
    """Sum of x over rows j <= i with ts[j] > ts[i] - window_us (the window (t - T, t])."""
    c = _cumsum0(x)
    first = np.searchsorted(ts_us, ts_us - window_us, side='right')
    return c[np.arange(1, len(x) + 1)] - c[first]


def value_at_or_before(ts_us: np.ndarray, values: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """values[j] for the last row j with ts[j] <= t_query (NaN when there is none)."""
    j = np.searchsorted(ts_us, t_query, side='right') - 1
    out = values[np.maximum(j, 0)].astype(np.float64)
    out[j < 0] = np.nan
    return out


def value_before(ts_us: np.ndarray, values: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """values[j] for the last row j with ts[j] < t_query: the state an order activated at t_query meets."""
    j = np.searchsorted(ts_us, t_query, side='left') - 1
    out = values[np.maximum(j, 0)].astype(np.float64)
    out[j < 0] = np.nan
    return out


def future_mid_change(
    ts_us: np.ndarray, mid_ffill: np.ndarray, horizon_us: int, latency_us: int = LATENCY_US
) -> np.ndarray:
    """Label: mid(t + latency + h) - mid(t + latency), mid(T) = state after all events with ts < T."""
    start = ts_us + latency_us
    end = start + horizon_us
    y = value_before(ts_us, mid_ffill, end) - value_before(ts_us, mid_ffill, start)
    y[end > ts_us[-1]] = np.nan  # the tape ends before the horizon: unknown
    return y


def _imbalance(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """(b - a) / (b + a), 0 where both are 0."""
    b = b.astype(np.float64)
    a = a.astype(np.float64)
    total = b + a
    return np.divide(b - a, total, out=np.zeros_like(total), where=total > 0)


def simple_signals(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The data card's signals: L1/top-5 imbalance, microprice - mid, trade flow by events and by time."""
    mid = mid_ticks(d)
    spread = (d['ask_px'] - d['bid_px']).astype(np.float64)
    imb1 = _imbalance(d['bid_qty'][:, 0], d['ask_qty'][:, 0])
    flow = d['trade_vol'].astype(np.float64)
    ts = d['ts_us']
    out = {
        'imb_l1': imb1,
        'imb_top5': _imbalance(d['bid_qty'].sum(axis=1), d['ask_qty'].sum(axis=1)),
        'micro_minus_mid': spread * imb1 / 2,
        'flow_10ev': sum_last_events(flow, 10),
        'flow_100ev': sum_last_events(flow, 100),
        'flow_100ms': sum_last_us(ts, flow, 100_000),
        'flow_1s': sum_last_us(ts, flow, 1_000_000),
    }
    invalid = np.isnan(mid)
    for values in out.values():
        values[invalid] = np.nan
    return out


def build_matrix(
    d: dict[str, np.ndarray],
    flow_events: tuple[int, ...] = FLOW_EVENTS,
    flow_ms: tuple[int, ...] = FLOW_MS,
    ret_ms: tuple[int, ...] = RET_MS,
    rows: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return X (float32, one row per event or per index in `rows`) and the feature names.

    Rows with a one-sided book get NaN book features; drop them with `np.isfinite(X).all(1)`.
    """
    ts = d['ts_us']
    n = len(ts)
    sel = slice(None) if rows is None else rows
    names = feature_names(flow_events, flow_ms, ret_ms)
    X = np.empty((n if rows is None else len(rows), len(names)), dtype=np.float32)
    mid = mid_ticks(d)
    invalid = np.isnan(mid)
    k = 0

    def put(column: np.ndarray) -> None:
        """Write the next column (NaN on one-sided rows); keeps only one full-length temporary alive."""
        nonlocal k
        X[:, k] = np.where(invalid, np.nan, column)[sel]
        k += 1

    bq, aq = d['bid_qty'], d['ask_qty']
    spread = (d['ask_px'] - d['bid_px']).astype(np.float64)
    imb1 = _imbalance(bq[:, 0], aq[:, 0])
    put(spread)
    put(imb1)
    for depth in (2, 3, N_LEVELS):
        put(_imbalance(bq[:, :depth].sum(1, dtype=np.int64), aq[:, :depth].sum(1, dtype=np.int64)))
    put(spread * imb1 / 2)
    del spread, imb1
    put(_imbalance(d['bid_cnt1'], d['ask_cnt1']))
    put(np.log1p(bq[:, 0].astype(np.float64) + aq[:, 0]))

    flow = d['trade_vol'].astype(np.float64)
    for count in flow_events:
        put(sum_last_events(flow, count))
    for t_ms in flow_ms:
        put(sum_last_us(ts, flow, t_ms * 1000))
    del flow
    put(sum_last_us(ts, d['trade_cnt'].astype(np.float64), 1_000_000))
    put(sum_last_us(ts, np.ones(n), 1_000_000))

    mid_ff = mid_ticks(d, ffill=True)
    for t_ms in ret_ms:
        put(mid_ff - value_at_or_before(ts, mid_ff, ts - t_ms * 1000))
    dmid = np.diff(mid_ff, prepend=np.nan)
    changed = np.isfinite(dmid) & (dmid != 0)
    last_change = np.maximum.accumulate(np.where(changed, np.arange(n), -1))
    put(np.where(last_change >= 0, dmid[np.maximum(last_change, 0)], 0.0))
    put(np.log1p((ts - np.where(last_change >= 0, ts[np.maximum(last_change, 0)], ts[0])) / 1000))
    ltp = d['last_trade_px'].astype(np.float64)
    put(np.where(ltp > 0, ltp - mid, 0.0))
    assert k == len(names)
    return X, names


def build_dataset(
    d: dict[str, np.ndarray],
    horizons_ms: tuple[int, ...],
    latency_us: int = LATENCY_US,
    max_rows: int | None = 3_000_000,
    n_chunks: int = 8,
) -> dict[str, np.ndarray]:
    """Features + labels for every usable event (strided down to `max_rows`), with chunk ids.

    Returns X, Y (one column per horizon), ts_us, row (index into the tape) and chunk (0..n_chunks-1,
    equal numbers of tape events), keeping rows where every feature and label is finite.
    """
    ts = d['ts_us']
    n = len(ts)
    stride = max(1, math.ceil(n / max_rows)) if max_rows else 1
    rows = np.arange(0, n, stride)
    X, names = build_matrix(d, rows=rows)
    mid_ff = mid_ticks(d, ffill=True)
    Y = np.column_stack([future_mid_change(ts, mid_ff, h * 1000, latency_us)[rows] for h in horizons_ms])
    keep = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    return {
        'X': X[keep],
        'Y': Y[keep].astype(np.float32),
        'ts_us': ts[rows][keep],
        'row': rows[keep],
        'chunk': (rows[keep] * n_chunks // n).astype(np.int8),
        'feature_names': np.array(names),
        'horizons_ms': np.array(horizons_ms),
        'stride': np.array(stride),
    }


# ---------------------------------------------------------------- statistics


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation in float64, NaN when either side is constant."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float(x @ x) * float(y @ y))
    return float(x @ y) / denom if denom > 0 else float('nan')


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (average ranks for ties)."""
    return pearson(rankdata(x), rankdata(y))


# ---------------------------------------------------------------- models


@dataclass
class LinearModel:
    """Winsorize -> standardize -> ridge, in the exported dict's terms."""

    names: list[str]
    lo: np.ndarray
    hi: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    horizon_ms: int = 0
    meta: dict[str, object] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the label for each row of X."""
        z = (np.clip(X, self.lo, self.hi) - self.mean) / self.scale
        return z @ self.coef + self.intercept


def fit_ridge(
    X: np.ndarray, y: np.ndarray, names: list[str], alpha: float = 10.0, clip_q: float = 0.001, block: int = 500_000
) -> LinearModel:
    """Fit ridge on winsorized, standardized features (closed form from float64 moments, in row blocks)."""
    n, p = X.shape
    sample = X[:: max(1, n // 500_000)].astype(np.float64)  # clip bounds from <= ~1M evenly spaced rows
    lo = np.quantile(sample, clip_q, axis=0)
    hi = np.quantile(sample, 1 - clip_q, axis=0)
    s1 = np.zeros(p)
    s2 = np.zeros((p, p))
    sxy = np.zeros(p)
    for start in range(0, n, block):
        xb = np.clip(X[start : start + block].astype(np.float64), lo, hi)
        s1 += xb.sum(axis=0)
        s2 += xb.T @ xb
        sxy += xb.T @ y[start : start + block]
    mean = s1 / n
    y_mean = float(y.mean())
    cov = s2 / n - np.outer(mean, mean)
    scale = np.sqrt(np.clip(np.diag(cov), 0, None))
    scale[scale <= 1e-12] = 1.0
    gram = n * cov / np.outer(scale, scale) + alpha * np.eye(p)  # Z'Z for Z = (clip(X) - mean) / scale
    zy = (sxy - mean * n * y_mean) / scale  # Z'(y - y_mean)
    coef = np.linalg.solve(gram, zy)
    return LinearModel(names, lo, hi, mean, scale, coef, y_mean)


def fit_hgb(X: np.ndarray, y: np.ndarray, max_rows: int = 300_000, seed: int = 0):  # noqa: ANN201
    """Fit a HistGradientBoostingRegressor on at most `max_rows` evenly strided rows."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    rows = np.arange(0, len(X), max(1, math.ceil(len(X) / max_rows)))
    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=500,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(X[rows], y[rows])
    return model


def oos_stats(pred: np.ndarray, y: np.ndarray, train_pred: np.ndarray) -> dict[str, float]:
    """Out-of-sample IC, R^2 vs a zero forecast, and the realised edge when the signal is strong.

    edge10 / edge1: mean of sign(pred) * y (ticks) over test rows whose |pred| is above the
    90th / 99th percentile of |pred| on the TRAINING rows.
    """
    stats = {
        'ic': pearson(pred, y),
        'rank_ic': spearman(pred, y),
        'r2': 1 - float(((y - pred) ** 2).sum()) / max(float((y**2).sum()), 1e-12),
    }
    for label, q in (('edge10', 0.9), ('edge1', 0.99)):
        threshold = np.quantile(np.abs(train_pred), q)
        strong = np.abs(pred) >= threshold
        stats[label] = float((np.sign(pred[strong]) * y[strong]).mean()) if strong.any() else float('nan')
        stats[f'n_{label}'] = float(strong.mean())
    return stats


@dataclass
class FoldResult:
    """One walk-forward fold: which chunks, sizes and OOS stats per model."""

    test_chunk: int
    n_train: int
    n_test: int
    stats: dict[str, dict[str, float]]
    ridge: LinearModel


def walk_forward(
    X: np.ndarray,
    y: np.ndarray,
    ts_us: np.ndarray,
    chunk: np.ndarray,
    names: list[str],
    horizon_us: int,
    latency_us: int = LATENCY_US,
    alpha: float = 10.0,
    use_hgb: bool = True,
    train_chunks: int | None = None,
    verbose: bool = True,
) -> tuple[list[FoldResult], object | None]:
    """Expanding (or rolling, `train_chunks`) walk-forward over chunk ids; returns folds and the last HGB."""
    folds: list[FoldResult] = []
    last_hgb = None
    n_chunks = int(chunk.max()) + 1
    for k in range(1, n_chunks):
        test = chunk == k
        if not test.any():
            continue
        test_start = ts_us[test].min()
        lo_chunk = 0 if train_chunks is None else max(0, k - train_chunks)
        train = (chunk >= lo_chunk) & (chunk < k) & (ts_us + latency_us + horizon_us < test_start)
        if train.sum() < 1000:
            continue
        train_rows = np.flatnonzero(train)
        some_train = train_rows[:: max(1, len(train_rows) // 200_000)]  # for the edge thresholds
        X_train, y_train = X[train_rows], y[train_rows]
        ridge = fit_ridge(X_train, y_train, names, alpha)
        stats = {'ridge': oos_stats(ridge.predict(X[test]), y[test], ridge.predict(X[some_train]))}
        if use_hgb:
            hgb = fit_hgb(X_train, y_train)
            stats['hgb'] = oos_stats(hgb.predict(X[test]), y[test], hgb.predict(X[some_train]))
            last_hgb = hgb
        del X_train, y_train
        folds.append(FoldResult(k, int(train.sum()), int(test.sum()), stats, ridge))
        if verbose:
            line = ' '.join(f"{m}: ic={s['ic']:+.4f} rank={s['rank_ic']:+.4f} r2={s['r2']:+.5f}" for m, s in stats.items())
            print(f'  fold {k}: train {train.sum():,} test {test.sum():,}  {line}')
    return folds, last_hgb


def hgb_importance(model: object, X: np.ndarray, y: np.ndarray, names: list[str], max_rows: int = 100_000) -> dict[str, float]:
    """Permutation importance (drop in Pearson IC) of each feature on at most `max_rows` rows."""
    from sklearn.inspection import permutation_importance

    stride = max(1, math.ceil(len(X) / max_rows))
    result = permutation_importance(
        model, X[::stride], y[::stride], scoring=lambda est, xs, ys: pearson(est.predict(xs), ys), n_repeats=3, random_state=0
    )
    return dict(zip(names, result.importances_mean.tolist(), strict=True))


# ---------------------------------------------------------------- exports


def export_linear(model: LinearModel, digits: int = 6, print_it: bool = True) -> dict[str, object]:
    """Return (and print) a plain dict for `predict_linear`, ready to paste into a strategy."""

    def r(x: float) -> float:
        return float(f'{x:.{digits}g}')

    out = {
        'intercept': r(model.intercept),
        'coef': {n: r(c) for n, c in zip(model.names, model.coef, strict=True)},
        'mean': {n: r(v) for n, v in zip(model.names, model.mean, strict=True)},
        'scale': {n: r(v) for n, v in zip(model.names, model.scale, strict=True)},
        'lo': {n: r(v) for n, v in zip(model.names, model.lo, strict=True)},
        'hi': {n: r(v) for n, v in zip(model.names, model.hi, strict=True)},
        'horizon_ms': model.horizon_ms,
        'latency_ms': LATENCY_US // 1000,
        **model.meta,
    }
    if print_it:
        print('LINEAR_MODEL = {')
        for key, value in out.items():
            if isinstance(value, dict):
                print(f'    {key!r}: {{')
                for name, number in value.items():
                    print(f'        {name!r}: {number!r},')
                print('    },')
            else:
                print(f'    {key!r}: {value!r},')
        print('}')
    return out


def predict_linear(model: dict, x: dict[str, float]) -> float:
    """Evaluate an `export_linear` dict on one event's features (pure Python, paste into a strategy)."""
    total = model['intercept']
    lo, hi, mean, scale = model['lo'], model['hi'], model['mean'], model['scale']
    for name, coef in model['coef'].items():
        value = x[name]
        if value < lo[name]:
            value = lo[name]
        elif value > hi[name]:
            value = hi[name]
        total += coef * (value - mean[name]) / scale[name]
    return total


def export_dataset(
    d: dict[str, np.ndarray],
    out: Path,
    horizon_ms: int = 1000,
    horizons_ms: tuple[int, ...] = (10, 100, 1000, 10000),
    max_rows: int | None = 3_000_000,
) -> Path:
    """Write a compressed npz for training a small model elsewhere (e.g. `research/train_mlp.py` on Colab).

    Keys: X (float32, rows x features), y (label at `horizon_ms`), Y (one column per `horizons_ms`),
    ts_us, row, chunk, feature_names, horizons_ms, y_horizon_ms, latency_ms. Rows are in time order:
    split by time (e.g. chunks 0-5 train, 6 validate, 7 test), never shuffle across it, standardize
    with training statistics, and paste the weights back for `mlp_predict`.
    """
    horizons_ms = tuple(sorted({*horizons_ms, horizon_ms}))
    data = build_dataset(d, horizons_ms, max_rows=max_rows)
    data['y'] = data['Y'][:, horizons_ms.index(horizon_ms)]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data, y_horizon_ms=np.array(horizon_ms), latency_ms=np.array(LATENCY_US // 1000))
    print(f'dataset: X {data["X"].shape}, y at {horizon_ms} ms, Y {data["Y"].shape} -> {out} ({out.stat().st_size / 1e6:.1f} MB)')
    return out


def mlp_predict(params: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """Forward pass of a small ReLU MLP: params mean, scale, W1, b1, ..., Wk, bk (W as (in, out)).

    x is a feature vector (or rows) in `feature_names` order; it is standardized with `mean` / `scale`.
    """
    h = (x - params['mean']) / params['scale']
    layer = 1
    while f'W{layer + 1}' in params:
        h = np.maximum(h @ params[f'W{layer}'] + params[f'b{layer}'], 0.0)
        layer += 1
    return h @ params[f'W{layer}'] + params[f'b{layer}']


# ---------------------------------------------------------------- online (per-event) features


class OnlineFeatures:
    """Pure-Python, O(1)-per-event version of `build_matrix` for use inside a strategy.

    Call `update(snapshot, trades)` from `on_order_book_change`; it returns the feature dict (None
    while the book is one-sided). Same names and definitions as the vectorized code.
    """

    def __init__(
        self, flow_events: tuple[int, ...] = FLOW_EVENTS, flow_ms: tuple[int, ...] = FLOW_MS, ret_ms: tuple[int, ...] = RET_MS
    ) -> None:
        self.flow_events = flow_events
        self.flow_ms = flow_ms
        self.ret_ms = ret_ms
        self._ev_hist: deque[float] = deque(maxlen=max(flow_events))
        self._ev_sums = [0.0] * len(flow_events)
        self._flow_windows = [(t * 1000, deque(), [0.0]) for t in flow_ms]  # (window, deque of (ts, v), [sum])
        self._fills: deque[tuple[int, int]] = deque()
        self._fills_sum = 0
        self._events: deque[int] = deque()
        self._mid_hist = [(t * 1000, deque()) for t in ret_ms]  # (window, deque of (ts, mid))
        self._mid_ff = math.nan
        self._last_dmid = 0.0
        self._last_change_us: int | None = None
        self._last_trade_px = 0

    def update(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> dict[str, float] | None:
        """Advance by one event and return its features (None when the book is one-sided)."""
        t = snapshot.timestamp_us
        if self._last_change_us is None:
            self._last_change_us = t
        flow = 0.0
        for trade in trades:
            flow += trade.quantity if trade.book_side == 'ask' else -trade.quantity
            self._last_trade_px = trade.price

        # event-count windows: sums over the last N values of a shared history
        hist = self._ev_hist
        for k, n in enumerate(self.flow_events):
            self._ev_sums[k] += flow
            if len(hist) >= n:
                self._ev_sums[k] -= hist[-n]
        hist.append(flow)

        # time windows (t - T, t]
        for window, items, total in self._flow_windows:
            if flow:
                items.append((t, flow))
                total[0] += flow
            while items and items[0][0] <= t - window:
                total[0] -= items.popleft()[1]
        if trades:
            self._fills.append((t, len(trades)))
            self._fills_sum += len(trades)
        while self._fills and self._fills[0][0] <= t - 1_000_000:
            self._fills_sum -= self._fills.popleft()[1]
        self._events.append(t)
        while self._events[0] <= t - 1_000_000:
            self._events.popleft()

        bid_px, bid_n, bq = aggregate_views(snapshot.bids)
        ask_px, ask_n, aq = aggregate_views(snapshot.asks)
        two_sided = bid_px > 0 and ask_px > 0
        if two_sided:
            mid = (bid_px + ask_px) / 2
            if mid != self._mid_ff:
                if self._mid_ff == self._mid_ff:  # not NaN: a real change
                    self._last_dmid = mid - self._mid_ff
                    self._last_change_us = t
                self._mid_ff = mid

        rets = []
        for window, items in self._mid_hist:
            items.append((t, self._mid_ff))
            while len(items) >= 2 and items[1][0] <= t - window:
                items.popleft()
            rets.append(self._mid_ff - items[0][1] if items[0][0] <= t - window else math.nan)
        if not two_sided:
            return None

        def imb(b: float, a: float) -> float:
            return (b - a) / (b + a) if b + a > 0 else 0.0

        spread = float(ask_px - bid_px)
        imb1 = imb(bq[0], aq[0])
        out = {
            'spread': spread,
            'imb1': imb1,
            'imb2': imb(bq[0] + bq[1], aq[0] + aq[1]),
            'imb3': imb(bq[0] + bq[1] + bq[2], aq[0] + aq[1] + aq[2]),
            'imb5': imb(sum(bq), sum(aq)),
            'micro': spread * imb1 / 2,
            'cnt_imb1': imb(bid_n, ask_n),
            'depth1_log': math.log1p(bq[0] + aq[0]),
        }
        for n, total in zip(self.flow_events, self._ev_sums, strict=True):
            out[f'flow_ev{n}'] = total
        for t_ms, (_, _, total) in zip(self.flow_ms, self._flow_windows, strict=True):
            out[f'flow_ms{t_ms}'] = total[0]
        out['ntr_ms1000'] = float(self._fills_sum)
        out['nev_ms1000'] = float(len(self._events))
        for t_ms, value in zip(self.ret_ms, rets, strict=True):
            out[f'ret_ms{t_ms}'] = value
        out['last_dmid'] = self._last_dmid
        out['dt_mid_log'] = math.log1p((t - self._last_change_us) / 1000)
        out['ltp_dev'] = self._last_trade_px - mid if self._last_trade_px > 0 else 0.0
        return out


def check_online_parity(csv_path: Path, max_events: int | None = None, tol: float = 1e-4) -> int:
    """Replay `csv_path` with `OnlineFeatures` and compare every row with `build_matrix`; return #rows checked."""
    from simulator.simulator import ExchangeSimulator, SimulationConfig
    from simulator.strategy import BaseStrategy

    rows: list[dict[str, float] | None] = []

    class _Probe(BaseStrategy):
        def __init__(self) -> None:
            super().__init__()
            self.features = OnlineFeatures()

        def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
            if max_events is None or len(rows) < max_events:
                rows.append(self.features.update(snapshot, trades))

    ExchangeSimulator(_Probe(), SimulationConfig()).run(csv_path)
    X, names = build_matrix(load_features(ensure_features(csv_path)))
    checked = 0
    for i, row in enumerate(rows):
        offline = X[i]
        if row is None:
            if np.isfinite(offline[0]):
                raise AssertionError(f'event {i}: online says one-sided, offline has a book')
            continue
        online = np.array([row[n] for n in names], dtype=np.float64)
        both_nan = np.isnan(online) & np.isnan(offline)
        bad = ~both_nan & ~(np.abs(online - offline) <= tol * np.maximum(1.0, np.abs(offline)))
        if bad.any():
            diffs = {names[k]: (online[k], float(offline[k])) for k in np.flatnonzero(bad)}
            raise AssertionError(f'event {i}: online != offline {diffs}')
        checked += 1
    return checked


# ---------------------------------------------------------------- report / CLI


def _fmt(x: float, spec: str = '+.4f') -> str:
    return 'nan' if x != x else format(x, spec)


def write_report(
    path: Path,
    source: str,
    horizon_ms: int,
    names: list[str],
    folds: list[FoldResult],
    final: LinearModel,
    importance: dict[str, float] | None,
    n_rows: int,
    stride: int,
) -> None:
    """Markdown summary of one walk-forward run."""
    lines = [
        f'# Signals - `{source}`, horizon {horizon_ms} ms',
        '',
        f'Label: mid change from t + {LATENCY_US // 1000} ms to t + {LATENCY_US // 1000} ms + {horizon_ms} ms (ticks). '
        f'{n_rows:,} rows (tape events with stride {stride}). Walk-forward over equal-event chunks: train on the '
        'chunks before, test on the next, training rows whose label reaches into the test chunk purged.',
        'ic/rank = Pearson/Spearman of prediction vs label on the test chunk; r2 vs a zero forecast; '
        'edge10/edge1 = mean sign(pred)*label (ticks) when |pred| is in the top 10% / 1% (train thresholds).',
        '',
        '| fold | model | n_test | ic | rank_ic | r2 | edge10 | edge1 |',
        '|---:|---|---:|---:|---:|---:|---:|---:|',
    ]
    for fold in folds:
        for model, s in fold.stats.items():
            lines.append(
                f"| {fold.test_chunk} | {model} | {fold.n_test:,} | {_fmt(s['ic'])} | {_fmt(s['rank_ic'])} | "
                f"{_fmt(s['r2'], '+.5f')} | {_fmt(s['edge10'], '+.3f')} | {_fmt(s['edge1'], '+.3f')} |"
            )
    for model in folds[0].stats if folds else []:
        mean = {key: float(np.nanmean([f.stats[model][key] for f in folds])) for key in ('ic', 'rank_ic', 'r2', 'edge10', 'edge1')}
        lines.append(
            f"| **mean** | {model} | | {_fmt(mean['ic'])} | {_fmt(mean['rank_ic'])} | {_fmt(mean['r2'], '+.5f')} | "
            f"{_fmt(mean['edge10'], '+.3f')} | {_fmt(mean['edge1'], '+.3f')} |"
        )
    lines += ['', '## Ridge coefficients (standardized) per fold', '', '| feature | ' + ' | '.join(str(f.test_chunk) for f in folds) + ' | final |']
    lines.append('|---|' + '---:|' * (len(folds) + 1))
    for k, name in enumerate(names):
        lines.append(f'| {name} | ' + ' | '.join(_fmt(f.ridge.coef[k], '+.4f') for f in folds) + f' | {_fmt(final.coef[k])} |')
    if importance:
        lines += ['', '## HGB permutation importance (drop in IC, last fold)', '', '| feature | importance |', '|---|---:|']
        for name, value in sorted(importance.items(), key=lambda kv: -kv[1]):
            lines.append(f'| {name} | {_fmt(value, "+.5f")} |')
    lines += ['', '## Feature definitions', '']
    lines += [f'- `{k}`: {v}' for k, v in FEATURE_DOC.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv: list[str] | None = None) -> None:
    """Walk-forward signal research on one dataset."""
    parser = argparse.ArgumentParser(description='Walk-forward short-horizon signal research')
    parser.add_argument('data', type=Path, help='features npz (or a tape CSV, built into one if needed)')
    parser.add_argument('--horizon-ms', type=int, default=1000, help='label horizon after the 5 ms latency')
    parser.add_argument('--chunks', type=int, default=8, help='equal-event time chunks for walk-forward')
    parser.add_argument('--train-chunks', type=int, default=None, help='rolling window of N chunks (default expanding)')
    parser.add_argument('--alpha', type=float, default=10.0, help='ridge penalty on standardized features')
    parser.add_argument('--max-rows', type=int, default=3_000_000, help='stride the events down to about this many')
    parser.add_argument('--no-hgb', action='store_true', help='skip the gradient-boosting check')
    parser.add_argument('--features', type=str, default=None, help='comma-separated subset of features')
    parser.add_argument('--export-linear', action='store_true', help='print the final ridge as a paste-ready dict')
    parser.add_argument('--export-dataset', type=Path, default=None, help='write X/Y/ts npz for MLP training')
    parser.add_argument('--parity', action='store_true', help='check OnlineFeatures against build_matrix (CSV only)')
    parser.add_argument('--out', type=Path, default=None, help='report path (default research/reports/<stem>_signals_h<H>ms.md)')
    args = parser.parse_args(argv)

    start = time.perf_counter()
    if args.data.suffix == '.csv':
        if args.parity:
            print(f'online parity: {check_online_parity(args.data):,} rows match')
        npz_path = ensure_features(args.data)
    else:
        npz_path = args.data
    d = load_features(npz_path)
    source = str(d.get('source', npz_path))
    if args.export_dataset:
        export_dataset(d, args.export_dataset, horizon_ms=args.horizon_ms, max_rows=args.max_rows)

    data = build_dataset(d, (args.horizon_ms,), max_rows=args.max_rows, n_chunks=args.chunks)
    names = [str(n) for n in data['feature_names']]
    X, y = data['X'], data['Y'][:, 0].astype(np.float64)
    if args.features:
        keep = [names.index(n) for n in args.features.split(',')]
        X, names = X[:, keep], [names[k] for k in keep]
    print(f'{npz_path}: {len(y):,} rows x {len(names)} features (stride {int(data["stride"])}), '
          f'label std {y.std():.4f} ticks, {np.mean(y != 0):.1%} non-zero  [{time.perf_counter() - start:.1f}s]')

    horizon_us = args.horizon_ms * 1000
    folds, hgb = walk_forward(
        X, y, data['ts_us'], data['chunk'], names, horizon_us,
        alpha=args.alpha, use_hgb=not args.no_hgb, train_chunks=args.train_chunks,
    )  # fmt: skip
    importance = None
    if hgb is not None:
        last = data['chunk'] == data['chunk'].max()
        importance = hgb_importance(hgb, X[last], y[last], names)

    final = fit_ridge(X, y, names, args.alpha)
    final.horizon_ms = args.horizon_ms
    final.meta = {'source': source, 'rows': int(len(y))}
    for model in folds[0].stats if folds else []:
        ics = [f.stats[model]['ic'] for f in folds]
        print(f'{model}: mean OOS ic {np.nanmean(ics):+.4f}, positive in {sum(ic > 0 for ic in ics)}/{len(ics)} folds')
    report = args.out or REPORTS_DIR / f'{Path(source).stem}_signals_h{args.horizon_ms}ms.md'
    write_report(report, source, args.horizon_ms, names, folds, final, importance, len(y), int(data['stride']))
    print(f'report -> {report}  [{time.perf_counter() - start:.1f}s]')
    if args.export_linear:
        export_linear(final)


if __name__ == '__main__':
    main()
