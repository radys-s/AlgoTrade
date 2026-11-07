"""Per-event book and trade features from one replay of a tape through the official simulator.

    uv run python -m research.features data/round1.csv            # -> data/features/round1.npz
    uv run python -m research.features data/round1.csv --out x.npz --no-compress

A non-trading recorder strategy is attached to `ExchangeSimulator`, so the book is exactly the
one a strategy sees (price-time matching, IOC sweeps, `public_depth` = 20 ticks visibility).
`on_order_book_change` fires once per tape row, so row `i` of every array below is the state
right AFTER tape row `i` has been applied (and the trades it caused). Nothing here looks ahead.

Columns of the npz (n = number of tape rows; prices in ticks, 0 = that side of the book is empty):

    ts_us          int64  (n,)   event time in microseconds (the simulator's own rounding of `Time`)
    bid_px         int32  (n,)   best bid price after the event (0 = no bids)
    ask_px         int32  (n,)   best ask price after the event (0 = no asks)
    bid_qty        int32  (n,5)  aggregated quantity of the best 5 distinct bid prices, best first
    ask_qty        int32  (n,5)  same for asks; a level that does not exist (or is not visible) is 0
    bid_cnt1       int32  (n,)   number of resting orders at the best bid
    ask_cnt1       int32  (n,)   number of resting orders at the best ask
    trade_vol      int32  (n,)   signed volume traded by this event: + buyer-initiated
                                 (taker lifted the ask, `book_side == 'ask'`), - seller-initiated
    trade_cnt      int32  (n,)   number of fills (maker orders matched) caused by this event
    last_trade_px  int32  (n,)   last trade price up to and including this event (0 before the first)
    ev_type        int8   (n,)   tape row type: 0 = insert limit (G), 1 = insert IOC (F), 2 = cancel
    ev_side        int8   (n,)   +1 = bid/buy, -1 = ask/sell; for a cancel the side of the cancelled
                                 order (from its latest insert), 0 if the id was never inserted
    ev_qty         int32  (n,)   inserted volume; for a cancel the cancelled order's original volume
    ev_px          int32  (n,)   inserted limit price; for a cancel the cancelled order's price

Scalars: `source` (CSV path), `public_depth`, `build_seconds`.

Levels are distinct prices (not ticks): level 2 may be several ticks behind level 1, and only
prices within `public_depth` ticks of the best count, exactly as in the strategy's snapshot.

Speed: summing every visible order on every event is O(book depth) per event and was 3-4x the
cost of the replay itself on deep books. The recorder therefore reads the simulator's book
(read-only) and caches (quantity, order count) per price level; a level is re-summed only after
an event touched it (the tape row's own price, or a trade price). Every `check_every`-th event
the cached result is compared with a plain scan of the public snapshot, and a mismatch raises.

`load_features(path)` returns the arrays as a dict; `mid_ticks(d)` the mid (NaN when one-sided,
or forward-filled), `ensure_features(csv)` builds the npz when it is missing or stale.
"""

from __future__ import annotations

import argparse
import time
from array import array
from pathlib import Path

import numpy as np
import pandas as pd

from simulator.main import (
    MAX_POSITION,
    MAX_POTENTIAL_POSITION,
    ORDER_ACCEPT_DELAY_MS,
    POSITION_BREACH_PENALTY_PER_SECOND,
)
from simulator.models import BookOrderView, OrderBookSnapshot, Trade
from simulator.simulator import ExchangeSimulator, OrderBook, SimulationConfig
from simulator.strategy import BaseStrategy

FEATURES_DIR = Path('data/features')
N_LEVELS = 5
EV_LIMIT, EV_IOC, EV_CANCEL = 0, 1, 2
_ZERO_LEVELS = bytes(4 * N_LEVELS)

SideAgg = tuple[int, int, array]  # (best price, order count at level 1, array('i') of 5 level quantities)


def aggregate_views(views: list[BookOrderView]) -> SideAgg:
    """Aggregate one side of a public snapshot (best first), stopping at the 6th distinct price.

    The reference implementation; a strategy can use the same loop on `snapshot.bids` / `.asks`.
    """
    q = array('i', _ZERO_LEVELS)
    if not views:
        return 0, 0, q
    best = price = views[0].price
    level = 0
    count1 = 0
    for view in views:
        if view.price != price:
            level += 1
            if level == N_LEVELS:
                break
            price = view.price
        q[level] += view.quantity
        if level == 0:
            count1 += 1
    return best, count1, q


class FeatureRecorder(BaseStrategy):
    """Non-trading strategy that writes one row of book/trade state per market event.

    Columns live in preallocated `array.array` buffers (cheap item assignment, zero-copy to
    numpy). `ev_side` / `ev_px` (the tape rows, aligned by index) tell which levels each event
    may have touched; see the module docstring.
    """

    def __init__(
        self,
        n_events: int,
        ev_side: np.ndarray,
        ev_px: np.ndarray,
        public_depth: int = 20,
        check_every: int = 1009,
    ) -> None:
        super().__init__()
        self.n = n_events
        self.i = 0
        self.book: OrderBook | None = None
        self.depth = public_depth
        self.check_every = check_every
        self.ev_side = array('b', np.ascontiguousarray(ev_side, dtype=np.int8).tobytes())
        self.ev_px = array('i', np.ascontiguousarray(ev_px, dtype=np.int32).tobytes())
        self.ts_us = array('q', bytes(8 * n_events))
        self.bid_px = array('i', bytes(4 * n_events))
        self.ask_px = array('i', bytes(4 * n_events))
        self.bid_qty = array('i', bytes(4 * N_LEVELS * n_events))
        self.ask_qty = array('i', bytes(4 * N_LEVELS * n_events))
        self.bid_cnt1 = array('i', bytes(4 * n_events))
        self.ask_cnt1 = array('i', bytes(4 * n_events))
        self.trade_vol = array('i', bytes(4 * n_events))
        self.trade_cnt = array('i', bytes(4 * n_events))
        self.last_trade_px = array('i', bytes(4 * n_events))
        self._bid_version = -1
        self._ask_version = -1
        self._bid_levels: dict[int, tuple[int, int]] = {}  # price -> (quantity, order count)
        self._ask_levels: dict[int, tuple[int, int]] = {}
        self._bid_agg: SideAgg = aggregate_views([])
        self._ask_agg: SideAgg = aggregate_views([])
        self._last_trade_px = 0
        self.checks = 0

    def _top_levels(self, is_bid: bool) -> SideAgg:
        """Aggregate the best 5 visible prices of one side from the book, via the level cache."""
        assert self.book is not None
        if is_bid:
            prices, levels, cache = self.book._bid_prices, self.book._bids, self._bid_levels
        else:
            prices, levels, cache = self.book._ask_prices, self.book._asks, self._ask_levels
        q = array('i', _ZERO_LEVELS)
        n = len(prices)
        if n == 0:
            return 0, 0, q
        best = prices[-1] if is_bid else prices[0]
        count1 = 0
        for k in range(min(N_LEVELS, n)):
            price = prices[n - 1 - k] if is_bid else prices[k]
            if abs(price - best) >= self.depth:
                break
            cached = cache.get(price)
            if cached is None:
                queue = levels[price].queue
                total = 0
                for order in queue:
                    total += order.remaining_quantity
                cached = cache[price] = (total, len(queue))
            q[k] = cached[0]
            if k == 0:
                count1 = cached[1]
        return best, count1, q

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        """Record the book after this event and the trades it caused."""
        i = self.i
        if i >= self.n:
            raise RuntimeError('More market events than counted rows in the tape')
        self.i = i + 1
        book = self.book
        if book is None:  # no book access: scan the snapshot (slow on deep books, always exact)
            bid_agg = aggregate_views(snapshot.bids)
            ask_agg = aggregate_views(snapshot.asks)
        else:
            bid_changed = book.bid_version != self._bid_version
            ask_changed = book.ask_version != self._ask_version
            if bid_changed or ask_changed:
                side = self.ev_side[i]
                if side > 0:
                    self._bid_levels.pop(self.ev_px[i], None)
                elif side < 0:
                    self._ask_levels.pop(self.ev_px[i], None)
                else:  # cancel of an id we could not resolve: forget everything
                    self._bid_levels.clear()
                    self._ask_levels.clear()
                for trade in trades:
                    (self._bid_levels if trade.book_side == 'bid' else self._ask_levels).pop(trade.price, None)
                if bid_changed:
                    self._bid_version = book.bid_version
                    self._bid_agg = self._top_levels(True)
                if ask_changed:
                    self._ask_version = book.ask_version
                    self._ask_agg = self._top_levels(False)
            bid_agg = self._bid_agg
            ask_agg = self._ask_agg
            if i % self.check_every == 0:
                self._check(snapshot, bid_agg, ask_agg)

        self.ts_us[i] = snapshot.timestamp_us
        self.bid_px[i] = bid_agg[0]
        self.ask_px[i] = ask_agg[0]
        k = i * N_LEVELS
        self.bid_qty[k : k + N_LEVELS] = bid_agg[2]
        self.ask_qty[k : k + N_LEVELS] = ask_agg[2]
        self.bid_cnt1[i] = bid_agg[1]
        self.ask_cnt1[i] = ask_agg[1]
        if trades:
            signed = 0
            for trade in trades:
                signed += trade.quantity if trade.book_side == 'ask' else -trade.quantity
            self.trade_vol[i] = signed
            self.trade_cnt[i] = len(trades)
            self._last_trade_px = trades[-1].price
        self.last_trade_px[i] = self._last_trade_px

    def _check(self, snapshot: OrderBookSnapshot, bid_agg: SideAgg, ask_agg: SideAgg) -> None:
        """Compare the cached aggregation with a plain scan of the public snapshot."""
        self.checks += 1
        for name, fast, views in (('bid', bid_agg, snapshot.bids), ('ask', ask_agg, snapshot.asks)):
            slow = aggregate_views(views)
            if fast[0] != slow[0] or fast[1] != slow[1] or fast[2] != slow[2]:
                raise RuntimeError(f'Level cache mismatch on the {name} side at event {self.i - 1}: {fast} != {slow}')

    def arrays(self) -> dict[str, np.ndarray]:
        """Return the recorded columns as numpy arrays (zero-copy views of the buffers)."""
        n = self.i
        return {
            'ts_us': np.frombuffer(self.ts_us, dtype=np.int64)[:n],
            'bid_px': np.frombuffer(self.bid_px, dtype=np.int32)[:n],
            'ask_px': np.frombuffer(self.ask_px, dtype=np.int32)[:n],
            'bid_qty': np.frombuffer(self.bid_qty, dtype=np.int32).reshape(-1, N_LEVELS)[:n],
            'ask_qty': np.frombuffer(self.ask_qty, dtype=np.int32).reshape(-1, N_LEVELS)[:n],
            'bid_cnt1': np.frombuffer(self.bid_cnt1, dtype=np.int32)[:n],
            'ask_cnt1': np.frombuffer(self.ask_cnt1, dtype=np.int32)[:n],
            'trade_vol': np.frombuffer(self.trade_vol, dtype=np.int32)[:n],
            'trade_cnt': np.frombuffer(self.trade_cnt, dtype=np.int32)[:n],
            'last_trade_px': np.frombuffer(self.last_trade_px, dtype=np.int32)[:n],
        }


def read_tape(csv_path: Path) -> pd.DataFrame:
    """Read the tape CSV with compact dtypes (categories for the text columns).

    `OrderId` is left to inference (int64 for numeric ids, ~8x less memory than strings).
    """
    return pd.read_csv(
        csv_path,
        dtype={
            'Time': np.float64,
            'Operation': 'category',
            'Side': 'category',
            'Volume': np.float64,
            'Price': np.float64,
            'Lifespan': 'category',
        },
        engine='c',
        float_precision='round_trip',  # parse `Time` exactly like the simulator's float()
    )


def tape_timestamps_us(tape: pd.DataFrame) -> np.ndarray:
    """Return `Time` in microseconds, rounded as the simulator does."""
    return np.rint(tape['Time'].to_numpy() * 1_000_000).astype(np.int64)


def cancel_source_rows(tape: pd.DataFrame) -> np.ndarray:
    """For every cancel row, the row of the latest earlier insert with the same OrderId (-1 if none, or not a cancel)."""
    n = len(tape)
    is_insert = (tape['Operation'] == 'Insert').to_numpy()
    codes, _ = pd.factorize(tape['OrderId'])
    # Sorted by (order id, row), a cancel's source is the last insert before it within its id run.
    order = np.lexsort((np.arange(n), codes))
    sorted_insert = is_insert[order]
    last_insert = np.maximum.accumulate(np.where(sorted_insert, np.arange(n), -1))
    cancel_pos = np.flatnonzero(~sorted_insert)
    source_pos = last_insert[cancel_pos]
    ok = source_pos >= 0
    ok[ok] = codes[order[source_pos[ok]]] == codes[order[cancel_pos[ok]]]
    source = np.full(n, -1, dtype=np.int64)
    source[order[cancel_pos[ok]]] = order[source_pos[ok]]
    return source


def tape_event_columns(tape: pd.DataFrame, source: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Return ev_type / ev_side / ev_qty / ev_px per tape row; a cancel gets its order's latest insert."""
    n = len(tape)
    is_insert = (tape['Operation'] == 'Insert').to_numpy()
    if 'Lifespan' in tape:
        is_ioc = (tape['Lifespan'] == 'F').to_numpy()
    else:
        is_ioc = np.zeros(n, dtype=bool)
    ev_type = np.where(is_insert, np.where(is_ioc, EV_IOC, EV_LIMIT), EV_CANCEL).astype(np.int8)

    side = np.select([(tape['Side'] == 'B').to_numpy(), (tape['Side'] == 'A').to_numpy()], [1, -1], 0)
    qty = tape['Volume'].fillna(0).to_numpy(copy=True)
    px = tape['Price'].fillna(0).to_numpy(copy=True)

    if source is None:
        source = cancel_source_rows(tape)
    dst = np.flatnonzero(source >= 0)
    src = source[dst]
    side[dst] = side[src]
    qty[dst] = qty[src]
    px[dst] = px[src]
    unresolved = np.flatnonzero(~is_insert & (source < 0))
    side[unresolved] = 0
    qty[unresolved] = 0
    px[unresolved] = 0
    return {
        'ev_type': ev_type,
        'ev_side': side.astype(np.int8),
        'ev_qty': qty.astype(np.int32),
        'ev_px': px.astype(np.int32),
    }


def build_features(
    csv_path: Path, public_depth: int = 20, check_every: int = 1009, verbose: bool = True
) -> dict[str, np.ndarray]:
    """Replay `csv_path` through the simulator with the recorder and return all npz columns."""
    start = time.perf_counter()
    tape = read_tape(csv_path)
    tape_ts = tape_timestamps_us(tape)
    events = tape_event_columns(tape)
    del tape
    if verbose:
        print(f'tape: {len(tape_ts):,} rows parsed in {time.perf_counter() - start:.1f}s')

    n_rows = ExchangeSimulator.count_backtest_rows(csv_path)
    recorder = FeatureRecorder(n_rows, events['ev_side'], events['ev_px'], public_depth, check_every)
    config = SimulationConfig(
        public_depth=public_depth,
        max_position=MAX_POSITION,
        max_potential_position=MAX_POTENTIAL_POSITION,
        position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND,
        order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS,
    )
    simulator = ExchangeSimulator(strategy=recorder, config=config)
    recorder.book = simulator.book
    start = time.perf_counter()
    simulator.run(csv_path)
    replay_s = time.perf_counter() - start
    if recorder.i != n_rows or len(tape_ts) != n_rows:
        raise RuntimeError(f'Recorded {recorder.i} events, counted {n_rows} rows, parsed {len(tape_ts)}')
    data = recorder.arrays()
    if not np.array_equal(tape_ts, data['ts_us']):
        raise RuntimeError('Tape rows do not line up with the recorded events')
    data.update(events)
    if verbose:
        print(
            f'replay: {n_rows:,} events in {replay_s:.1f}s = {n_rows / max(replay_s, 1e-9):,.0f} events/s '
            f'({recorder.checks:,} cache checks passed)'
        )
    return data


def default_npz_path(csv_path: Path) -> Path:
    """Return data/features/<stem>.npz for a tape CSV."""
    return FEATURES_DIR / f'{csv_path.stem}.npz'


def save_features(data: dict[str, np.ndarray], out: Path, compress: bool = True, **meta: object) -> None:
    """Write the columns (plus scalar metadata) to an npz."""
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {**data, **{key: np.asarray(value) for key, value in meta.items()}}
    (np.savez_compressed if compress else np.savez)(out, **payload)


def load_features(path: Path) -> dict[str, np.ndarray]:
    """Load every array of a features npz into memory."""
    with np.load(path, allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def ensure_features(csv_path: Path, out: Path | None = None) -> Path:
    """Return the npz for `csv_path`, building it first if missing or older than the CSV."""
    out = out or default_npz_path(csv_path)
    if not out.exists() or out.stat().st_mtime < csv_path.stat().st_mtime:
        main([str(csv_path), '--out', str(out)])
    return out


def mid_ticks(d: dict[str, np.ndarray], ffill: bool = False) -> np.ndarray:
    """Return the mid price in ticks (float64).

    Where either side is empty the mid is NaN, or with `ffill` the last two-sided mid (NaN before
    the first one) - the natural "current mid" for a strategy that remembers it.
    """
    mid = (d['bid_px'].astype(np.float64) + d['ask_px'].astype(np.float64)) / 2
    valid = (d['bid_px'] > 0) & (d['ask_px'] > 0)
    if not ffill:
        mid[~valid] = np.nan
        return mid
    last = np.maximum.accumulate(np.where(valid, np.arange(len(mid)), -1))
    out = np.full(len(mid), np.nan)
    out[last >= 0] = mid[last[last >= 0]]
    return out


def main(argv: list[str] | None = None) -> None:
    """Build data/features/<stem>.npz from a tape CSV."""
    parser = argparse.ArgumentParser(description='Per-event book/trade features via one simulator replay')
    parser.add_argument('csv', type=Path, help='tape CSV (Time,Operation,OrderId,Side,Volume,Price,Lifespan)')
    parser.add_argument('--out', type=Path, default=None, help='output npz (default data/features/<stem>.npz)')
    parser.add_argument('--public-depth', type=int, default=20, help='visible ticks per side, as in the game')
    parser.add_argument('--check-every', type=int, default=1009, help='verify the level cache every N events')
    parser.add_argument('--no-compress', action='store_true', help='plain np.savez: bigger file, faster write')
    args = parser.parse_args(argv)

    start = time.perf_counter()
    data = build_features(args.csv, public_depth=args.public_depth, check_every=args.check_every)
    out = args.out or default_npz_path(args.csv)
    write_start = time.perf_counter()
    save_features(
        data,
        out,
        compress=not args.no_compress,
        source=str(args.csv),
        public_depth=args.public_depth,
        build_seconds=time.perf_counter() - start,
    )
    total_s = time.perf_counter() - start
    n = len(data['ts_us'])
    size_mb = out.stat().st_size / 1e6
    print(f'write: {time.perf_counter() - write_start:.1f}s -> {out} ({size_mb:.1f} MB)')
    print(f'total: {total_s:.1f}s = {n / max(total_s, 1e-9):,.0f} events/s end to end')


if __name__ == '__main__':
    main()
