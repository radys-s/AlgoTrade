"""exp_r2_take (radys, R2): mm_r2_fair + hidden-fair TAKER overlay that also unloads pinned inventory.

mm_r2_fair sits at |I| = 100 almost all the time, so the growing side has no capacity and the unload side waits
passively. Here, when the fair signal is strong, we cross the touch in its direction:
  * reducing trade (sig against our inventory, e.g. long and sig < -take_exit_thr): IOC at the touch for up to
    take_size lots -> frees capacity before the move, the maker re-buys lower.
  * growing trade (|sig| > take_entry_thr in the direction that grows |I|, only while there is room): IOC entry.
If the side has no free room because our own resting quote holds it, that quote is pulled first (the IOC goes out
on a later row once CANCELED frees OUT). One IOC in flight at a time, take_cool_us between sends.
take_exit_thr 0 and take_entry_thr 0 = off (identical to mm_r2_fair). The invariant gate in _send is untouched.
"""

import importlib.util
import sys
from pathlib import Path

_base_path = Path(__file__).with_name('mm_r2_fair.py')
_spec = importlib.util.spec_from_file_location('mm_r2_fair_base', _base_path)
_base = importlib.util.module_from_spec(_spec)
sys.modules.setdefault('mm_r2_fair_base', _base)
_spec.loader.exec_module(_base)

_NEVER_US = -(1 << 62)


class MarketMaker(_base.MarketMaker):
    PARAMS = dict(_base.MarketMaker.PARAMS)
    PARAMS.update({
        'take_exit_thr': 0.0,
        'take_entry_thr': 0.0,
        'take_size': 50,
        'take_cool_us': 20000,
        'take_max_spread': 1,
    })

    def __init__(self) -> None:
        super().__init__()
        p = self.PARAMS
        self.take_exit_thr = float(p['take_exit_thr'])
        self.take_entry_thr = float(p['take_entry_thr'])
        self.take_size = max(1, int(p['take_size']))
        self.take_cool_us = int(p['take_cool_us'])
        self.take_max_spread = int(p['take_max_spread'])
        self.take_id = None
        self.take_last = _NEVER_US
        self.n_takes = 0

    def on_order_book_change(self, snapshot, trades) -> None:
        super().on_order_book_change(snapshot, trades)
        if self.take_exit_thr <= 0.0 and self.take_entry_thr <= 0.0:
            return
        if self.take_id is not None:
            return
        now = snapshot.timestamp_us
        if now - self.take_last < self.take_cool_us:
            return
        bb = ba = -1
        for v in snapshot.bids:
            if v.owner == 'market':
                bb = v.price
                break
        for v in snapshot.asks:
            if v.owner == 'market':
                ba = v.price
                break
        if bb < 0 or ba < 0 or ba - bb > self.take_max_spread:
            return
        sig = self.fair_sig
        inv = self.inv
        if sig > 0:
            k, sign, side, px = 0, 1, _base._BUY, ba
        else:
            k, sign, side, px = 1, -1, _base._SELL, bb
        reducing = sign * inv < 0
        thr = self.take_exit_thr if reducing else self.take_entry_thr
        if thr <= 0.0 or abs(sig) <= thr:
            return
        want = min(self.take_size, abs(inv)) if reducing else self.take_size
        if want <= 0:
            return
        free = self.max_pos - sign * inv - self.out[k]
        if free < want:
            # our own quote on that side holds the room: pull it, the IOC goes out once CANCELED frees OUT
            if self.act_qty[k] > 0:
                self._pull(k)
                self._pull(k + 2)
            if free <= 0:
                return
            want = free
        oid = self._send(k, sign, side, px, want, True, k)
        if oid is not None:
            self.take_id = oid
            self.take_last = now
            self.n_takes += 1

    def _forget(self, order) -> None:
        super()._forget(order)
        if order.order_id == self.take_id:
            self.take_id = None
