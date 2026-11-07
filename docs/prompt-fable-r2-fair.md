# Prompt for Honza's Claude (Fable): R2 — exploit the hidden fair price (radys, 20:50)

Paste everything below the line into Honza's Claude session.

---

**Situation (20:50, round closes ~00:15).** R2 leaderboard: median best 55k, the 75th percentile jumped to
~**400k** at 20:30 (20 teams). Our best is your `exp_r2_ladder` = 86,333, now baked as `strategies/mm_r2_gate86.py`
and `submission/best_strategy.py` on `main` (gate_thr 4, gate_max_spread 3, gate_rungs 1, park 0; |I| median = 100).
400k is above your own front-of-queue oracle for the gate maker (232k), so queue position alone does not explain the
leaders. They are trading something directional on the 0.1-1 s horizon. We found what.

## The finding (verify in 5 min: `playground/explore_data.py`, cell 3; details `docs/radys-findings.md` §11-13)

The generator places limit orders around a **hidden fair price F**; the visible book lags F. Every new market limit
order is a noisy reading: `ev = price + 1.22` for bids, `price - 1.20` for asks (R2 offsets vs the pre-event mid;
recompute per tape). `sig = time-decayed mean of (ev - mid)`, tau 20-50 ms, limit inserts only (visible to us).

| | corr with mid move t+5ms -> t+100ms | -> 1 s | -> 5 s |
|---|---|---|---|
| R2 | +0.41 | +0.21 | +0.09 |
| R1 | +0.31 | +0.37 | +0.17 |

Book imbalance gives 0.03, trade-flow / past-return signals were break-even in your taker_econ. Offline taker
(buy the ask at t+5 ms, sell the bid at t+500 ms, overlapping samples so optimistic): sig > 1.0 +0.25,
> 1.5 +0.82, > 2.0 +1.37, > 3.0 +2.9 ticks/lot AFTER both spreads. The vertical jumps in the R2 mid chart are where
F jumps and the book follows with a lag: that is the taker's money. Counts of new bids vs asks carry nothing (0.00);
the information is in the PRICES.

## Implementation notes (arena mechanics)

- The snapshot lists individual orders within 20 ticks with `order_id`; market order ids are strictly increasing
  with no gaps. Keep `max_seen_id`; per callback scan each side only within ~3 ticks of the touch and take orders
  with id > max_seen_id (O(new orders), no set differences). Skip `owner == 'user'`.
- Decay in microseconds: `w = exp(-(now - last)/tau)`, `s = s*w + (ev - mid)`, `n = n*w + 1`, `sig = s/n`.
- Everything acts 5 ms later: a pull only avoids fills that happen > 5 ms after the signal. Count inventory from
  fills only; keep |I| + live + in-flight same-side <= 100 by construction (your `_send` invariant).
- `modify_order` only shrinks; growing back = new order at the back of the queue.

## Work plan (strict order, one flag-gated experiment file, default off = gate86 parity)

1. **Insurance:** if `mm_r2_gate86` is not uploaded yet, upload it now.
2. **T2 taker standalone (`strategies/exp_fair_taker.py`)**, to measure the edge in the real engine:
   IOC at the touch when |sig| > thr, qty Q, exit by IOC after hold_us or when sig crosses 0; |I| <= 100.
   Grid thr {1.0, 1.5, 2.0, 3.0} x hold {200, 500, 1000 ms} x Q {10, 30, 60} x tau {20, 50 ms}, R2 and R1.
   Report pnl, fills, pnl/lot, penalty, chunk wins. This sizes the prize.
3. **T3 maker filter on gate86 (`exp_r2_fair.py`)**: bid stays FULL unless sig < -s, then pull it (variant: modify
   down to a 1-3 lot stub); ask stays full unless sig > +s. (Careful: the rule "quote the bid only when sig > s"
   is inverted and kills the maker.) Grid s {0.3, 0.5, 0.8, 1.2}. Your fill decomposition says -0.32/lot adverse
   within 1 s: this targets exactly that.
4. **Combine:** gate86 + filter + taker overlay sharing the 100-lot capacity (taker only with |sig| > the best thr
   from step 2; the maker keeps the rest).
5. **Then structure:** gate on F drift instead of mid drift (cleaner, faster: try windows 5-15 s); re-enable the
   ladder/park WITH the fair filter (rung fills at retreat levels were -0.57/lot; keep a rung only while sig is not
   against it); passive inventory unload when |I| = 100 (it is pinned there half the time).

**Acceptance:** higher R2 pnl than 86,333, >= 6/8 chunks, no penalty, R1 not negative (public/private rank corr was
0.995 in R1, so the public tape is a good proxy; score is a z-score with pnl <= 0 counted as 0). Upload every new
best immediately (absolute path, local pnl in `submission/README.md`), log in `docs/progress.md`, push after each
result. Quick sanity before any sim: `explore_data.py` markout of touch trades bucketed by sig (measured >= 5 ms
before the trade) gives the filter threshold s directly.

## UPDATE 21:05 — independent check (Copilot) changes the order: maker filter first, taker second

- With offsets learned on the first 20 % only and samples spaced 500 ms apart (no overlap) the signal holds:
  corr R2 **+0.53** at 100 ms, +0.16 at 1 s; R1 +0.34 / +0.38. But the non-overlapping taker is weak: R2 thr 1.0
  mean slightly NEGATIVE, positive only at high thresholds with few entries. So: **do T3 (maker filter) first,
  T2 only as a small high-threshold overlay** (thr >= 2), judged on a holdout chunk.
- Online parity is the first gate: every G insert is within 7 ticks of the touch (inside the 20-tick feed) and
  `on_order_book_change` fires once per tape row, so the new id in each snapshot IS the insert. Build sig online,
  log it, and check corr with the offline value > 0.99 before any grid.
- Generator facts (Copilot): insert id = 1,000,001 + insert index; cancel hazard is not memoryless (0.6/s in the
  first 25 ms, ~2.3/s at 150-400 ms, then lower); at-touch arrivals rise with touch depth (22/s -> 61/s), so deep
  queues get MORE joiners behind us (good for front fills). IOC limit price / original size are not in the feed.
- T3 rule detail (worth trying as the variant): when the side is against the signal, modify to a 1-lot stub if
  < 20 lots are ahead of us (keeps the front), cancel if >= 20 ahead; re-quote full size when |sig| <= s again
  and the queue at the touch is short, or when the touch price moves.
- Do NOT use the Gemini taker code circulating in chat: it returns orders instead of calling send_order, never
  fills `pending_orders` (inventory stays 0 -> the 100 cap is not enforced), and its signal is not time-decayed.
