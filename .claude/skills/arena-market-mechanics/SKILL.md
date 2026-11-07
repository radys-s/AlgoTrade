---
name: arena-market-mechanics
description: How the Algo Trade Arena exchange simulator actually behaves (replayed non-reactive tape, 5 ms activation rule, event loop order, price-time matching with our orders, self-trade skip, zombie and missing liquidity, markout, limits) and what that implies for strategy design. Use before designing or debugging any strategy, signal or fill analysis in this repo.
---

# The simulated market: what is real here

The arena scores submissions with the same code as `simulator/` (workshop deck, slide 5:
"the exchange, same code as scoring"). So the simulator's quirks *are* the market. Optimise
for this engine, not for a real exchange; verify any claim below in `simulator/simulator.py`
before betting a strategy on it.

## The world

- One anonymous instrument. The tape (`Time,Operation,OrderId,Side,Volume,Price,Lifespan`) is
  other traders' Inserts and Cancels only; every trade is the engine matching them. `G` rests,
  `F` is IOC. Integer ticks and lots, no fees. The rounds are recorded data; the sample is a
  synthetic trending hour and looks nothing like them.
- The tape is fixed and never reacts to us. Nobody re-quotes, chases or fades our orders.

## Event loop, per tape row (in this order)

1. Our pending orders/cancels/modifies whose ready time `t_sent + 5 ms` is <= this row's time
   are applied, oldest first, at their ready time; fills fire `on_private_order_update` now.
2. The market row is applied (insert crosses then rests, or cancel); it may fill our resting
   orders -> private updates.
3. `on_order_book_change(snapshot, trades)`: book after the row, plus every trade since the
   previous call (market-vs-market too; fills from our own activations show up here next call).

Consequences: an order sent at time t is applied at t+5 ms, just before the first row at or
after t+5 ms; every row inside the 5 ms window happens before it. In a quiet market nothing
changes in between, but you only hear about it (private update, then the next snapshot) when
that next row arrives, possibly seconds later, and that row is then processed against a book
that already contains your order. A cancel can lose the race to a fill (count inventory from
fills only). Orders sent from inside `on_private_order_update` also get +5 ms.

## Matching, as implemented

- Price-time priority, one FIFO queue per price; fills at the resting (maker) price; a LIMIT
  residual joins the back of its level, an IOC residual is CANCELED.
- Market takers match our resting orders in the same queues (`allow_user_orders=True`).
- Our takers skip our own resting orders (no self-trade) and match the market orders behind
  them. Never send a buy at or above our own resting ask (or the mirror): the residual would
  rest crossed against ourselves.
- The snapshot lists **individual orders** (id, price, quantity, owner) within 20 ticks per
  side, best first, in queue order. Our exact queue position = market quantity listed before
  our order at our price. Fill probability can be modelled from it directly.
- Modify only shrinks and keeps queue place; a cancel or modify of something gone is REJECTED.

## Replay artefacts (real in scoring, so usable, but know them)

- **Zombie liquidity.** If our resting order is ahead of a market order in the queue (we joined
  a level earlier, or improved the price), a market taker fills us instead of the order it
  historically hit. That market order stays in the book; often no tape cancel ever comes for
  it, because historically it was filled. The simulated book drifts from history and grows
  extra resting size where we quoted aggressively. Zombies count in the markout.
- **Missing liquidity.** Orders our IOCs take are gone for good: later tape cancels for them are
  no-ops, and later market takers that historically hit them go deeper or rest instead. Taking
  has a permanent footprint; heavy taking thins the simulated book.
- The 1000 reject check uses inventory plus *resting* same-side quantity at activation; orders
  still in flight are not counted, so in-flight quantity must be tracked by the strategy.

## Score and risk

- PnL = cash + markout(I) - penalty. Markout liquidates I against resting **market** orders
  (not ours) on the opposite side from the best price outwards, at the end and every second for
  `pnl.csv`. A large final |I| sells into a thin book: keep inventory small.
- |I| > 100 costs 10,000 per second, accrued for the exact time spent outside. Even 10 ms costs
  100. Guarantee |I| + live + in-flight same-side quantity <= 100 by construction.
- Round score is the z-score of PnL across teams, with PnL <= 0 counted as 0: steady positive
  PnL in every round beats rare big wins. Scoring is on private data from the same market.

## Design implications worth testing

- Passive market making earns the spread with no fees, but the tape does not care about us:
  adverse selection comes from informed flow in the recording (measure the mid move 100 ms to
  5 s after our fills by spread, queue position and recent flow).
- Use queue position: quote where enough market size sits *behind* us to absorb takers, pull
  when we are last in a thin queue before a likely sweep.
- Latency game: quote changes take effect only before the next row >= t+5 ms; in bursts, many
  rows pass before our cancel lands. Throttle requotes and size by expected activity.
- Signals from the exact trade list and per-order book: order-flow imbalance, microprice,
  queue depletion at the touch, sweep detection, order-id patterns of repeated participants.
- Runtime is part of the market: up to ~9M callbacks in 15 min on 1 CPU.
