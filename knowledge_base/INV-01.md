# INV-01: Stockout rate rising with order volume falling

Category: Inventory
Metrics: stockout_rate, order_volume, cancellation_rate, expiry_damage_rate

**Symptom.** stockout_rate rises: a larger share of the seller's listings is out of stock on a given day. order_volume usually falls by a quarter or more.

**Likely causes.**
1. Replenishment failure: supplier delays or late purchase orders.
2. Demand forecast errors, so popular items run out while slow items fill the warehouse.
3. Stock count mistakes: listings show out of stock although stock exists, or the reverse.
4. A cash flow problem delaying restocking (check payout_delay_rate).

**How to tell them apart.** If stockout_rate rises and order_volume falls with no change in listing quality, lost sales from empty shelves are the cause (causes 1 and 2). If cancellation_rate also rises, the listings were still live while the stock was gone, so customers ordered items that could not be shipped (cause 3). If payout_delay_rate also rose, the seller may be short of cash (cause 4).

**Knock-on effects.** Cancellations rise, then late shipments, because the seller scrambles for stock. The problem then spreads into Logistics.

**First actions.** List the top out-of-stock listings by past sales, place replenishment orders for them, and de-list items that cannot be fulfilled.
