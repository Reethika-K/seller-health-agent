# LOG-02: Cancellation rate rising

Category: Logistics
Metrics: cancellation_rate, late_shipment_rate, stockout_rate, order_volume, refund_rate

**Symptom.** cancellation_rate rises: more orders are cancelled before delivery, mostly by the seller.

**Likely causes.**
1. Overselling: the listing shows stock that is not there (inventory sync errors, stock sold on another channel).
2. The seller cannot ship in time because of a dispatch backlog, and cancels to avoid late-shipment penalties.
3. Pricing or listing errors that the seller then cancels (wrong price, wrong variant).

**How to tell them apart.** Look at stockout_rate and order_volume. If stockout_rate rises and order_volume falls, overselling is the cause and the problem belongs to Inventory (cause 1). If late_shipment_rate rises together with cancellation_rate while order_volume stays stable, the seller is struggling to dispatch (cause 2) and the problem belongs to Logistics. A short spike on a few listings with no other change usually points to a pricing error (cause 3).

**Knock-on effects.** Cancelled prepaid orders are refunded, so refund_rate rises. Buyers who had their order cancelled rate the seller lower and may raise complaints.

**First actions.** Reconcile stock counts with the listings, pause listings that cannot be fulfilled, and check recent price or variant edits.
