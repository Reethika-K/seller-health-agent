# LOG-01: Late shipment rate rising

Category: Logistics
Metrics: late_shipment_rate, cancellation_rate, stockout_rate, order_volume, complaint_count

**Symptom.** late_shipment_rate rises: more orders are handed to the courier after the promised dispatch time.

**Likely causes.**
1. Dispatch backlog: order volume grew faster than packing capacity, or staff were absent.
2. Courier or pickup problems: missed pickups, a change of courier, or pickup slots that no longer match the seller's cut-off.
3. Stock was not actually ready when the order arrived, so packing waited on replenishment.
4. Order processing cut-offs changed, so orders placed late in the day miss the dispatch window.

**How to tell them apart.** If order_volume is stable or growing and stockout_rate is flat, suspect dispatch capacity or the courier (causes 1, 2, 4). If stockout_rate rises and order_volume falls, the root cause is inventory and the case should be treated as an Inventory problem (cause 3). If cancellation_rate rises together with late_shipment_rate, the seller cannot fulfil orders on time and is cancelling some of them.

**Knock-on effects.** Buyers complain and rate the seller lower: complaint_count and negative_review_rate rise one to three weeks later. Buyers also message more ("where is my order"), which loads seller support.

**First actions.** Check pickup logs and courier performance for the seller, compare dispatch capacity with order volume, and set realistic handling times.
