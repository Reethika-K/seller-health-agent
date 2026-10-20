# INV-03: Inventory problems turning into cancellations and late shipments

Category: Inventory
Metrics: stockout_rate, cancellation_rate, late_shipment_rate, order_volume

**Symptom.** A seller was flagged for Inventory (stockout_rate and/or expiry_damage_rate), and one to three weeks later Logistics is also flagged (cancellation_rate, late_shipment_rate).

**Why it happens.** When stock is short, the seller either cancels orders it cannot fill or waits for new stock before packing, so orders ship late. The extra cancellations and delays are consequences of the stock problem, not a separate courier or packing failure.

**How to recognize it.** The Inventory flag started first. order_volume dropped at the same time as stockout_rate rose. The Logistics metrics worsened afterwards, especially cancellation_rate. If late_shipment_rate rose with no stockout change and order_volume stayed stable, the Logistics problem is independent.

**What it means for the action.** Improving courier pickups or packing speed will not help while the stock is missing. Inventory should be the primary category and Logistics the secondary one. Fixing replenishment resolves both.

**First actions.** Restock the most-ordered listings, stop selling items that are not in stock, and review all open orders for items with no stock.
