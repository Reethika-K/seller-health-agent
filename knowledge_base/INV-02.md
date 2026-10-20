# INV-02: Expiry and damage rate rising

Category: Inventory
Metrics: expiry_damage_rate, stockout_rate, order_volume

**Symptom.** expiry_damage_rate rises: a larger share of the seller's stock is expired or damaged. This matters most in categories with shelf life, such as beauty and personal care, and food.

**Likely causes.**
1. Overstocking of slow-moving items that reach their expiry date.
2. Poor storage conditions (heat, humidity, handling) or damage during storage.
3. Receiving stock with short remaining shelf life from suppliers.
4. Weak first-expiry-first-out discipline, so old stock is not sold first.

**How to tell them apart.** If expiry_damage_rate rises while stockout_rate stays flat, the problem is storage or stock rotation (causes 2 to 4). If stockout_rate rises at the same time, inventory control is breaking down overall: some items are missing while others are wasted (cause 1 together with forecast errors). A sudden jump on one batch points to the supplier or a storage incident.

**Knock-on effects.** Written-off stock hurts margin and reduces what can be sold, which lowers order_volume. Damaged or near-expiry items that are shipped can lead to returns and complaints.

**First actions.** Count stock by expiry date, mark down items that expire soon, fix storage conditions, and enforce first-expiry-first-out.
