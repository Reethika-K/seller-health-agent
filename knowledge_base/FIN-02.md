# FIN-02: Refund rate rising without more returns

Category: Finance
Metrics: refund_rate, return_rate, payout_delay_rate, cancellation_rate

**Symptom.** refund_rate rises. The important question is whether return_rate rose with it.

**Likely causes.**
1. Goodwill or dispute refunds granted without a return, for example after complaints.
2. Duplicate or wrong-amount refunds from a process or system error.
3. Refunds on cancelled prepaid orders (look at cancellation_rate).
4. More returns, which naturally raise refunds.

**How to tell them apart.** If return_rate is flat or barely moved while refund_rate rises and payout_delay_rate also rises, the cause is in finance handling (causes 1 and 2) and Finance is the primary category. If return_rate rose strongly, refunds follow returns and the root cause is in Customer Experience or Marketing (cause 4). If cancellation_rate rose, refunds follow cancellations and the root cause is Logistics or Inventory (cause 3).

**Knock-on effects.** Refunds without returns reduce revenue directly. Disputes about refunds also create buyer messages and complaints.

**First actions.** Sample the refunds from the last two weeks, mark each as returned, cancelled or no-return, and investigate the no-return group first.
