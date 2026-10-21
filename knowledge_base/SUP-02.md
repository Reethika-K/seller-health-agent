# SUP-02: SLA compliance falling

Category: Seller support
Metrics: sla_compliance, avg_response_hours, order_volume

**Symptom.** sla_compliance (the share of buyer messages answered within the required time) falls.

**Likely causes.**
1. Coverage gaps: nobody answers at night, on weekends or in busy hours.
2. A long tail of slow replies: most messages are handled quickly but a few queues are neglected.
3. A broad slowdown, where nearly all replies are slower (staff shortage or tool problem).
4. Higher volume of messages without more staff.

**How to tell them apart.** If sla_compliance falls but avg_response_hours rises only a little, the problem is specific gaps in time or queue (causes 1 and 2). If both fall and rise strongly together, it is a broad slowdown (causes 3 and 4). For a seller with very few messages, a handful of late replies can move the ratio a lot, so judge it over a week.

**Knock-on effects.** Late replies increase complaints and negative reviews. Repeated SLA misses can limit the seller's eligibility for platform programs.

**First actions.** Look at the times of day when replies are late, add coverage for those slots, and prioritize unanswered messages older than the SLA window.
