# SUP-01: Response time rising

Category: Seller support
Metrics: avg_response_hours, sla_compliance, late_shipment_rate, complaint_count

**Symptom.** avg_response_hours (the average time the seller takes to answer buyer messages) increases sharply, often by a factor of several.

**Likely causes.**
1. Staffing: the person answering messages is absent, left, or overloaded.
2. Notification or tooling problems: messages are not reaching the seller (app notifications off, wrong contact).
3. Message volume jumped because of other problems, such as late deliveries.
4. A change of process, for example a new ticketing tool, without training.

**How to tell them apart.** If late_shipment_rate or complaint_count rose first, the extra messages are a symptom of a Logistics or Customer Experience problem, and the response-time metric is a consequence (cause 3). If avg_response_hours is the only metric that moved and the other categories are stable, the problem is in the seller's support capacity itself (causes 1, 2, 4). A very sudden jump with no other change often means notifications or tooling.

**Knock-on effects.** Buyers who wait are more likely to file complaints and leave low ratings, so Customer Experience may follow after one to three weeks.

**First actions.** Check that notifications reach the right person, assign a backup responder, and set up an auto-reply with expected response times.
