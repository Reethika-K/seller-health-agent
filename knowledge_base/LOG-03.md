# LOG-03: Logistics failure turning into customer and support problems

Category: Logistics
Metrics: late_shipment_rate, complaint_count, negative_review_rate, avg_rating, avg_response_hours, sla_compliance

**Symptom.** A seller was flagged for Logistics (late_shipment_rate and/or cancellation_rate), and one to three weeks later Customer Experience or Seller support is also flagged.

**Why it happens.** Late and cancelled orders create worried buyers. They send messages asking for status, file complaints, return items and leave low ratings. The seller then receives more buyer messages than usual and replies more slowly, so avg_response_hours rises and sla_compliance falls.

**How to recognize it.** The secondary category started after the Logistics problem, not before. Complaint_count, negative_review_rate and avg_rating worsen with a lag, and the response-time metrics follow. Late shipments are the cause; the rest is the effect. If a Customer Experience decline started first and late shipments came after it, the order of events is different and this document does not apply.

**What it means for the action.** Fixing the support or review metrics on their own will not last. The late shipments must be fixed first. Adding proactive delivery updates to buyers helps limit new complaints while the fix is in progress.

**First actions.** Treat Logistics as the primary issue, list orders currently late, send status updates to affected buyers, and clear the backlog of unanswered messages.
