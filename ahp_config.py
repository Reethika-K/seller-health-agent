# ahp_config.py

# The order here must match the row/column order of the matrices below
CATEGORIES = ["Marketing", "Logistics", "Customer Experience",
              "Seller support", "Finance", "Inventory"]

METRICS = {
    "Marketing": ["listing_quality_score", "content_flag_count", "ad_return"],
    "Logistics": ["late_shipment_rate", "cancellation_rate"],
    "Customer Experience": ["avg_rating", "return_rate", "negative_review_rate", "complaint_count"],
    "Seller support": ["avg_response_hours", "sla_compliance"],
    "Finance": ["payout_delay_rate", "refund_rate"],
    "Inventory": ["stockout_rate", "expiry_damage_rate"],
}

MEESHO = {
    "categories": [
        [1,   1,   2,   3,   5,   2],
        [1,   1,   1,   2,   4,   2],
        [1/2, 1,   1,   2,   3,   2],
        [1/3, 1/2, 1/2, 1,   2,   1],
        [1/5, 1/4, 1/3, 1/2, 1,   1/2],
        [1/2, 1/2, 1/2, 1,   2,   1],
    ],
    "Marketing": [[1, 2, 3], [1/2, 1, 2], [1/3, 1/2, 1]],
    "Customer Experience": [[1, 1, 2, 3], [1, 1, 2, 3], [1/2, 1/2, 1, 2], [1/3, 1/3, 1/2, 1]],
    "Logistics":      [[1, 2],   [1/2, 1]],
    "Seller support": [[1, 1/2], [2, 1]],
    "Finance":        [[1, 3],   [1/3, 1]],
    "Inventory":      [[1, 5],   [1/5, 1]],
}