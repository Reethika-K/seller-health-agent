"""
Turns the Meesho event tables into one row per seller per day with the 15 metrics.
This is the GROUP BY step the monitoring agent reads from.

Run:  python aggregate_metrics.py meesho      (or nykaa / instamart)
Out:  <platform>_data/seller_daily_metrics.csv
"""
import sys
import pandas as pd

PLATFORM = sys.argv[1].lower() if len(sys.argv) > 1 else "meesho"
D = f"{PLATFORM}_data/"
sellers = pd.read_csv(D + "sellers.csv")
listings = pd.read_csv(D + "listings.csv")
ldaily = pd.read_csv(D + "listing_daily.csv")
orders = pd.read_csv(D + "orders.csv")
items = pd.read_csv(D + "order_items.csv")
pay = pd.read_csv(D + "order_payments.csv")
rets = pd.read_csv(D + "order_returns.csv")
revs = pd.read_csv(D + "order_reviews.csv")
cmps = pd.read_csv(D + "complaints.csv")
msgs = pd.read_csv(D + "buyer_messages.csv")

# every item gets its date, and flags from the child tables
items = items.merge(orders, on="order_id")
items = items.merge(pay[["order_item_id", "is_payout_delayed", "is_refunded"]], on="order_item_id")
items["is_returned"] = items["order_item_id"].isin(rets["order_item_id"]).astype(int)
keys = ["seller_id", "order_date"]

g = items.groupby(keys)
daily = pd.DataFrame({
    "order_volume": g["order_id"].nunique(),
    "late_shipment_rate": g["is_late_shipment"].mean(),
    "cancellation_rate": g["is_cancelled"].mean(),
    "return_rate": g["is_returned"].mean(),
    "payout_delay_rate": g["is_payout_delayed"].mean(),
    "refund_rate": g["is_refunded"].mean(),
})

r = revs.merge(items[["order_item_id", "seller_id", "order_date"]], on="order_item_id")
rg = r.groupby(keys)["rating"]
daily["avg_rating"] = rg.mean()
daily["negative_review_rate"] = (r["rating"] <= 2).groupby([r["seller_id"], r["order_date"]]).mean()

c = cmps.merge(items[["order_item_id", "seller_id", "order_date"]], on="order_item_id")
daily["complaint_count"] = c.groupby(keys).size()

m = msgs.merge(orders, on="order_id")
mg = m.groupby(keys)
daily["avg_response_hours"] = mg["response_time_hours"].mean()
daily["sla_compliance"] = mg["is_sla_met"].mean()

ld = ldaily.merge(listings, on="listing_id").rename(columns={"date": "order_date"})
lg = ld.groupby(keys)
daily["listing_quality_score"] = lg["listing_quality_score"].mean()
daily["content_flag_count"] = lg["content_flag"].sum()
daily["ad_return"] = lg["ad_return"].mean()
daily["stockout_rate"] = lg["is_stockout"].mean()
daily["expiry_damage_rate"] = lg["is_expired_or_damaged"].mean()

daily = daily.reset_index().rename(columns={"order_date": "date"})
full = pd.MultiIndex.from_product([sellers["seller_id"], sorted(ld["order_date"].unique())],
                                  names=["seller_id", "date"]).to_frame(index=False)
daily = full.merge(daily, on=["seller_id", "date"], how="left")
daily["order_volume"] = daily["order_volume"].fillna(0).astype(int)
daily["complaint_count"] = daily["complaint_count"].fillna(0).astype(int)
daily = daily.round(4)
daily.to_csv(D + "seller_daily_metrics.csv", index=False)
print(f"{PLATFORM}: seller_daily_metrics.csv -> {len(daily):,} rows, {daily.shape[1]} columns")