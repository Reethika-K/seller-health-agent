"""
Checks that a platform's data matches the Meesho schema before running the pipeline.

Run:  python check_data.py nykaa      (or instamart)
It compares every table's columns with meesho_data/, so meesho_data/ must be in the same folder.
"""
import sys
import pandas as pd

TABLES = ["sellers", "listings", "listing_daily", "orders", "order_items", "order_payments",
          "order_returns", "order_reviews", "complaints", "buyer_messages", "ground_truth"]
platform = sys.argv[1].lower() if len(sys.argv) > 1 else "nykaa"
problems = []


def read(folder, name):
    try:
        return pd.read_csv(f"{folder}/{name}.csv")
    except FileNotFoundError:
        return None


for t in TABLES:
    ref, new = read("meesho_data", t), read(f"{platform}_data", t)
    if new is None:
        problems.append(f"{t}.csv is missing in {platform}_data/")
        continue
    if list(new.columns) != list(ref.columns):
        missing = [c for c in ref.columns if c not in new.columns]
        extra = [c for c in new.columns if c not in ref.columns]
        problems.append(f"{t}.csv columns differ. Missing: {missing or '-'}  Extra: {extra or '-'}")
    print(f"{t:16s} {len(new):>8,} rows")

sellers = read(f"{platform}_data", "sellers")
if sellers is not None:
    print("\nsellers:", sellers.seller_id.nunique(), "| size tiers:", sellers.size_tier.value_counts().to_dict())
    if set(sellers.size_tier) != {"Small", "Medium", "Large"}:
        problems.append("size_tier must contain exactly Small, Medium and Large")
ld = read(f"{platform}_data", "listing_daily")
if ld is not None and "date" in ld.columns:
    d = pd.to_datetime(ld["date"])
    print("dates:", d.min().date(), "to", d.max().date(), f"({d.nunique()} days)")
    if d.nunique() != 90 or str(d.min().date()) != "2026-07-01":
        problems.append("expected 90 days starting 2026-07-01 (same grid as Meesho)")
gt = read(f"{platform}_data", "ground_truth")
if gt is not None and "is_at_risk" in gt.columns:
    print("declining sellers in ground truth:", int(gt.is_at_risk.sum()))

print("\n" + ("PROBLEMS FOUND:\n- " + "\n- ".join(problems) if problems else "All checks passed. Next: python aggregate_metrics.py " + platform))