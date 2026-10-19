import pandas as pd
from ahp_config import METRICS

HIGHER_IS_BETTER = {"listing_quality_score", "ad_return", "avg_rating", "sla_compliance"}

# Level order: 0 = Healthy, 1 = Watch, 2 = At-risk, 3 = Critical
LEVELS = ["Healthy", "Watch", "At-risk", "Critical"]


def build_thresholds(platform):
    """For every metric, return 3 cut-offs (Watch, At-risk, Critical) learned from
    the platform's own early data, before any seller has started to decline."""
    df = pd.read_csv(f"{platform}_data/seller_daily_metrics.csv", parse_dates=["date"])
    df = df.sort_values(["seller_id", "date"])

    # Scoring compares 7-day averages, so build the cut-offs from 7-day averages too
    cols = [c for c in df.columns if c not in ("seller_id", "date", "order_volume")]
    for m in cols:
        df[m] = df.groupby("seller_id")[m].transform(
            lambda s: s.rolling(7, min_periods=3).mean())

    start = df.date.min()
    ref = df[(df.date >= start + pd.Timedelta(days=6)) &
             (df.date < start + pd.Timedelta(days=25))]

    th = {}
    for cat, metrics in METRICS.items():
        for m in metrics:
            if m in HIGHER_IS_BETTER:          # bad values are LOW
                qs = [0.25, 0.10, 0.03]
            else:                              # bad values are HIGH
                qs = [0.75, 0.90, 0.97]
            th[m] = ref[m].quantile(qs).tolist()
    return th


def metric_level(value, cuts, higher_is_better):
    """Return 0 (Healthy), 1 (Watch), 2 (At-risk) or 3 (Critical).
    The level is the number of cut-offs the value has crossed.
    Returns None if the value is missing."""
    if pd.isna(value):
        return None
    if higher_is_better:
        return sum(value < c for c in cuts)
    return sum(value > c for c in cuts)


if __name__ == "__main__":
    th = build_thresholds("meesho")
    print(f"{'metric':24s} {'Watch':>9s} {'At-risk':>9s} {'Critical':>9s}")
    for m, cuts in th.items():
        print(f"{m:24s} {cuts[0]:9.3f} {cuts[1]:9.3f} {cuts[2]:9.3f}")

    print("\nlate_shipment_rate tests (expect 0, 1, 2, 3):")
    for v in [0.05, 0.12, 0.17, 0.40]:
        print(v, "->", metric_level(v, th["late_shipment_rate"], False))

    print("\nlisting_quality_score tests (expect 0, 1, 2, 3):")
    for v in [78, 69, 65, 55]:
        print(v, "->", metric_level(v, th["listing_quality_score"], True))

    print("\nmissing value ->", metric_level(float("nan"), th["avg_rating"], True))