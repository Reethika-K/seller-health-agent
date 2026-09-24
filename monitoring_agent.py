import numpy as np
import pandas as pd

# Each weight is the max points lost if that metric is at its worst.
# They sum to 100, so health_score is always readable as "out of 100."
def compute_health_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["late_rate"] = (df["late_shipments"] / df["orders"]).replace([np.inf, -np.inf], np.nan)
    df["return_rate"] = (df["returns"] / df["orders"]).replace([np.inf, -np.inf], np.nan)
    df["cancel_rate"] = (df["cancellations"] / df["orders"]).replace([np.inf, -np.inf], np.nan)

    # zero-order days: carry the seller's last known rate forward instead of treating as "perfect"
    for col in ["late_rate", "return_rate", "cancel_rate"]:
        df[col] = df.groupby("seller_id")[col].transform(lambda s: s.ffill().bfill().fillna(0))

    late_pen = (df["late_rate"] / 0.15).clip(upper=1) * 20
    return_pen = (df["return_rate"] / 0.25).clip(upper=1) * 25
    cancel_pen = (df["cancel_rate"] / 0.10).clip(upper=1) * 10
    rating_pen = ((4.5 - df["avg_rating"]).clip(lower=0) / 1.5).clip(upper=1) * 25
    sla_pen = ((0.95 - df["sla_pct"]).clip(lower=0) / 0.35).clip(upper=1) * 10
    complaint_pen = (df["complaints"] / 3).clip(upper=1) * 10

    penalty = late_pen + return_pen + cancel_pen + rating_pen + sla_pen + complaint_pen
    df["health_score"] = (100 - penalty).clip(lower=0, upper=100)
    return df


def compute_trends(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["seller_id", "day"]).copy()
    grp = df.groupby("seller_id")["health_score"]
    df["health_7d"] = grp.transform(lambda s: s.rolling(7, min_periods=3).mean())
    df["health_30d"] = grp.transform(lambda s: s.rolling(30, min_periods=10).mean())
    # positive trend_drop = recent average is worse than the baseline = declining
    df["trend_drop"] = (df["health_30d"] - df["health_7d"]).fillna(0)
    return df


def classify_status(health_7d, trend_drop) -> str:
    if pd.isna(health_7d):
        return "Insufficient Data"
    if health_7d < 50 or trend_drop > 20:
        return "Critical"
    if health_7d < 65 or trend_drop > 12:
        return "At-Risk"
    if health_7d < 80 or trend_drop > 6:
        return "Watch"
    return "Healthy"