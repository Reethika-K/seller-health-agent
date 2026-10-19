import pandas as pd
from ahp_config import METRICS

HIGHER_IS_BETTER = {"listing_quality_score", "ad_return", "avg_rating", "sla_compliance"}
ALL_METRICS = [m for metrics in METRICS.values() for m in metrics]

RECENT_DAYS = 7            # "now" = the last 7 days
BASELINE_DAYS = 21         # compared with the 21 days before that
CUT_QUANTILES = [0.85, 0.95, 0.99]    # Watch / At-risk / Critical cut-offs
REFERENCE_END_DAYS = range(28, 39)    # early, healthy period used to learn the cut-offs


def load_daily(platform):
    df = pd.read_csv(f"{platform}_data/seller_daily_metrics.csv", parse_dates=["date"])
    return df.sort_values(["seller_id", "date"])


def load_tiers(platform):
    """Seller size tier (Small / Medium / Large) - a General-category attribute."""
    return pd.read_csv(f"{platform}_data/sellers.csv").set_index("seller_id")["size_tier"]


def window_mean(df, end, days):
    """Average of every metric per seller over the `days` days ending on `end`."""
    end = pd.Timestamp(end)
    w = df[(df.date > end - pd.Timedelta(days=days)) & (df.date <= end)]
    return w.groupby("seller_id")[ALL_METRICS].mean()


def worsening(df, end):
    """How much worse the last 7 days are than the seller's own previous 21 days.
    Positive = got worse, for every metric."""
    end = pd.Timestamp(end)
    recent = window_mean(df, end, RECENT_DAYS)
    base = window_mean(df, end - pd.Timedelta(days=RECENT_DAYS), BASELINE_DAYS)
    change = recent - base
    for m in HIGHER_IS_BETTER:
        change[m] = -change[m]       # for "higher is better" metrics, a drop is worse
    return change


def build_thresholds(df, tiers):
    """Learn, per size tier and per metric, how much worsening is normal.
    Returns th[tier][metric] = [watch_cut, at_risk_cut, critical_cut]."""
    start = df.date.min()
    windows = [worsening(df, start + pd.Timedelta(days=d)).assign(tier=tiers)
               for d in REFERENCE_END_DAYS]
    ref = pd.concat(windows)
    th = {}
    for tier in ["Small", "Medium", "Large"]:
        sub = ref[ref.tier == tier]
        th[tier] = {m: sub[m].quantile(CUT_QUANTILES).tolist() for m in ALL_METRICS}
    return th


def metric_level(worse_by, cuts):
    """0 = Healthy, 1 = Watch, 2 = At-risk, 3 = Critical
    (the number of cut-offs the worsening has crossed). None if the value is missing."""
    if pd.isna(worse_by):
        return None
    return sum(worse_by > c for c in cuts)


if __name__ == "__main__":
    df = load_daily("meesho")
    th = build_thresholds(df, load_tiers("meesho"))
    print("Small sellers: how much worse than their own baseline before it counts as Watch / At-risk / Critical")
    for m in ALL_METRICS:
        print(f"{m:24s}", [round(c, 3) for c in th["Small"][m]])
    print("\nmetric_level(0.30, [0.10, 0.20, 0.25]) ->", metric_level(0.30, [0.10, 0.20, 0.25]), "(expect 3)")
    print("metric_level(0.05, [0.10, 0.20, 0.25]) ->", metric_level(0.05, [0.10, 0.20, 0.25]), "(expect 0)")