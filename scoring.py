import sys 
import numpy as np
import pandas as pd

from ahp import ahp_weights
from ahp_config import CATEGORIES, METRICS, LEVEL_MATRIX, NYKAA
from levels import (ALL_METRICS, build_thresholds, load_daily, load_tiers,
                    metric_level, worsening)

FLAG_SCORE = 0.35        # a category is flagged when its sub-score is at or below this
FIRST_RUN_DAY = 34       # need ~4 weeks of history before the first weekly run


def level_scores():
    """Score of each level (Healthy..Critical) from the AHP level matrix, scaled so Healthy = 1."""
    w, _ = ahp_weights(LEVEL_MATRIX)
    return w / w.max()


def get_weights(matrices):
    """Category weights (Level 1) and local metric weights inside each category (Level 2)."""
    cw, _ = ahp_weights(matrices["categories"])
    cat_w = dict(zip(CATEGORIES, cw))
    local = {}
    for cat in CATEGORIES:
        lw, _ = ahp_weights(matrices[cat])
        local[cat] = dict(zip(METRICS[cat], lw))
    return cat_w, local


def score_sellers(df, as_of, th, tiers, cat_w, local, lscores):
    """One row per seller: a sub-score per category (1.0 = no deterioration,
    lower = worse) and an overall weighted score."""
    change = worsening(df, as_of)
    rows = {}
    for seller, r in change.iterrows():
        cuts = th[tiers[seller]]
        row = {}
        for cat in CATEGORIES:
            num = den = 0.0
            for m, lw in local[cat].items():
                lvl = metric_level(r[m], cuts[m])
                if lvl is None:        # no data for this metric: skip it and
                    continue           # re-scale the remaining weights
                num += lw * lscores[lvl]
                den += lw
            row[cat] = num / den if den > 0 else np.nan

        num = den = 0.0
        for cat in CATEGORIES:
            if not np.isnan(row[cat]):
                num += cat_w[cat] * row[cat]
                den += cat_w[cat]
        row["overall"] = num / den if den > 0 else np.nan
        rows[seller] = row
    return pd.DataFrame(rows).T


def run_weekly(platform, matrices):
    """Run the scoring every week. An alert is raised when the same category is
    flagged in two consecutive weekly runs (this filters out one-off blips)."""
    df = load_daily(platform)
    tiers = load_tiers(platform)
    th = build_thresholds(df, tiers)
    cat_w, local = get_weights(matrices)
    ls = level_scores()

    start = df.date.min()
    run_days = range(FIRST_RUN_DAY, (df.date.max() - start).days + 1, 7)
    previous = {}                  # seller -> category flagged in the previous run
    alerts = {}                    # seller -> first confirmed alert
    for d in run_days:
        as_of = start + pd.Timedelta(days=d)
        scores = score_sellers(df, as_of, th, tiers, cat_w, local, ls)
        current = {}
        for seller, row in scores.iterrows():
            worst_cat = row[CATEGORIES].astype(float).idxmin()
            if row[worst_cat] <= FLAG_SCORE:
                current[seller] = worst_cat
                if previous.get(seller) == worst_cat and seller not in alerts:
                    alerts[seller] = {"seller_id": seller, "as_of": as_of.date(),
                                      "category": worst_cat,
                                      "category_score": round(row[worst_cat], 3),
                                      "overall": round(row["overall"], 3)}
        previous = current
    return pd.DataFrame(alerts.values())


def evaluate(platform, alerts):
    """Compare the alerts with the hidden ground truth (for testing only)."""
    gt = pd.read_csv(f"{platform}_data/ground_truth.csv").set_index("seller_id")
    risk = set(gt.index[gt.is_at_risk == 1])
    dips = set(gt.index[gt.temporary_dip_category.notna() & (gt.is_at_risk == 0)])
    healthy = set(gt.index) - risk - dips
    flagged = set(alerts.seller_id)
    caught = risk & flagged
    a = alerts.set_index("seller_id")
    routed = sum(a.loc[s, "category"] in (gt.loc[s, "primary_category"],
                                          gt.loc[s, "secondary_category"]) for s in caught)
    lags = [(pd.Timestamp(a.loc[s, "as_of"]) - pd.Timestamp(gt.loc[s, "decline_start_date"])).days
            for s in caught]
    print(f"Declining sellers caught:     {len(caught)}/{len(risk)}")
    print(f"Routed to the right category: {routed}/{len(caught)}")
    print(f"Temporary dips flagged:       {len(dips & flagged)}/{len(dips)}")
    print(f"Healthy sellers flagged:      {len(healthy & flagged)}/{len(healthy)}")
    print(f"Median days from decline start to alert: {np.median(lags):.0f}")


if __name__ == "__main__":
    platform = "nykaa"
    print("level scores:", level_scores().round(3))
    alerts = run_weekly(platform, NYKAA)
    print(f"\n{len(alerts)} alerts raised. Highest priority first:")
    print(alerts.sort_values("overall").head(10).to_string(index=False))
    print()
    evaluate(platform, alerts)