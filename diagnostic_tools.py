"""
Tools for the diagnostic agent. Same idea as monitor_tools.py: the arithmetic is done in plain
Python, and the LLM only reads the results. The type hints and docstrings matter, because Gemini
reads them to decide how to call each tool.

What is new here is the TIMELINE: when did a category start getting worse, and how fast?
How it is measured (one metric at a time):
  1. Baseline = the seller's first 21 days of data (the seller was healthy then).
  2. Deviation = how much worse each day is than that baseline (a drop for "higher is better"
     metrics, a rise for the others), smoothed with a 7-day rolling average.
  3. A metric is "elevated" if its last-7-day deviation is bigger than 3x the normal noise.
  4. Onset = the first day of the final unbroken run above that noise limit.
  5. Days to 80% = how many days after onset the deviation reached 80% of its current level.
     Few days means sudden, many days means gradual.
These are estimates. A decline that starts slowly is only noticed a few days after it began.
"""
import numpy as np
import pandas as pd

import monitor_tools as mt
from ahp_config import CATEGORIES, METRICS
from kb import search_knowledge          # noqa: F401  (re-exported as a tool for the agent)
from levels import ALL_METRICS, HIGHER_IS_BETTER

BASE_DAYS = 21        # the seller's first 3 weeks are treated as its healthy baseline
ROLL = 7              # smoothing window in days
NOISE_K = 3.0         # a metric is elevated when it is this many noise-widths above baseline
REACH = 0.8           # "reached its current level" = 80% of it
MIN_EFFECT = 4.0      # ignore metrics whose change is small relative to their noise
SUDDEN_DAYS = 5       # reaching 80% within this many days of onset counts as sudden
WEEKS = 8


def metric_timeline(seller_id, metric, as_of):
    """Onset and speed of ONE metric for ONE seller, or None if it is not elevated."""
    df = mt._load()["df"]
    s = df[(df.seller_id == seller_id) & (df.date <= as_of)].set_index("date")[metric].astype(float)
    if len(s) < BASE_DAYS + ROLL:
        return None
    base = s.iloc[:BASE_DAYS]
    b, sd = base.mean(), max(base.std(), 1e-6)
    dev = (b - s) if metric in HIGHER_IS_BETTER else (s - b)
    smooth = dev.rolling(ROLL, min_periods=3).mean()
    level_now = dev.iloc[-7:].mean()
    limit = NOISE_K * sd / np.sqrt(ROLL)
    if not level_now > limit:
        return None
    above = (smooth > limit).values
    i, gaps = len(above) - 1, 0
    while i >= 0:                              # walk back through the final run above the limit
        gaps = 0 if above[i] else gaps + 1
        if gaps > 2:                           # a gap of 3+ days ends the run
            break
        i -= 1
    start = min(max(i + gaps + 1, 0), len(smooth) - 1)
    seg = smooth.iloc[start:]
    reached = (seg >= REACH * level_now).values
    days = int(np.argmax(reached)) if reached.any() else len(seg) - 1
    weekly = [s.iloc[max(len(s) - 7 * (k + 1), 0): len(s) - 7 * k].mean() for k in range(WEEKS)][::-1]
    return {"metric": metric, "onset_date": str(smooth.index[start].date()),
            "strength": round(float(level_now / sd), 1), "days_to_80_percent": days,
            "weekly_values_oldest_to_newest": [None if pd.isna(v) else round(float(v), 3) for v in weekly]}


def category_facts(seller_id, category, as_of):
    """Onset date and speed for a whole category, from its elevated metrics."""
    found = [x for x in (metric_timeline(seller_id, m, as_of) for m in METRICS[category]) if x]
    strong = [x for x in found if x["strength"] >= MIN_EFFECT]
    if not found:
        return {"elevated_metrics": [], "estimated_onset_date": None, "decline_speed": "unclear"}
    use = strong or [max(found, key=lambda x: x["strength"])]
    onset = min(x["onset_date"] for x in use)                     # earliest of the clear signals
    days = float(np.median([x["days_to_80_percent"] for x in use]))
    speed = "sudden" if days <= SUDDEN_DAYS else "gradual"
    return {"elevated_metrics": found, "estimated_onset_date": onset,
            "median_days_to_80_percent": days, "decline_speed": speed}


def get_category_timeline(seller_id: str, category: str, as_of_date: str) -> dict:
    """When did a category start getting worse for one seller, and how fast?

    Returns, for the metrics of that category that are clearly worse than the seller's
    healthy baseline: the estimated onset date, the weekly values of the metric over the
    last 8 weeks (oldest first), and how many days it took to reach 80% of its current
    level. Also returns estimated_onset_date for the whole category and decline_speed
    ("sudden" or "gradual"). Use the onset dates of two categories to see which one started
    first, which tells you which is the root cause and which is a knock-on effect.

    Args:
        seller_id: for example "S0042".
        category: one of Marketing, Logistics, Customer Experience, Seller support,
            Finance, Inventory.
        as_of_date: the date to assess, formatted YYYY-MM-DD.
    """
    if category not in METRICS:
        return {"error": f"category must be one of {CATEGORIES}"}
    out = category_facts(seller_id, category, pd.Timestamp(as_of_date))
    return {"seller_id": seller_id, "category": category, "as_of_date": as_of_date, **out}


def get_seller_snapshot(seller_id: str, as_of_date: str) -> dict:
    """All 15 metrics of one seller, grouped by category, for the last 7 days.

    For every metric: the average over the last 7 days, the average over the 21 days before
    that, and the level (Healthy, Watch, At-risk, Critical). Use it first, to see which
    metrics moved and which did not. Metrics that did NOT move are evidence too: for example
    stockouts staying flat rules out a stock problem as the cause of late shipments.

    Args:
        seller_id: for example "S0042".
        as_of_date: the date to assess, formatted YYYY-MM-DD.
    """
    cats = {}
    for c in CATEGORIES:
        detail = mt.get_metric_detail(seller_id, c, as_of_date)
        if "error" in detail:
            return detail
        cats[c] = detail["metrics"]
    d = pd.Timestamp(as_of_date)
    df = mt._load()["df"]
    w = df[(df.seller_id == seller_id) & (df.date > d - pd.Timedelta(days=7)) & (df.date <= d)]
    base = df[(df.seller_id == seller_id) & (df.date > d - pd.Timedelta(days=28))
              & (df.date <= d - pd.Timedelta(days=7))]
    return {"seller_id": seller_id, "size_tier": mt._load()["tiers"][seller_id], "as_of_date": as_of_date,
            "order_volume_last_7_days_per_day": round(float(w.order_volume.mean()), 1),
            "order_volume_previous_21_days_per_day": round(float(base.order_volume.mean()), 1),
            "categories": cats}


TOOLS = [get_seller_snapshot, get_category_timeline, mt.get_seller_history, search_knowledge]