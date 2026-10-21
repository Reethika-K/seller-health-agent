"""
Tools the LLM monitoring agent can call. Each one is plain Python that reads the
daily metrics and uses the AHP scoring code - the LLM never does arithmetic itself.
The type hints and docstrings matter: Gemini reads them to know how to call each tool.
"""
import pandas as pd

from ahp_config import CATEGORIES, METRICS, MEESHO
from levels import (ALL_METRICS, HIGHER_IS_BETTER, build_thresholds, load_daily,
                    load_tiers, metric_level, worsening)
from scoring import get_weights, level_scores, score_sellers

PLATFORM = "meesho"          # set this before running the agent
MATRICES = MEESHO            # the AHP matrices for that platform
LEVEL_NAMES = ["Healthy", "Watch", "At-risk", "Critical"]
FLAG_SCORE = 0.35            # a category at or below this is "flagged"
CANDIDATE_SCORE = 0.60       # categories at or below this are shown for context

_state = {}                  # loaded once, reused by every tool call
_score_cache = {}


def _load():
    if not _state:
        df = load_daily(PLATFORM)
        tiers = load_tiers(PLATFORM)
        cat_w, local = get_weights(MATRICES)
        _state.update(df=df, tiers=tiers, th=build_thresholds(df, tiers),
                      cat_w=cat_w, local=local, ls=level_scores())
    return _state


def _scores(as_of):
    """Category sub-scores for every seller at a given date (cached)."""
    s = _load()
    key = str(pd.Timestamp(as_of).date())
    if key not in _score_cache:
        _score_cache[key] = score_sellers(s["df"], key, s["th"], s["tiers"],
                                          s["cat_w"], s["local"], s["ls"])
    return _score_cache[key]


def _flagged(row, limit):
    """Categories whose sub-score is at or below `limit`, as {category: score}."""
    return {c: round(float(row[c]), 3) for c in CATEGORIES
            if not pd.isna(row[c]) and row[c] <= limit}


def _episode(tables, seller, cat):
    """Flagged weekly runs in the category's current episode, counting back from now.

    One unflagged week in between is allowed, because a noisy week can lift a score above the
    line while the problem is still there. Two unflagged weeks in a row end the episode.
    Returns (number of flagged runs, score at the most recent flagged run or None)."""
    count, gap, last_score = 0, 0, None
    for t in tables:                       # tables[0] = now, tables[1] = one week ago, ...
        v = t.loc[seller, cat]
        if not pd.isna(v) and v <= FLAG_SCORE:
            count += 1
            gap = 0
            if last_score is None:
                last_score = round(float(v), 3)
        else:
            gap += 1
            if gap >= 2:
                break
    return count, last_score


def get_flagged_candidates(as_of_date: str) -> dict:
    """Shortlist of sellers with at least one category in trouble on a given date.

    Every seller on the platform is scored by code, and a seller is shortlisted when at
    least one category sub-score is at or below 0.35 (flagged). Sub-scores run from 1.0
    (no deterioration versus the seller's own recent past) down to about 0.09 (critical).

    The rules are already applied for you. For each candidate you get:
    - status: "confirmed" if a category has been flagged for 2 or more consecutive
      weekly runs, otherwise "watch" (a possible one-off blip).
    - primary_category: the category flagged for the most consecutive weeks (ties go to
      the lower score). secondary_categories: other categories flagged now, which are
      usually knock-on effects.
    - severity: "Critical" if the primary score is below 0.15, "At-risk" otherwise
      (watch candidates are labelled "Watch").
    Confirmed candidates come first, most serious first.

    Args:
        as_of_date: the date to assess, formatted YYYY-MM-DD.
    """
    d = pd.Timestamp(as_of_date)
    tables = [_scores(d - pd.Timedelta(days=7 * k)) for k in range(6)]
    now = tables[0]
    rows = []
    for seller, r in now.iterrows():
        flagged = _flagged(r, FLAG_SCORE)
        if not flagged:
            continue
        episodes = {c: _episode(tables, seller, c) for c in CATEGORIES}
        active = {c: e for c, e in episodes.items() if e[0] >= 1}       # flagged in this episode
        primary = min(active, key=lambda c: (-active[c][0], c not in flagged,
                                             flagged.get(c, active[c][1])))
        weeks, primary_score = active[primary]
        status = "confirmed" if weeks >= 2 else "watch"
        if status == "watch":
            severity = "Watch"
        else:
            severity = "Critical" if primary_score < 0.15 else "At-risk"
        rows.append({
            "seller_id": seller, "size_tier": _load()["tiers"][seller],
            "status": status, "severity": severity,
            "primary_category": primary, "primary_score": primary_score,
            "primary_consecutive_weeks": weeks,
            "secondary_categories": {c: flagged[c] for c in flagged if c != primary},
            "overall_score": round(float(r["overall"]), 3)})
    rows.sort(key=lambda x: (x["status"] != "confirmed", x["overall_score"]))
    n_conf = sum(x["status"] == "confirmed" for x in rows)
    return {"as_of_date": str(d.date()), "sellers_scored": len(now),
            "total_candidates": len(rows), "confirmed": n_conf, "watch": len(rows) - n_conf,
            "candidates": rows}


def get_seller_history(seller_id: str, as_of_date: str, weeks: int = 4) -> dict:
    """Category sub-scores of ONE seller over the last few weekly runs.

    Use this to see whether a decline is getting worse, is recovering, or has
    been stable. Scores run from 1.0 (no deterioration) down to about 0.09.

    Args:
        seller_id: for example "S0042".
        as_of_date: the latest date to include, formatted YYYY-MM-DD.
        weeks: how many weekly runs to return, most recent first (maximum 6).
    """
    d = pd.Timestamp(as_of_date)
    out = []
    for k in range(min(weeks, 6)):
        t = d - pd.Timedelta(days=7 * k)
        if seller_id not in _scores(t).index:
            return {"error": f"unknown seller {seller_id}"}
        row = _scores(t).loc[seller_id]
        entry = {"as_of": str(t.date()), "overall": round(float(row["overall"]), 3)}
        for c in CATEGORIES:
            v = row[c]
            entry[c] = None if pd.isna(v) else round(float(v), 3)
        out.append(entry)
    return {"seller_id": seller_id, "size_tier": _load()["tiers"][seller_id], "history": out}


def get_metric_detail(seller_id: str, category: str, as_of_date: str) -> dict:
    """The actual metric values behind one category for one seller.

    For each metric in the category, returns the average over the last 7 days, the
    average over the 21 days before that, and the performance level (Healthy,
    Watch, At-risk or Critical) that the change earns. Use it as evidence.

    Args:
        seller_id: for example "S0042".
        category: one of Marketing, Logistics, Customer Experience, Seller support,
            Finance, Inventory.
        as_of_date: the date to assess, formatted YYYY-MM-DD.
    """
    if category not in METRICS:
        return {"error": f"category must be one of {CATEGORIES}"}
    s = _load()
    d = pd.Timestamp(as_of_date)
    df = s["df"]

    def avg(end, days):
        w = df[(df.seller_id == seller_id) & (df.date > end - pd.Timedelta(days=days)) & (df.date <= end)]
        return w[ALL_METRICS].mean()

    recent, base = avg(d, 7), avg(d - pd.Timedelta(days=7), 21)
    change = worsening(df, d).loc[seller_id]
    cuts = s["th"][s["tiers"][seller_id]]
    rows = []
    for m in METRICS[category]:
        lvl = metric_level(change[m], cuts[m])
        rows.append({"metric": m,
                     "last_7_days": None if pd.isna(recent[m]) else round(float(recent[m]), 3),
                     "previous_21_days": None if pd.isna(base[m]) else round(float(base[m]), 3),
                     "higher_is_better": m in HIGHER_IS_BETTER,
                     "level": "no data" if lvl is None else LEVEL_NAMES[lvl]})
    return {"seller_id": seller_id, "category": category, "as_of_date": str(d.date()), "metrics": rows}


TOOLS = [get_flagged_candidates, get_seller_history, get_metric_detail]


if __name__ == "__main__":
    import json
    r = get_flagged_candidates("2026-09-08")
    print(f"{r['total_candidates']} candidates: {r['confirmed']} confirmed, {r['watch']} watch")
    print(json.dumps(r["candidates"][:2], indent=1))
    top = r["candidates"][0]
    print(json.dumps(get_seller_history(top["seller_id"], "2026-09-08", 2), indent=1)[:500])
    cat = list(top["now"])[0]
    print(json.dumps(get_metric_detail(top["seller_id"], cat, "2026-09-08"), indent=1))