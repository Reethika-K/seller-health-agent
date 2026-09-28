import pandas as pd
from pathlib import Path
from monitoring_agent import compute_health_score, compute_trends, classify_status

STATUS_RANK = {"Critical": 0, "At-Risk": 1, "Watch": 2, "Healthy": 3, "Insufficient Data": 4}
DATA_DIR = Path("data")


def run_weekly_triage(df: pd.DataFrame, week_end_day: int) -> pd.DataFrame:
    snap = df[df["day"] == week_end_day].copy()
    snap["status"] = [classify_status(h, t) for h, t in zip(snap["health_7d"], snap["trend_drop"])]
    return snap


def main():
    raw = pd.read_csv(DATA_DIR / "seller_health_daily.csv")
    scored = compute_trends(compute_health_score(raw))

    checkpoint_days = list(range(6, 90, 7))  # end of each 7-day week: day 6, 13, 20, ... 83
    log_rows = []

    for week_end_day in checkpoint_days:
        snap = run_weekly_triage(scored, week_end_day)
        counts = snap["status"].value_counts().to_dict()
        total = len(snap)
        flagged = total - counts.get("Healthy", 0) - counts.get("Insufficient Data", 0)
        log_rows.append({
            "week_end_day": week_end_day,
            "total_sellers": total,
            "healthy": counts.get("Healthy", 0),
            "watch": counts.get("Watch", 0),
            "at_risk": counts.get("At-Risk", 0),
            "critical": counts.get("Critical", 0),
            "pct_flagged": flagged / total,
        })
        print(f"Day {week_end_day:>2}: {total} sellers scanned -> {flagged} flagged "
              f"({flagged/total:.1%}), {total - flagged} screened out")

    triage_log = pd.DataFrame(log_rows)
    triage_log.to_csv(DATA_DIR / "triage_log.csv", index=False)

    # final shortlist: the most recent week, worst sellers first
    latest = run_weekly_triage(scored, checkpoint_days[-1])
    shortlist = latest[latest["status"].isin(["Critical", "At-Risk", "Watch"])].copy()
    shortlist["status_rank"] = shortlist["status"].map(STATUS_RANK)
    shortlist = shortlist.sort_values(["status_rank", "health_7d"])
    cols = ["seller_id", "category", "size_tier", "status", "health_7d", "health_30d",
            "trend_drop", "late_rate", "return_rate", "avg_rating", "sla_pct", "complaints",
            "is_at_risk_seller", "decline_start_day"]
    shortlist[cols].to_csv(DATA_DIR / "flagged_shortlist.csv", index=False)

    # quick sanity check against the ground truth you injected in Week 2
    # (this is NOT the formal evaluation - that's Week 7 - just an early gut check)
    truly_declining = latest[(latest["is_at_risk_seller"]) & (latest["decline_start_day"] <= latest["day"])]
    flagged_ids = set(shortlist["seller_id"])
    declining_ids = set(truly_declining["seller_id"])
    caught = len(flagged_ids & declining_ids)
    print(f"\nFinal week ({checkpoint_days[-1]}): {len(shortlist)} sellers flagged, "
          f"{len(latest) - len(shortlist)} screened out ({(len(latest)-len(shortlist))/len(latest):.1%})")
    print(f"Of {len(declining_ids)} sellers actually declining by this point, "
          f"{caught} were caught by the flag ({caught/max(len(declining_ids),1):.1%})")
    print(f"Saved: data/triage_log.csv, data/flagged_shortlist.csv")


if __name__ == "__main__":
    main()