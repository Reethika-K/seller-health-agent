import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)  # fixed seed = reproducible data every run

N_SELLERS = 300
N_DAYS = 90
CATEGORIES = ["Fashion", "Electronics", "Home & Kitchen", "Beauty", "Grocery", "Sports"]
SIZE_TIERS = ["Small", "Medium", "Large"]
SIZE_WEIGHTS = [0.6, 0.3, 0.1]
AT_RISK_FRACTION = 0.18

OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)


def base_order_volume(size_tier):
    if size_tier == "Small":
        return RNG.poisson(5)
    if size_tier == "Medium":
        return RNG.poisson(20)
    return RNG.poisson(80)


def make_seller(seller_idx):
    category = RNG.choice(CATEGORIES)
    size_tier = RNG.choice(SIZE_TIERS, p=SIZE_WEIGHTS)
    tenure_months = int(RNG.integers(1, 61))
    quality = float(np.clip(RNG.normal(0.85, 0.08), 0.5, 1.0))  # latent "how good is this seller", 0-1

    is_at_risk = RNG.random() < AT_RISK_FRACTION
    decline_start = int(RNG.integers(15, 51)) if is_at_risk else None
    decline_len = int(RNG.integers(20, 41)) if is_at_risk else None

    return {
        "seller_id": f"S{seller_idx:04d}",
        "category": category,
        "size_tier": size_tier,
        "tenure_months": tenure_months,
        "quality": quality,
        "is_at_risk": is_at_risk,
        "decline_start_day": decline_start,
        "decline_len_days": decline_len,
    }


def decline_progress(day, decline_start, decline_len):
    """0 before decline starts, ramps to 1 by the end of the decline window, stays at 1 after."""
    if decline_start is None or day < decline_start:
        return 0.0
    return float(np.clip((day - decline_start) / decline_len, 0, 1))


def simulate_seller_days(seller):
    rows = []
    quality = seller["quality"]

    base_late_rate = 0.02 + (1 - quality) * 0.06
    base_return_rate = 0.05 + (1 - quality) * 0.08
    if seller["category"] == "Fashion":
        base_return_rate += 0.03  # fashion runs naturally higher on returns (sizing)
    base_cancel_rate = 0.01 + (1 - quality) * 0.02
    base_rating = 3.8 + quality * 1.1
    base_response_hours = 2 + (1 - quality) * 6
    base_sla_pct = 0.98 - (1 - quality) * 0.25
    base_listing_quality = 70 + quality * 28

    rating_walk = base_rating
    full_decline_day = (
        seller["decline_start_day"] + seller["decline_len_days"] if seller["is_at_risk"] else None
    )

    for day in range(N_DAYS):
        prog = decline_progress(day, seller["decline_start_day"], seller["decline_len_days"])

        late_rate = np.clip(base_late_rate + prog * 0.10, 0, 0.6)
        return_rate = np.clip(base_return_rate + prog * 0.14, 0, 0.7)
        cancel_rate = np.clip(base_cancel_rate + prog * 0.04, 0, 0.4)
        response_hours = base_response_hours + prog * 10
        sla_pct = np.clip(base_sla_pct - prog * 0.35, 0.1, 1.0)
        listing_quality = float(np.clip(base_listing_quality - prog * 15 + RNG.normal(0, 1.5), 0, 100))

        orders = int(max(base_order_volume(seller["size_tier"]), 0))
        late_shipments = int(RNG.binomial(orders, late_rate)) if orders > 0 else 0
        cancellations = int(RNG.binomial(orders, cancel_rate)) if orders > 0 else 0
        returns = int(RNG.binomial(orders, return_rate)) if orders > 0 else 0

        target_rating = base_rating - prog * 0.9
        rating_walk += (target_rating - rating_walk) * 0.08 + RNG.normal(0, 0.03)
        rating_walk = float(np.clip(rating_walk, 1.0, 5.0))

        reviews_today = int(RNG.poisson(orders * 0.15)) if orders > 0 else 0
        neg_review_prob = np.clip(0.05 + prog * 0.25, 0, 0.8)
        negative_reviews = int(RNG.binomial(reviews_today, neg_review_prob)) if reviews_today > 0 else 0

        complaints = int(RNG.poisson(0.1 + prog * 1.2))
        content_flags = int(RNG.poisson(0.05 + prog * 0.3))

        rows.append({
            "seller_id": seller["seller_id"],
            "category": seller["category"],
            "size_tier": seller["size_tier"],
            "tenure_months": seller["tenure_months"],
            "day": day,
            "orders": orders,
            "late_shipments": late_shipments,
            "cancellations": cancellations,
            "returns": returns,
            "avg_rating": round(rating_walk, 2),
            "reviews_count": reviews_today,
            "negative_reviews": negative_reviews,
            "complaints": complaints,
            "avg_response_hours": round(response_hours, 2),
            "sla_pct": round(sla_pct, 3),
            "listing_quality_score": round(listing_quality, 1),
            "content_accuracy_flags": content_flags,
            "is_at_risk_seller": seller["is_at_risk"],
            "decline_start_day": seller["decline_start_day"],
            "full_decline_day": full_decline_day,
        })
    return rows


def main():
    all_rows = []
    for i in range(N_SELLERS):
        seller = make_seller(i)
        all_rows.extend(simulate_seller_days(seller))

    df = pd.DataFrame(all_rows)
    out_path = OUT_DIR / "seller_health_daily.csv"
    df.to_csv(out_path, index=False)

    n_sellers = df["seller_id"].nunique()
    at_risk_share = df.groupby("seller_id")["is_at_risk_seller"].first().mean()
    print(f"Generated {n_sellers} sellers x {N_DAYS} days = {len(df)} rows")
    print(f"At-risk sellers: {at_risk_share:.1%}")
    print(f"Saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()