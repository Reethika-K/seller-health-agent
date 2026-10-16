"""
Synthetic NYKAA marketplace data for the Seller Health Monitoring Agent.

Writes the 10 tables of the relational model (sellers, listings, listing_daily,
orders, order_items, order_payments, order_returns, order_reviews, complaints,
buyer_messages) plus ground_truth.csv, which is for EVALUATION ONLY - the
agents must never read it.

Same machinery as generate_meesho_data.py (seeded numpy, per-seller-per-day
"severity" matrices that inject declines, hidden ground truth). What differs is
the *world* it simulates: a beauty & personal-care marketplace, where
  - hygiene products are rarely returned (low return rate),
  - shelf life matters (expiry / damage, stockouts of bestsellers),
  - Marketing (inaccurate claims, weak ads) and Inventory are the dominant
    failure modes,
  - and there is ONE platform-wide sale event that lifts every seller's order
    volume for 5 days. That is a confounder, not a decline: a good agent
    should not flag it.

Run:  python generate_nykaa_data.py
Then: python aggregate_metrics.py nykaa
"""
import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ settings
SEED = 11                       # Meesho used 42; a different seed = a different world
N_SELLERS = 200
N_DAYS = 90
START = pd.Timestamp("2026-07-01")   # 90 days -> ends 2026-09-28
OUT = "nykaa_data"
SLA_HOURS = 12                  # a buyer message answered within 12h meets the SLA

# Platform-wide sale event (confounder). Every seller's order volume goes up
# by 40% for 5 days starting on day 45 (2026-08-15 .. 2026-08-19).
SALE_START_DAY = 45
SALE_LEN = 5
SALE_BOOST = 1.40

rng = np.random.default_rng(SEED)
os.makedirs(OUT, exist_ok=True)
dates = pd.date_range(START, periods=N_DAYS)
S, D = N_SELLERS, N_DAYS

# ------------------------------------------------------------------- sellers
CATS = ["Skincare", "Makeup", "Haircare", "Fragrance", "Personal Care & Wellness"]
seller_cat = rng.choice(CATS, S, p=[0.30, 0.28, 0.17, 0.10, 0.15])
tier = rng.choice(["Small", "Medium", "Large"], S, p=[0.35, 0.45, 0.20])   # Nykaa mix
tenure_days = rng.integers(30, 2000, S)
join_date = START - pd.to_timedelta(tenure_days, unit="D")
is_new = tenure_days < 180      # sellers under ~6 months old are a bit shakier

# Bigger tiers sell more and list more products
base_vol = np.where(tier == "Small", rng.uniform(3, 8, S),
            np.where(tier == "Medium", rng.uniform(10, 25, S), rng.uniform(30, 60, S)))
n_list = np.where(tier == "Small", rng.integers(4, 11, S),
          np.where(tier == "Medium", rng.integers(10, 26, S), rng.integers(25, 51, S)))

# ---- Healthy baselines: each seller gets their own "normal" level.
# Return rate: hygiene products are rarely returned, so ~6-8% (Meesho fashion was ~22%).
RETURN_BY_CAT = {"Skincare": 0.070, "Makeup": 0.080, "Haircare": 0.065,
                 "Fragrance": 0.075, "Personal Care & Wellness": 0.060}
base_late = rng.beta(2, 45, S) + 0.005 + 0.02 * is_new        # ~5% late shipments
base_cancel = rng.beta(2, 90, S) + 0.003                       # ~2.5% cancellations
base_return = np.clip(np.array([RETURN_BY_CAT[c] for c in seller_cat])
                      + rng.normal(0, 0.01, S), 0.02, 0.20)
base_rating = rng.normal(4.76, 0.15, S) - 0.2 * is_new         # observed avg lands ~4.4
base_payout = rng.beta(1.5, 45, S) + 0.01                      # ~4% delayed payouts
base_resp = rng.lognormal(np.log(3.8), 0.35, S)                # median hours -> ~95% SLA met
base_quality = np.clip(rng.normal(80, 6, S), 55, 95)           # listing quality ~80
base_roas = rng.lognormal(np.log(3.2), 0.3, S)                 # ad return (ROAS)

seller_ids = np.array([f"S{i+1:04d}" for i in range(S)])
sellers = pd.DataFrame({
    "seller_id": seller_ids, "category": seller_cat,
    "join_date": join_date.strftime("%Y-%m-%d"), "size_tier": tier})

# -------------------------------------------- severity matrices (the decline)
# sev[category][seller, day] is in [0, 1]: 0 = healthy, 1 = fully degraded.
# Every metric formula below adds (effect size x severity) to its healthy baseline,
# which is how one "cause" spreads into the right group of metrics.
KEYS = ["marketing", "logistics", "support", "cx", "finance", "inventory"]
NICE = {"marketing": "Marketing", "logistics": "Logistics", "support": "Seller support",
        "cx": "Customer Experience", "finance": "Finance", "inventory": "Inventory"}
sev = {k: np.zeros((S, D)) for k in KEYS}
day = np.arange(D)

# 40 declining sellers. Nykaa: Marketing and Inventory dominate (13 each).
primary_plan = (["marketing"] * 13 + ["inventory"] * 13 + ["logistics"] * 5 +
                ["cx"] * 5 + ["finance"] * 2 + ["support"] * 2)
# A later "knock-on" problem. Marketing -> CX and Inventory -> Logistics are the
# Nykaa stories; the rest are carried over from the Meesho script.
SECONDARY = {"marketing": ["cx"], "inventory": ["logistics"],
             "logistics": ["support", "cx"], "cx": ["support"],
             "finance": ["support"], "support": ["cx"]}

at_risk = rng.choice(S, len(primary_plan), replace=False)
rng.shuffle(primary_plan)
gt = pd.DataFrame({"seller_id": seller_ids, "is_at_risk": 0, "primary_category": "",
                   "secondary_category": "", "decline_start_date": "",
                   "full_decline_date": "", "decline_speed": "",
                   "temporary_dip_category": "", "dip_start_date": "", "dip_end_date": ""})


def ramp(start, length, peak=1.0):
    """0 before `start`, rises linearly to `peak` over `length` days, then stays."""
    return peak * np.clip((day - start) / length, 0, 1)


for s, prim in zip(at_risk, primary_plan):
    start = int(rng.integers(25, 56))                 # decline begins somewhere in days 25-55
    sudden = rng.random() < 0.25                      # ~25% sudden, ~75% gradual
    length = int(rng.integers(3, 8)) if sudden else int(rng.integers(14, 29))
    sev[prim][s] = np.maximum(sev[prim][s], ramp(start, length))
    row = {"is_at_risk": 1, "primary_category": NICE[prim],
           "decline_start_date": dates[start].strftime("%Y-%m-%d"),
           "full_decline_date": dates[min(start + length, D - 1)].strftime("%Y-%m-%d"),
           "decline_speed": "sudden" if sudden else "gradual"}
    if rng.random() < 0.25:                           # ~25% get a later knock-on problem
        sec = str(rng.choice(SECONDARY[prim]))
        sstart = min(start + int(rng.integers(10, 21)), D - 15)
        sev[sec][s] = np.maximum(sev[sec][s], ramp(sstart, int(rng.integers(10, 21)), 0.8))
        row["secondary_category"] = NICE[sec]
    for k, v in row.items():
        gt.loc[s, k] = v

# 12 healthy sellers with a short (7-day) dip that recovers by itself.
# These are false-alarm traps: a good agent should NOT escalate them.
healthy = np.setdiff1d(np.arange(S), at_risk)
for s in rng.choice(healthy, 12, replace=False):
    cat = str(rng.choice(KEYS))
    st = int(rng.integers(20, 70))
    bump = np.zeros(D)
    w = np.arange(7)
    bump[st:st + 7] = (0.7 * (1 - np.abs(w - 3) / 4))[: len(bump[st:st + 7])]  # up then down
    sev[cat][s] = np.maximum(sev[cat][s], bump)
    gt.loc[s, "temporary_dip_category"] = NICE[cat]
    gt.loc[s, "dip_start_date"] = dates[st].strftime("%Y-%m-%d")
    gt.loc[s, "dip_end_date"] = dates[min(st + 6, D - 1)].strftime("%Y-%m-%d")

# ------------------------------------------------------------------ listings
list_seller = np.repeat(np.arange(S), n_list)          # which seller owns each listing
L = len(list_seller)
list_start = np.concatenate([[0], np.cumsum(n_list)[:-1]])   # first listing index per seller
listing_ids = np.array([f"L{i+1:05d}" for i in range(L)])
listings = pd.DataFrame({"listing_id": listing_ids, "seller_id": seller_ids[list_seller]})

# ----------------------------------------------------------- listing_daily
vuln = rng.uniform(0.5, 1.2, L)                  # how hard a listing is hit by a decline
lst_off = rng.normal(0, 6, L)                    # listing-level quality offset
advertised = rng.random(L) < 0.6                 # only 60% of listings run ads (others: NaN)
bestseller = rng.random(L) < 0.18                # ~18% bestsellers sell out far more often
sm = sev["marketing"][list_seller]               # shape L x D
si = sev["inventory"][list_seller]

# Marketing decline -> quality falls, content flags (inaccurate claims) rise, ad return falls
quality = (base_quality[list_seller][:, None] + lst_off[:, None]
           + rng.normal(0, 1.2, (L, D)) - 38 * sm * vuln[:, None])
quality = np.clip(quality, 0, 100).round(1)
content_flag = (rng.random((L, D)) < 0.004 + 0.35 * sm).astype(int)   # very low when healthy
roas = (base_roas[list_seller][:, None] * rng.lognormal(0, 0.25, (L, D))
        * (1 - 0.65 * sm * vuln[:, None]))
roas = np.clip(roas, 0.1, None).round(2)
roas[~advertised] = np.nan

# Inventory decline -> stockouts and expiry/damage rise.
# Healthy: bestsellers stock out ~12% of days, others ~3.5%  -> ~5% overall.
stock_base = np.where(bestseller, 0.12, 0.035)
stockout = (rng.random((L, D)) < stock_base[:, None] + 0.40 * si).astype(int)
damaged = (rng.random((L, D)) < 0.017 + 0.10 * si).astype(int)        # ~1.7% shelf-life losses

listing_daily = pd.DataFrame({
    "listing_id": np.repeat(listing_ids, D),
    "date": np.tile(dates.strftime("%Y-%m-%d"), L),
    "listing_quality_score": quality.ravel(),
    "content_flag": content_flag.ravel(),
    "ad_return": roas.ravel(),
    "is_stockout": stockout.ravel(),
    "is_expired_or_damaged": damaged.ravel()})

# ------------------------------------------------------------- order_items
# Expected orders per seller per day = size x weekday effect x SALE EVENT x noise,
# reduced when marketing / inventory decline.
wk = np.array([1.0, 1.0, 1.0, 1.0, 1.05, 1.12, 1.10])[dates.dayofweek]
sale = np.ones(D)
sale[SALE_START_DAY:SALE_START_DAY + SALE_LEN] = SALE_BOOST     # same for ALL sellers
lam = (base_vol[:, None] * wk[None, :] * sale[None, :] * rng.lognormal(0, 0.10, (S, D))
       * (1 - 0.35 * sev["marketing"] - 0.25 * sev["inventory"]))
counts = rng.poisson(lam)
sidx = np.repeat(np.repeat(np.arange(S), D), counts.ravel())    # seller of each order item
didx = np.repeat(np.tile(day, S), counts.ravel())               # day of each order item
n = len(sidx)


def sv(k):
    """Severity of category k for every order item (looked up by its seller and day)."""
    return sev[k][sidx, didx]


lid_idx = list_start[sidx] + (rng.random(n) * n_list[sidx]).astype(int)   # pick a listing

# Cancellations: logistics trouble and (Nykaa) stockouts cause them
p_cancel = np.clip(base_cancel[sidx] + 0.18 * sv("logistics") + 0.15 * sv("inventory"), 0, 0.9)
cancelled = rng.random(n) < p_cancel
delivered = ~cancelled
p_late = np.clip(base_late[sidx] + 0.40 * sv("logistics"), 0, 0.9)
late = delivered & (rng.random(n) < p_late)

# Returns: low baseline (hygiene!), pushed up by CX problems, mis-sold products, lateness
p_ret = np.clip(base_return[sidx] + 0.28 * sv("cx") + 0.12 * sv("marketing") + 0.06 * late, 0, 0.95)
returned = delivered & (rng.random(n) < p_ret)

refunded = ((returned & (rng.random(n) < 0.95)) | (cancelled & (rng.random(n) < 0.6))
            | (delivered & ~returned & (rng.random(n) < 0.005 + 0.10 * sv("finance"))))
payout_delayed = (~cancelled) & (rng.random(n) < np.clip(base_payout[sidx] + 0.35 * sv("finance"), 0, 0.9))

has_review = delivered & (rng.random(n) < 0.35 + 0.15 * returned)
mu = (base_rating[sidx] - 1.3 * late - 1.1 * returned - 1.4 * sv("cx") - 0.6 * sv("marketing"))
rating = np.clip(np.rint(rng.normal(mu, 0.85)), 1, 5).astype(int)

p_cmp = 0.010 + 0.04 * late + 0.06 * returned + 0.10 * sv("cx") + 0.03 * sv("logistics")
has_cmp = delivered & (rng.random(n) < p_cmp)

p_msg = 0.10 + 0.20 * late + 0.05 * cancelled
has_msg = rng.random(n) < p_msg
resp_h = rng.lognormal(np.log(base_resp[sidx] * (1 + 3.5 * sv("support"))), 0.6)

# ------------------------------------------- group items into orders (some
# orders contain items from two different sellers, as in Olist)
order_key = np.arange(n)
for d in range(D):
    ids = np.where(didx == d)[0]
    m = (int(0.10 * len(ids)) // 2) * 2
    if m < 2:
        continue
    pick = rng.choice(ids, m, replace=False)
    a, b = pick[0::2], pick[1::2]
    ok = sidx[a] != sidx[b]
    order_key[b[ok]] = order_key[a[ok]]

items = pd.DataFrame({
    "k": order_key, "day": didx, "seller": sidx, "listing": lid_idx,
    "is_late_shipment": late.astype(int), "is_cancelled": cancelled.astype(int),
    "ret": returned, "ref": refunded, "pd": payout_delayed, "rev": has_review,
    "rating": rating, "cmp": has_cmp, "msg": has_msg, "resp": resp_h})
items = items.sort_values(["day", "k"], kind="stable").reset_index(drop=True)

# ------------------------------------------------ build the 10 tables (IDs)
order_map = {k: i for i, k in enumerate(pd.unique(items["k"]))}
items["order_num"] = items["k"].map(order_map)
items["order_id"] = "ORD" + (items["order_num"] + 1).astype(str).str.zfill(7)
items["order_item_id"] = "OI" + (items.index + 1).astype(str).str.zfill(7)
items["seller_id"] = seller_ids[items["seller"].to_numpy()]
items["listing_id"] = listing_ids[items["listing"].to_numpy()]

orders = (items.drop_duplicates("order_id")[["order_id", "day"]]
          .assign(order_date=lambda x: dates[x["day"].to_numpy()].strftime("%Y-%m-%d"))
          [["order_id", "order_date"]])
order_items = items[["order_item_id", "order_id", "listing_id", "seller_id",
                     "is_late_shipment", "is_cancelled"]]


def child(mask, prefix, cols):
    """Rows of `items` where mask is True, plus a fresh ID column like PAY0000001."""
    sub = items[mask].reset_index(drop=True)
    out = pd.DataFrame({cols[0]: prefix + (sub.index + 1).astype(str).str.zfill(7)})
    return sub, out


sub, order_payments = child(np.ones(len(items), bool), "PAY", ["payment_id"])
order_payments["order_item_id"] = sub["order_item_id"]
order_payments["is_payout_delayed"] = sub["pd"].astype(int)
order_payments["is_refunded"] = sub["ref"].astype(int)

sub, order_returns = child(items["ret"].to_numpy(), "RET", ["return_id"])
order_returns["order_item_id"] = sub["order_item_id"]

sub, order_reviews = child(items["rev"].to_numpy(), "REV", ["review_id"])
order_reviews["order_item_id"] = sub["order_item_id"]
order_reviews["rating"] = sub["rating"]

sub, complaints = child(items["cmp"].to_numpy(), "CMP", ["complaint_id"])
complaints["order_item_id"] = sub["order_item_id"]

sub, buyer_messages = child(items["msg"].to_numpy(), "MSG", ["message_id"])
buyer_messages["order_id"] = sub["order_id"]
buyer_messages["seller_id"] = sub["seller_id"]
buyer_messages["response_time_hours"] = sub["resp"].round(2)
buyer_messages["is_sla_met"] = (sub["resp"] <= SLA_HOURS).astype(int)

# ------------------------------------------------------------------- write
tables = {"sellers": sellers, "listings": listings, "listing_daily": listing_daily,
          "orders": orders, "order_items": order_items, "order_payments": order_payments,
          "order_returns": order_returns, "order_reviews": order_reviews,
          "complaints": complaints, "buyer_messages": buyer_messages,
          "ground_truth": gt}
for name, df in tables.items():
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    print(f"{name:16s} {len(df):>8,} rows")
print(f"\nAt-risk sellers: {int(gt.is_at_risk.sum())} | temporary dips: "
      f"{int((gt.temporary_dip_category != '').sum())} | window: {dates[0].date()} -> {dates[-1].date()}")
print(f"Sale event (all sellers, +{round((SALE_BOOST - 1) * 100)}% orders): "
      f"{dates[SALE_START_DAY].date()} -> {dates[SALE_START_DAY + SALE_LEN - 1].date()}")
