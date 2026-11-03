"""
Synthetic INSTAMART (quick-commerce grocery) data for the Seller Health Monitoring Agent.

Instamart model: brand/vendor sellers supply SKUs to dark stores, and buyers get a
10-30 minute delivery. Compared with Meesho this means
  - high order volume per seller, multi-item baskets that mix several sellers
  - "late shipment" = the delivery promise was missed
  - cancellations mostly mean the item was unavailable
  - very few returns, but many refunds without a return (missing or damaged item)
  - buyer messages are live chat, so the SLA is 1 hour, not 12
  - perishables (fruit, veg, dairy) expire or get damaged far more often

Writes the same 10 tables as the Meesho generator (sellers, listings, listing_daily,
orders, order_items, order_payments, order_returns, order_reviews, complaints,
buyer_messages) plus ground_truth.csv, which is for EVALUATION ONLY - the
agents must never read it.

Also injects one platform-wide heavy-rain event (late deliveries spike for ALL
sellers for 3 days). It is a confounder: the agent should not flag sellers for it.
The rain dates are printed at the end and stored in RAIN_DAYS below.

Run:  python generate_instamart_data.py
Then: python aggregate_metrics.py instamart
"""
import os
import numpy as np
import pandas as pd

SEED = 23
N_SELLERS = 200
N_DAYS = 90
START = pd.Timestamp("2026-07-01")          # window: 2026-07-01 -> 2026-09-28
OUT = "instamart_data"
SLA_HOURS = 1                                # chat support: reply within 1 hour

# Daily order-item ranges per tier are Small 15-30, Medium 40-80, Large 100-160.
# At full size that is ~930k order items over 90 days. VOLUME_SCALE shrinks every
# tier by the same factor so the total stays near 400k (set 1.0 for full volume).
VOLUME_SCALE = 0.39

# Platform-wide heavy rain: +20 percentage points late delivery for every seller.
RAIN_DAYS = [49, 50, 51]                     # 0-indexed -> 2026-08-19 .. 2026-08-21
RAIN_LATE_BUMP = 0.20

rng = np.random.default_rng(SEED)
os.makedirs(OUT, exist_ok=True)
dates = pd.date_range(START, periods=N_DAYS)
S, D = N_SELLERS, N_DAYS

# ---------------------------------------------------------------- sellers
CATS = ["Fruits & Vegetables", "Dairy & Bakery", "Snacks & Beverages",
        "Staples", "Personal & Home Care"]
seller_cat = rng.choice(CATS, S, p=[0.22, 0.18, 0.25, 0.20, 0.15])
tier = rng.choice(["Small", "Medium", "Large"], S, p=[0.50, 0.35, 0.15])
tenure_days = rng.integers(30, 2000, S)
join_date = START - pd.to_timedelta(tenure_days, unit="D")
is_new = tenure_days < 180

base_vol = VOLUME_SCALE * np.where(
    tier == "Small", rng.uniform(15, 30, S),
    np.where(tier == "Medium", rng.uniform(40, 80, S), rng.uniform(100, 160, S)))
# number of SKUs (listings) each seller supplies to the dark stores
n_list = np.where(tier == "Small", rng.integers(5, 13, S),
          np.where(tier == "Medium", rng.integers(12, 26, S), rng.integers(25, 46, S)))

# Category-dependent baselines. Returns are rare in grocery; expiry/damage is
# much higher for perishables.
RETURN_BY_CAT = {"Fruits & Vegetables": 0.030, "Dairy & Bakery": 0.025,
                 "Snacks & Beverages": 0.020, "Staples": 0.020,
                 "Personal & Home Care": 0.030}
EXPIRY_BY_CAT = {"Fruits & Vegetables": 0.050, "Dairy & Bakery": 0.030,
                 "Snacks & Beverages": 0.005, "Staples": 0.005,
                 "Personal & Home Care": 0.005}

# Per-seller healthy baselines (each seller is a little different)
base_late = np.clip(rng.normal(0.088, 0.012, S) + 0.01 * is_new, 0.04, 0.20)   # missed promise
base_cancel = np.clip(rng.normal(0.045, 0.006, S), 0.02, 0.10)                  # item unavailable
base_return = np.clip(np.array([RETURN_BY_CAT[c] for c in seller_cat])
                      + rng.normal(0, 0.004, S), 0.005, 0.10)
base_refund_noret = np.clip(rng.normal(0.055, 0.008, S), 0.03, 0.09)           # missing/damaged
base_rating = rng.normal(4.53, 0.12, S) - 0.1 * is_new
base_payout = rng.beta(1.5, 45, S) + 0.01
base_resp = rng.lognormal(np.log(0.4), 0.2, S)                                  # hours, chat
base_quality = np.clip(rng.normal(88, 3, S), 78, 96)                            # high and stable
base_roas = rng.lognormal(np.log(2.5), 0.25, S)                                 # moderate
base_stockout = np.clip(rng.normal(0.09, 0.012, S), 0.05, 0.14)
base_expiry = (np.array([EXPIRY_BY_CAT[c] for c in seller_cat])
               * rng.lognormal(0, 0.15, S))

seller_ids = np.array([f"S{i+1:04d}" for i in range(S)])
sellers = pd.DataFrame({
    "seller_id": seller_ids, "category": seller_cat,
    "join_date": join_date.strftime("%Y-%m-%d"), "size_tier": tier})

# -------------------------------------------- severity matrices (the decline)
# sev[category][seller, day] in [0, 1]: 0 = healthy, 1 = fully degraded
KEYS = ["marketing", "logistics", "support", "cx", "finance", "inventory"]
NICE = {"marketing": "Marketing", "logistics": "Logistics", "support": "Seller support",
        "cx": "Customer Experience", "finance": "Finance", "inventory": "Inventory"}
sev = {k: np.zeros((S, D)) for k in KEYS}
day = np.arange(D)

# Instamart: Inventory and Logistics dominate (40 declining sellers in total)
primary_plan = (["inventory"] * 12 + ["logistics"] * 12 + ["marketing"] * 5 +
                ["cx"] * 5 + ["finance"] * 3 + ["support"] * 3)
# Knock-on problems: a stocked-out seller starts missing delivery promises;
# a seller with delivery problems gets swamped in support chats.
SECONDARY = {"inventory": ["logistics"], "logistics": ["support"], "marketing": ["cx"],
             "cx": ["support"], "finance": ["support"], "support": ["cx"]}

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
    start = int(rng.integers(25, 56))
    sudden = rng.random() < 0.25                       # ~25% sudden, 75% gradual
    length = int(rng.integers(3, 8)) if sudden else int(rng.integers(14, 29))
    sev[prim][s] = np.maximum(sev[prim][s], ramp(start, length))
    row = {"is_at_risk": 1, "primary_category": NICE[prim],
           "decline_start_date": dates[start].strftime("%Y-%m-%d"),
           "full_decline_date": dates[min(start + length, D - 1)].strftime("%Y-%m-%d"),
           "decline_speed": "sudden" if sudden else "gradual"}
    if rng.random() < 0.25:  # a knock-on problem appears later
        sec = str(rng.choice(SECONDARY[prim]))
        sstart = min(start + int(rng.integers(10, 21)), D - 15)
        sev[sec][s] = np.maximum(sev[sec][s], ramp(sstart, int(rng.integers(10, 21)), 0.8))
        row["secondary_category"] = NICE[sec]
    for k, v in row.items():
        gt.loc[s, k] = v

# Healthy sellers that have a short, recovering dip (false-alarm traps)
healthy = np.setdiff1d(np.arange(S), at_risk)
for s in rng.choice(healthy, 12, replace=False):
    cat = str(rng.choice(KEYS))
    st = int(rng.integers(20, 70))
    bump = np.zeros(D)
    w = np.arange(7)
    bump[st:st + 7] = (0.7 * (1 - np.abs(w - 3) / 4))[: len(bump[st:st + 7])]
    sev[cat][s] = np.maximum(sev[cat][s], bump)
    gt.loc[s, "temporary_dip_category"] = NICE[cat]
    gt.loc[s, "dip_start_date"] = dates[st].strftime("%Y-%m-%d")
    gt.loc[s, "dip_end_date"] = dates[min(st + 6, D - 1)].strftime("%Y-%m-%d")

# Platform-wide rain shock, shape (D,). NOT a seller problem, so it is kept out
# of sev[] and out of ground_truth.
rain = np.zeros(D)
rain[RAIN_DAYS] = RAIN_LATE_BUMP

# ---------------------------------------------------------------- listings
list_seller = np.repeat(np.arange(S), n_list)
L = len(list_seller)
list_start = np.concatenate([[0], np.cumsum(n_list)[:-1]])
listing_ids = np.array([f"L{i+1:05d}" for i in range(L)])
listings = pd.DataFrame({"listing_id": listing_ids, "seller_id": seller_ids[list_seller]})

# ----------------------------------------------------------- listing_daily
vuln = rng.uniform(0.5, 1.2, L)                 # how hard a listing is hit by a decline
lst_off = rng.normal(0, 3, L)                   # small per-SKU offset: quality is stable
advertised = rng.random(L) < 0.5
sm = sev["marketing"][list_seller]               # L x D
si = sev["inventory"][list_seller]

# Marketing decline: listing quality drops, content flags appear, ads stop paying back
quality = (base_quality[list_seller][:, None] + lst_off[:, None]
           + rng.normal(0, 0.8, (L, D)) - 30 * sm * vuln[:, None])
quality = np.clip(quality, 0, 100).round(1)
content_flag = (rng.random((L, D)) < 0.005 + 0.25 * sm).astype(int)
roas = (base_roas[list_seller][:, None] * rng.lognormal(0, 0.25, (L, D))
        * (1 - 0.65 * sm * vuln[:, None]))
roas = np.clip(roas, 0.1, None).round(2)
roas[~advertised] = np.nan

# Inventory decline: more stockouts, and more expired/damaged stock
# (perishable sellers get hit harder because their baseline is higher)
stockout = (rng.random((L, D)) < base_stockout[list_seller][:, None] + 0.40 * si).astype(int)
p_dmg = base_expiry[list_seller][:, None] * (1 + 2 * si) + 0.04 * si
damaged = (rng.random((L, D)) < p_dmg).astype(int)

listing_daily = pd.DataFrame({
    "listing_id": np.repeat(listing_ids, D),
    "date": np.tile(dates.strftime("%Y-%m-%d"), L),
    "listing_quality_score": quality.ravel(),
    "content_flag": content_flag.ravel(),
    "ad_return": roas.ravel(),
    "is_stockout": stockout.ravel(),
    "is_expired_or_damaged": damaged.ravel()})

# ------------------------------------------------------------- order_items
wk = np.array([1.0, 1.0, 1.0, 1.0, 1.05, 1.12, 1.10])[dates.dayofweek]  # weekend lift
lam = (base_vol[:, None] * wk[None, :] * rng.lognormal(0, 0.10, (S, D))
       * (1 - 0.15 * sev["marketing"] - 0.30 * sev["inventory"]))
counts = rng.poisson(lam)
sidx = np.repeat(np.repeat(np.arange(S), D), counts.ravel())   # seller of each item
didx = np.repeat(np.tile(day, S), counts.ravel())              # day of each item
n = len(sidx)

def sv(k):
    """Severity of problem k for every order item (seller, day)."""
    return sev[k][sidx, didx]

lid_idx = list_start[sidx] + (rng.random(n) * n_list[sidx]).astype(int)

# Cancellation = item unavailable at the dark store -> driven by Inventory
p_cancel = np.clip(base_cancel[sidx] + 0.20 * sv("inventory") + 0.04 * sv("logistics"), 0, 0.9)
cancelled = rng.random(n) < p_cancel
delivered = ~cancelled

# Late = missed 10-30 min promise -> Logistics, plus the platform-wide rain
p_late = np.clip(base_late[sidx] + 0.45 * sv("logistics") + rain[didx], 0, 0.95)
late = delivered & (rng.random(n) < p_late)

p_ret = np.clip(base_return[sidx] + 0.06 * sv("cx") + 0.03 * sv("marketing") + 0.01 * late, 0, 0.95)
returned = delivered & (rng.random(n) < p_ret)

# Refunds: almost every return, most cancellations, and - common in grocery -
# delivered items refunded without a return (missing / damaged in the bag)
p_ref_noret = np.clip(base_refund_noret[sidx] + 0.10 * sv("finance")
                      + 0.06 * sv("cx") + 0.04 * sv("inventory"), 0, 0.9)
refunded = ((returned & (rng.random(n) < 0.95)) | (cancelled & (rng.random(n) < 0.6))
            | (delivered & ~returned & (rng.random(n) < p_ref_noret)))
payout_delayed = (~cancelled) & (rng.random(n) < np.clip(base_payout[sidx] + 0.35 * sv("finance"), 0, 0.9))

# Reviews are rare (~15%). Late deliveries and Logistics problems pull ratings down.
has_review = delivered & (rng.random(n) < 0.14 + 0.10 * returned)
mu = (base_rating[sidx] - 1.0 * late - 1.1 * returned - 1.4 * sv("cx")
      - 0.5 * sv("marketing") - 0.3 * sv("logistics"))
rating = np.clip(np.rint(rng.normal(mu, 0.85)), 1, 5).astype(int)

p_cmp = (0.008 + 0.06 * late + 0.06 * returned + 0.10 * sv("cx")
         + 0.04 * sv("logistics") + 0.02 * sv("inventory"))
has_cmp = delivered & (rng.random(n) < p_cmp)

# Chat messages: mostly "where is my order?" -> late items generate many more
p_msg = 0.08 + 0.25 * late + 0.10 * cancelled
has_msg = rng.random(n) < p_msg
# median ~0.4 h; sigma 0.68 (+ seller spread) puts ~90% under the 1-hour SLA
resp_h = rng.lognormal(np.log(base_resp[sidx] * (1 + 3.5 * sv("support"))), 0.68)

# ------------------------------------------- group items into baskets (orders)
# A quick-commerce basket holds 1-5 items, usually from DIFFERENT brand sellers.
# Items are shuffled within each day and cut into baskets, so every order
# belongs to exactly one date.
order_key = np.zeros(n, dtype=np.int64)
BASKET_SIZES, BASKET_P = [1, 2, 3, 4, 5], [0.35, 0.25, 0.20, 0.12, 0.08]
for d in range(D):
    ids = rng.permutation(np.where(didx == d)[0])
    sizes = rng.choice(BASKET_SIZES, len(ids), p=BASKET_P)        # more than enough
    basket = np.repeat(np.arange(len(sizes)), sizes)[: len(ids)]
    order_key[ids] = d * 10_000_000 + basket

items = pd.DataFrame({
    "k": order_key, "day": didx, "seller": sidx, "listing": lid_idx,
    "is_late_shipment": late.astype(int), "is_cancelled": cancelled.astype(int),
    "ret": returned, "ref": refunded, "pd": payout_delayed, "rev": has_review,
    "rating": rating, "cmp": has_cmp, "msg": has_msg, "resp": resp_h})
items = items.sort_values(["day", "k"], kind="stable").reset_index(drop=True)

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
print(f"Rain event (platform-wide, not in ground truth): "
      f"{dates[RAIN_DAYS[0]].date()} -> {dates[RAIN_DAYS[-1]].date()}, late +{RAIN_LATE_BUMP:.0%}")
