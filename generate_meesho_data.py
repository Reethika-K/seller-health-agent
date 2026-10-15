"""
Synthetic Meesho marketplace data for the Seller Health Monitoring Agent.

Writes the 10 tables of the relational model (sellers, listings, listing_daily,
orders, order_items, order_payments, order_returns, order_reviews, complaints,
buyer_messages) plus ground_truth.csv, which is for EVALUATION ONLY - the
agents must never read it.

Run:  python generate_meesho_data.py
"""
import os
import numpy as np
import pandas as pd

SEED = 42
N_SELLERS = 200
N_DAYS = 90
START = pd.Timestamp("2026-07-01")
OUT = "meesho_data"
SLA_HOURS = 12

rng = np.random.default_rng(SEED)
os.makedirs(OUT, exist_ok=True)
dates = pd.date_range(START, periods=N_DAYS)
S, D = N_SELLERS, N_DAYS

# ---------------------------------------------------------------- sellers
CATS = ["Fashion", "Home & Kitchen", "Beauty & Personal Care",
        "Electronics Accessories", "Kids & Toys"]
seller_cat = rng.choice(CATS, S, p=[0.38, 0.22, 0.15, 0.13, 0.12])
tier = rng.choice(["Small", "Medium", "Large"], S, p=[0.60, 0.30, 0.10])
tenure_days = rng.integers(30, 2000, S)
join_date = START - pd.to_timedelta(tenure_days, unit="D")
is_new = tenure_days < 180

base_vol = np.where(tier == "Small", rng.uniform(3, 8, S),
            np.where(tier == "Medium", rng.uniform(10, 25, S), rng.uniform(30, 60, S)))
n_list = np.where(tier == "Small", rng.integers(4, 11, S),
          np.where(tier == "Medium", rng.integers(10, 26, S), rng.integers(25, 51, S)))

RETURN_BY_CAT = {"Fashion": 0.22, "Home & Kitchen": 0.10, "Beauty & Personal Care": 0.09,
                 "Electronics Accessories": 0.13, "Kids & Toys": 0.11}
base_late = rng.beta(2, 32, S) + 0.01 + 0.03 * is_new
base_cancel = rng.beta(2, 60, S) + 0.005
base_return = np.clip(np.array([RETURN_BY_CAT[c] for c in seller_cat]) + rng.normal(0, 0.03, S), 0.02, 0.5)
base_rating = rng.normal(4.4, 0.2, S) - 0.2 * is_new
base_payout = rng.beta(1.5, 45, S) + 0.01
base_resp = rng.lognormal(np.log(4.5), 0.35, S)
base_quality = np.clip(rng.normal(76, 7, S), 50, 92)
base_roas = rng.lognormal(np.log(3.2), 0.3, S)

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

# Meesho: Marketing is the dominant failure mode, then Logistics, then CX
primary_plan = (["marketing"] * 14 + ["logistics"] * 10 + ["cx"] * 8 +
                ["finance"] * 4 + ["support"] * 4 + ["inventory"] * 4)
SECONDARY = {"marketing": ["cx"], "logistics": ["support", "cx"], "cx": ["support"],
             "finance": ["support"], "support": ["cx"], "inventory": ["logistics"]}

at_risk = rng.choice(S, len(primary_plan), replace=False)
rng.shuffle(primary_plan)
gt = pd.DataFrame({"seller_id": seller_ids, "is_at_risk": 0, "primary_category": "",
                   "secondary_category": "", "decline_start_date": "",
                   "full_decline_date": "", "decline_speed": "",
                   "temporary_dip_category": "", "dip_start_date": "", "dip_end_date": ""})

def ramp(start, length, peak=1.0):
    return peak * np.clip((day - start) / length, 0, 1)

for s, prim in zip(at_risk, primary_plan):
    start = int(rng.integers(25, 56))
    sudden = rng.random() < 0.25
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

# ---------------------------------------------------------------- listings
list_seller = np.repeat(np.arange(S), n_list)
L = len(list_seller)
list_start = np.concatenate([[0], np.cumsum(n_list)[:-1]])
listing_ids = np.array([f"L{i+1:05d}" for i in range(L)])
listings = pd.DataFrame({"listing_id": listing_ids, "seller_id": seller_ids[list_seller]})

# ----------------------------------------------------------- listing_daily
vuln = rng.uniform(0.5, 1.2, L)                 # how hard a listing is hit by a decline
lst_off = rng.normal(0, 6, L)
advertised = rng.random(L) < 0.6
sm = sev["marketing"][list_seller]               # L x D
si = sev["inventory"][list_seller]
quality = (base_quality[list_seller][:, None] + lst_off[:, None]
           + rng.normal(0, 1.2, (L, D)) - 38 * sm * vuln[:, None])
quality = np.clip(quality, 0, 100).round(1)
content_flag = (rng.random((L, D)) < 0.02 + 0.30 * sm).astype(int)
roas = (base_roas[list_seller][:, None] * rng.lognormal(0, 0.25, (L, D))
        * (1 - 0.65 * sm * vuln[:, None]))
roas = np.clip(roas, 0.1, None).round(2)
roas[~advertised] = np.nan
stockout = (rng.random((L, D)) < 0.03 + 0.35 * si).astype(int)
damaged = (rng.random((L, D)) < 0.006 + 0.08 * si).astype(int)

listing_daily = pd.DataFrame({
    "listing_id": np.repeat(listing_ids, D),
    "date": np.tile(dates.strftime("%Y-%m-%d"), L),
    "listing_quality_score": quality.ravel(),
    "content_flag": content_flag.ravel(),
    "ad_return": roas.ravel(),
    "is_stockout": stockout.ravel(),
    "is_expired_or_damaged": damaged.ravel()})

# ------------------------------------------------------------- order_items
wk = np.array([1.0, 1.0, 1.0, 1.0, 1.05, 1.12, 1.10])[dates.dayofweek]
lam = (base_vol[:, None] * wk[None, :] * rng.lognormal(0, 0.10, (S, D))
       * (1 - 0.35 * sev["marketing"] - 0.25 * sev["inventory"]))
counts = rng.poisson(lam)
sidx = np.repeat(np.repeat(np.arange(S), D), counts.ravel())
didx = np.repeat(np.tile(day, S), counts.ravel())
n = len(sidx)

def sv(k):
    return sev[k][sidx, didx]

lid_idx = list_start[sidx] + (rng.random(n) * n_list[sidx]).astype(int)

p_cancel = np.clip(base_cancel[sidx] + 0.18 * sv("logistics") + 0.12 * sv("inventory"), 0, 0.9)
cancelled = rng.random(n) < p_cancel
delivered = ~cancelled
p_late = np.clip(base_late[sidx] + 0.40 * sv("logistics"), 0, 0.9)
late = delivered & (rng.random(n) < p_late)

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