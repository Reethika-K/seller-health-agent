"""
Synthetic Instamart (quick-commerce grocery) data for the Seller Health Monitoring Agent.

Same relational model and mechanics as generate_meesho_data.py:
  10 tables  -> sellers, listings, listing_daily, orders, order_items, order_payments,
                order_returns, order_reviews, complaints, buyer_messages
  + ground_truth.csv, which is for EVALUATION ONLY - the agents must never read it.

What is different is the *business*: Instamart sells groceries from dark stores with a
10-30 minute delivery promise. Brand/vendor sellers supply items to those dark stores, so:
  - volumes per seller are high, returns are rare, but "refund without a return"
    (missing / damaged item) is common
  - stockouts and expiry/damage are everyday problems (worst for fresh produce)
  - buyer messages are live chat, so the SLA is 1 hour, not 12
  - Inventory and Logistics are the dominant failure modes
  - a 3-day heavy-rain event delays EVERY seller - a confounder the agent should not flag

Run:  python generate_instamart_data.py
Then: python aggregate_metrics.py instamart
"""
import os
import numpy as np
import pandas as pd

SEED = 23
N_SELLERS = 100
N_DAYS = 90
START = pd.Timestamp("2026-07-01")          # window: 2026-07-01 -> 2026-09-28
OUT = "instamart_data"
SLA_HOURS = 1                               # chat support must reply within 1 hour

# Platform-wide heavy rain: every seller misses more delivery promises for 3 days.
RAIN_START_DAY = 49                         # day index 49 = 2026-08-19
RAIN_DAYS = 3
RAIN_LATE_BUMP = 0.20                       # +20 percentage points late shipment

rng = np.random.default_rng(SEED)
os.makedirs(OUT, exist_ok=True)
dates = pd.date_range(START, periods=N_DAYS)
S, D = N_SELLERS, N_DAYS
day = np.arange(D)

# ---------------------------------------------------------------- sellers
CATS = ["Fruits & Vegetables", "Dairy & Bakery", "Snacks & Beverages",
        "Staples", "Personal & Home Care"]
seller_cat = rng.choice(CATS, S, p=[0.22, 0.20, 0.22, 0.20, 0.16])

# Exact 50/35/15 size mix (shuffled), so total volume is predictable with only 100 sellers.
tier = rng.permutation(np.repeat(["Small", "Medium", "Large"], [50, 35, 15]))
tenure_days = rng.integers(30, 2000, S)
join_date = START - pd.to_timedelta(tenure_days, unit="D")
is_new = tenure_days < 180                  # new sellers are a bit less reliable

# Average order items per day for each seller (before weekday effects / declines).
base_vol = np.where(tier == "Small", rng.uniform(15, 30, S),
            np.where(tier == "Medium", rng.uniform(40, 80, S), rng.uniform(100, 160, S)))
# Number of SKUs (listings) each seller supplies to the dark stores.
n_list = np.where(tier == "Small", rng.integers(6, 13, S),
          np.where(tier == "Medium", rng.integers(12, 26, S), rng.integers(25, 46, S)))

# Category-level baselines. Fresh produce returns / spoils more; packaged goods barely spoil.
RETURN_BY_CAT = {"Fruits & Vegetables": 0.030, "Dairy & Bakery": 0.025,
                 "Snacks & Beverages": 0.020, "Staples": 0.020, "Personal & Home Care": 0.030}
EXPIRY_BY_CAT = {"Fruits & Vegetables": 0.050, "Dairy & Bakery": 0.030,
                 "Snacks & Beverages": 0.005, "Staples": 0.005, "Personal & Home Care": 0.005}

# Healthy per-seller baselines (each seller gets its own value around the target).
base_late = rng.beta(10, 100, S) + 0.02 * is_new          # ~9% missed delivery promise
base_cancel = rng.beta(9, 195, S) + 0.002                 # ~4.5% item unavailable
base_return = np.clip(np.array([RETURN_BY_CAT[c] for c in seller_cat])
                      + rng.normal(0, 0.005, S), 0.005, 0.10)   # ~2-3%
base_refund_nr = rng.beta(11, 190, S)                     # ~5.5% refund WITHOUT a return
base_rating = rng.normal(4.62, 0.12, S) - 0.15 * is_new   # tuned so observed avg ~4.2
base_payout = rng.beta(1.5, 45, S) + 0.01                 # payout delays, as Meesho
base_resp = rng.lognormal(np.log(0.4), 0.2, S)            # median chat reply ~0.4 h
base_quality = np.clip(rng.normal(88, 3.5, S), 75, 97)    # listings are high quality
base_roas = rng.lognormal(np.log(2.6), 0.25, S)           # moderate ad return
base_stockout = np.clip(rng.normal(0.09, 0.008, S), 0.06, 0.12)   # ~8-10%
base_expiry = (np.array([EXPIRY_BY_CAT[c] for c in seller_cat])
               * rng.lognormal(0, 0.10, S))

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

# Instamart: Inventory and Logistics dominate (40 declining sellers in total)
primary_plan = (["inventory"] * 12 + ["logistics"] * 12 + ["marketing"] * 5 +
                ["cx"] * 5 + ["finance"] * 3 + ["support"] * 3)
SECONDARY = {"inventory": "logistics", "logistics": "support"}   # knock-on problems
n_decl = len(primary_plan)

at_risk = rng.choice(S, n_decl, replace=False)
rng.shuffle(primary_plan)
# Exactly 25% sudden (10 of 40), and 10 knock-ons (25% of 40) picked among the
# Inventory/Logistics sellers, since those are the only ones with a knock-on path.
sudden_set = set(rng.choice(n_decl, n_decl // 4, replace=False))
eligible = [j for j, p in enumerate(primary_plan) if p in SECONDARY]
knock_set = set(rng.choice(eligible, n_decl // 4, replace=False))

gt = pd.DataFrame({"seller_id": seller_ids, "is_at_risk": 0, "primary_category": "",
                   "secondary_category": "", "decline_start_date": "",
                   "full_decline_date": "", "decline_speed": "",
                   "temporary_dip_category": "", "dip_start_date": "", "dip_end_date": ""})

def ramp(start, length, peak=1.0):
    """0 before `start`, rises linearly to `peak` over `length` days, then stays there."""
    return peak * np.clip((day - start) / length, 0, 1)

for j, (s, prim) in enumerate(zip(at_risk, primary_plan)):
    start = int(rng.integers(25, 56))
    sudden = j in sudden_set
    length = int(rng.integers(3, 8)) if sudden else int(rng.integers(14, 29))
    sev[prim][s] = np.maximum(sev[prim][s], ramp(start, length))
    row = {"is_at_risk": 1, "primary_category": NICE[prim],
           "decline_start_date": dates[start].strftime("%Y-%m-%d"),
           "full_decline_date": dates[min(start + length, D - 1)].strftime("%Y-%m-%d"),
           "decline_speed": "sudden" if sudden else "gradual"}
    if j in knock_set:  # a knock-on problem appears 10-20 days later
        sec = SECONDARY[prim]
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

# Platform-wide rain: NOT a seller problem, so it is not in ground_truth and not in sev.
rain = np.zeros(D)
rain[RAIN_START_DAY:RAIN_START_DAY + RAIN_DAYS] = RAIN_LATE_BUMP

# ---------------------------------------------------------------- listings
list_seller = np.repeat(np.arange(S), n_list)
L = len(list_seller)
list_start = np.concatenate([[0], np.cumsum(n_list)[:-1]])
listing_ids = np.array([f"L{i+1:05d}" for i in range(L)])
listings = pd.DataFrame({"listing_id": listing_ids, "seller_id": seller_ids[list_seller]})

# ----------------------------------------------------------- listing_daily
vuln = rng.uniform(0.5, 1.2, L)                 # how hard a listing is hit by a decline
lst_off = rng.normal(0, 3, L)                   # small per-SKU quality offset
advertised = rng.random(L) < 0.5
sm = sev["marketing"][list_seller]              # L x D
si = sev["inventory"][list_seller]

# Listing quality: high and stable (small daily noise); Marketing decline erodes it.
quality = (base_quality[list_seller][:, None] + lst_off[:, None]
           + rng.normal(0, 0.8, (L, D)) - 30 * sm * vuln[:, None])
quality = np.clip(quality, 0, 100).round(1)
content_flag = (rng.random((L, D)) < 0.008 + 0.25 * sm).astype(int)   # rare when healthy
roas = (base_roas[list_seller][:, None] * rng.lognormal(0, 0.20, (L, D))
        * (1 - 0.60 * sm * vuln[:, None]))
roas = np.clip(roas, 0.1, None).round(2)
roas[~advertised] = np.nan
# Inventory decline: SKUs go out of stock and spoil/arrive damaged far more often.
stockout = (rng.random((L, D)) < base_stockout[list_seller][:, None] + 0.40 * si).astype(int)
damaged = (rng.random((L, D)) < base_expiry[list_seller][:, None] + 0.10 * si).astype(int)

listing_daily = pd.DataFrame({
    "listing_id": np.repeat(listing_ids, D),
    "date": np.tile(dates.strftime("%Y-%m-%d"), L),
    "listing_quality_score": quality.ravel(),
    "content_flag": content_flag.ravel(),
    "ad_return": roas.ravel(),
    "is_stockout": stockout.ravel(),
    "is_expired_or_damaged": damaged.ravel()})

# ------------------------------------------------------------- order_items
# Weekend uplift, normalised to average 1.0 so base_vol stays the "typical day".
wk = np.array([1.0, 1.0, 1.0, 1.0, 1.05, 1.12, 1.10])[dates.dayofweek]
wk = wk / wk.mean()
lam = (base_vol[:, None] * wk[None, :] * rng.lognormal(0, 0.08, (S, D))
       * (1 - 0.30 * sev["marketing"] - 0.30 * sev["inventory"]))
counts = rng.poisson(lam)
sidx = np.repeat(np.repeat(np.arange(S), D), counts.ravel())   # seller of each item
didx = np.repeat(np.tile(day, S), counts.ravel())              # day of each item
n = len(sidx)

def sv(k):
    """Severity of category k for every order item (looked up by its seller and day)."""
    return sev[k][sidx, didx]

lid_idx = list_start[sidx] + (rng.random(n) * n_list[sidx]).astype(int)

# Cancellation = item unavailable at the dark store. Inventory decline drives it.
p_cancel = np.clip(base_cancel[sidx] + 0.25 * sv("inventory") + 0.06 * sv("logistics"), 0, 0.9)
cancelled = rng.random(n) < p_cancel
delivered = ~cancelled
# Late = missed the 10-30 minute promise. Logistics decline + the rain event drive it.
p_late = np.clip(base_late[sidx] + 0.45 * sv("logistics") + rain[didx], 0, 0.95)
late = delivered & (rng.random(n) < p_late)

# Returns are rare in grocery (you don't send milk back); bad quality (CX) raises them.
p_ret = np.clip(base_return[sidx] + 0.02 * late + 0.10 * sv("cx") + 0.04 * sv("marketing"), 0, 0.95)
returned = delivered & (rng.random(n) < p_ret)

# Refund without a return: missing or damaged item, refunded on the spot. Common.
p_ref_nr = np.clip(base_refund_nr[sidx] + 0.12 * sv("finance") + 0.10 * sv("cx")
                   + 0.06 * sv("inventory"), 0, 0.9)
refunded_nr = delivered & ~returned & (rng.random(n) < p_ref_nr)
# Prepaid orders: cancellations are almost always refunded (Meesho used 0.6 for COD).
refunded = ((returned & (rng.random(n) < 0.95)) | (cancelled & (rng.random(n) < 0.90))
            | refunded_nr)
payout_delayed = (~cancelled) & (rng.random(n) < np.clip(base_payout[sidx] + 0.35 * sv("finance"), 0, 0.9))

# Reviews: few people rate grocery orders (~15%).
has_review = delivered & (rng.random(n) < 0.15 + 0.05 * returned)
mu = (base_rating[sidx] - 1.3 * late - 1.1 * returned - 1.0 * refunded_nr
      - 1.4 * sv("cx") - 0.5 * sv("marketing") - 0.6 * sv("logistics"))
rating = np.clip(np.rint(rng.normal(mu, 0.85)), 1, 5).astype(int)

# Complaints: knock-on from late deliveries (Logistics) and bad items (CX).
p_cmp = (0.008 + 0.05 * late + 0.06 * returned + 0.05 * refunded_nr
         + 0.10 * sv("cx") + 0.06 * sv("logistics"))
has_cmp = delivered & (rng.random(n) < p_cmp)

# Buyer messages = live chat. Short replies; Seller-support decline makes them slow.
p_msg = 0.08 + 0.25 * late + 0.10 * cancelled + 0.15 * refunded_nr
has_msg = rng.random(n) < p_msg
resp_h = rng.lognormal(np.log(base_resp[sidx] * (1 + 5.0 * sv("support"))), 0.68)

# ------------------------------------------- group items into orders (baskets)
# A quick-commerce basket usually holds 1-4 items from different brand sellers,
# all packed at the same dark store on the same day.
order_key = np.empty(n, dtype=np.int64)
next_key = 0
for d in range(D):
    ids = rng.permutation(np.where(didx == d)[0])
    sizes = rng.choice([1, 2, 3, 4], len(ids), p=[0.35, 0.30, 0.20, 0.15])
    keys = np.repeat(np.arange(len(sizes)), sizes)[: len(ids)]
    order_key[ids] = next_key + keys
    next_key += int(keys.max()) + 1 if len(ids) else 0

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
rain_dates = dates[RAIN_START_DAY:RAIN_START_DAY + RAIN_DAYS]
print(f"\nAt-risk sellers: {int(gt.is_at_risk.sum())} | temporary dips: "
      f"{int((gt.temporary_dip_category != '').sum())} | window: {dates[0].date()} -> {dates[-1].date()}")
print(f"Platform-wide rain event (confounder): {rain_dates[0].date()} -> {rain_dates[-1].date()}")
