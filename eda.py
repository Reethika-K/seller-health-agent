import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/seller_health_daily.csv")

print(df.shape)
print(df.head())
print(df.groupby("seller_id")["is_at_risk_seller"].first().value_counts())

df["return_rate"] = df["returns"] / df["orders"].replace(0, pd.NA)

at_risk_id = df.loc[df["is_at_risk_seller"], "seller_id"].iloc[0]
healthy_id = df.loc[~df["is_at_risk_seller"], "seller_id"].iloc[0]

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
for seller_id, label, color in [(at_risk_id, "At-risk", "crimson"), (healthy_id, "Healthy", "seagreen")]:
    sub = df[df["seller_id"] == seller_id].sort_values("day")
    axes[0].plot(sub["day"], sub["return_rate"].rolling(7, min_periods=1).mean(),
                 label=f"{seller_id} ({label})", color=color)
    axes[1].plot(sub["day"], sub["avg_rating"], label=f"{seller_id} ({label})", color=color)

axes[0].set_ylabel("7-day avg return rate")
axes[0].legend()
axes[1].set_ylabel("Rating")
axes[1].set_xlabel("Day")
axes[1].legend()
fig.suptitle("Healthy vs. at-risk seller: return rate and rating over 90 days")
fig.tight_layout()
fig.savefig("data/eda_trends.png", dpi=120)
print("Saved plot to data/eda_trends.png")