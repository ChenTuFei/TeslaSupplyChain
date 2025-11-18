import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Raw input data ---
data = {
    "Description": [
        "Raw material", "Labor", "Indirect costs",
        "Packaging/transport/inventory", "Electricity",
        "Depreciation", "Tariff costs"
    ],
    "US": [40, 12, 10, 9, 4, 5, 0],
    "Mexico": [35, 8, 8, 7, 3, 1, 15.5],
    "China": [30, 4, 4, 12, 4, 5, 15],
}

df = pd.DataFrame(data)
total_US = df["US"].sum()       # 80
total_MX = df["Mexico"].sum()   # 77.5
total_CN = df["China"].sum()    # 74

# --- Time series setup (0 → 6 months) ---
months = np.linspace(0, 6, 61)  # monthly resolution (0.1-month steps)

# Smooth yield ramp function (ease-out quadratic)
def ease_to_one(t, t_end, y0):
    x = np.clip(t / t_end, 0, 1)
    return y0 + (1 - y0) * (1 - (1 - x)**2)

# US yield: 80% → 100%
yield_US = ease_to_one(months, 6, 0.80)
# Mexico yield: 90% → 100%
yield_MX = ease_to_one(months, 6, 0.90)
# China yield: 95% → ~99% (approaches but doesn’t reach 100%)
y0_cn = 0.95
target_at_6 = 0.99
k = -np.log((1 - target_at_6) / (1 - y0_cn)) / 6.0
yield_CN = 1 - (1 - y0_cn) * np.exp(-k * months)

# --- Convert yield → cost per good lamp ---
price_US = total_US / yield_US
price_MX = total_MX / yield_MX
price_CN = total_CN / yield_CN

# --- Plot ---
plt.figure(figsize=(8,5))
plt.plot(months, price_US, label="US", color="royalblue", linewidth=2)
plt.plot(months, price_MX, label="Mexico", color="green", linewidth=2)
plt.plot(months, price_CN, label="China", color="red", linewidth=2)

plt.title("Per-Lamp Cost During 6-Month Production Ramp")
plt.xlabel("Time (months)")
plt.ylabel("Price per lamp (USD)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# Optional: show total baseline costs
print("Baseline total costs (before yield adjustment):")
print(f"US: ${total_US:.2f}, Mexico: ${total_MX:.2f}, China: ${total_CN:.2f}")

