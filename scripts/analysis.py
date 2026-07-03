import pandas as pd

# Load raw data
df = pd.read_csv("./logs/delta_sensitivity/delta_sensitivity_raw_seeds.csv")

# Filter for single environment
single_df = df[df["environment"] == "multi"]

# Pivot to see per-seed FZC across deltas
pivot = single_df.pivot(index="seed", columns="delta", values="fzc")
print(pivot)