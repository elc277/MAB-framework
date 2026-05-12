import pandas as pd

csv_1 = r"results/full_ratio_sweep_degradation_part_1/full_ratio_sweep_degradation_part_1_under.csv"
csv_2 = r"results/full_ratio_sweep_degradation_part_2/full_ratio_sweep_degradation_part_2_over.csv"

df1 = pd.read_csv(csv_1)
df2 = pd.read_csv(csv_2)

merged = pd.concat([df1, df2], ignore_index=True)

merged.to_csv(
    r"results/full_ratio_sweep_degradation_1_to_100_agents_degradation/full_ratio_sweep_degradation.csv",
    index=False
)