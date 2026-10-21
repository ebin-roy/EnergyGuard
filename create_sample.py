
import pandas as pd

# Read your cleaned dataset
df = pd.read_csv("dataset/SGCC_Cleaned.csv")

# Select the first 10 consumers, keeping all columns
sample = df.head(10)

# Save the sample CSV
sample.to_csv(
    "dataset/EnergyGuard_test_sample.csv",
    index=False
)

print("Sample CSV created successfully!")
print("Rows:", len(sample))
print("Columns:", len(sample.columns))
