import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create output folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("data/Crop_recommendation.csv")

print("=" * 60)
print("CROP RECOMMENDATION DATASET OVERVIEW")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\nStatistical summary:")
print(df.describe())

print("\nUnique crop labels:")
print(df["label"].unique())

print(f"\nNumber of crop classes: {df['label'].nunique()}")

print("\nClass distribution:")
print(df["label"].value_counts())

# -----------------------------
# Plot 1: Class distribution
# -----------------------------
plt.figure(figsize=(12, 6))
df["label"].value_counts().plot(kind="bar")
plt.title("Crop Class Distribution")
plt.xlabel("Crop")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/class_distribution.png")

plt.show()\

# -----------------------------
# Plot 2: Histograms
# -----------------------------
df.hist(figsize=(14, 10), bins=20)
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.savefig("outputs/feature_histograms.png")
plt.show()

# -----------------------------
# Plot 3: Correlation Heatmap
# -----------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(df.drop("label", axis=1).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.show()

print("\nEDA completed successfully.")
print("Plots saved in the 'outputs' folder.")