   # Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set graph style
sns.set_style("whitegrid")

# Load Dataset
df = pd.read_csv("traffic_accidents.csv")

# Display first 5 rows
print(df.head())

# Dataset Information
print("\nDataset Information:")
print(df.info())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove Missing Values
df = df.dropna()

print("\nDataset Shape After Cleaning:")
print(df.shape)
# 1. Weather Condition Analysis
plt.figure(figsize=(10,5))
sns.countplot(data=df,
              y='weather_condition',
              order=df['weather_condition'].value_counts().index)

plt.title("Accidents by Weather Condition")
plt.xlabel("Number of Accidents")
plt.ylabel("Weather")
plt.show()
# 2. Road Surface Condition Analysis
plt.figure(figsize=(10,5))
sns.countplot(data=df,
              y='roadway_surface_cond',
              order=df['roadway_surface_cond'].value_counts().index)

plt.title("Accidents by Road Surface Condition")
plt.xlabel("Number of Accidents")
plt.ylabel("Road Surface")
plt.show()
# 3. Time of Day Analysis
plt.figure(figsize=(12,5))
sns.countplot(data=df, x='crash_hour')

plt.title("Accidents by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Accidents")
plt.show()
# 4. Primary Contributing Cause

plt.figure(figsize=(12,8))
sns.countplot(data=df,
              y='prim_contributory_cause',
              order=df['prim_contributory_cause'].value_counts().head(10).index)

plt.title("Top 10 Accident Causes")
plt.xlabel("Number of Accidents")
plt.ylabel("Cause")
plt.show()
# 5. Crash Type
plt.figure(figsize=(10,6))
sns.countplot(data=df,
              y='crash_type',
              order=df['crash_type'].value_counts().index)

plt.title("Crash Type Distribution")
plt.xlabel("Number of Accidents")
plt.ylabel("Crash Type")
plt.show()
# 6. Injury Severity
plt.figure(figsize=(10,5))
sns.countplot(data=df,
              y='most_severe_injury',
              order=df['most_severe_injury'].value_counts().index)

plt.title("Most Severe Injury")
plt.xlabel("Number of Accidents")
plt.ylabel("Injury Type")
plt.show()
# 7. Damage Analysis
plt.figure(figsize=(8,5))
sns.countplot(data=df,
              x='damage',
              order=df['damage'].value_counts().index)

plt.title("Vehicle Damage")
plt.xticks(rotation=20)
plt.show()
# 8. Heatmap (Correlation)
numeric = df.select_dtypes(include=['int64','float64'])

plt.figure(figsize=(10,8))
sns.heatmap(numeric.corr(),
            annot=True,
            cmap='coolwarm')

plt.title("Correlation Heatmap")
plt.show()
# 9. Summary
print("\n========== SUMMARY ==========\n")

print("Top Weather Condition:")
print(df['weather_condition'].value_counts().head())

print("\nTop Road Condition:")
print(df['roadway_surface_cond'].value_counts().head())

print("\nTop Accident Cause:")
print(df['prim_contributory_cause'].value_counts().head())

print("\nTop Crash Type:")
print(df['crash_type'].value_counts().head())

print("\nTop Injury:")
print(df['most_severe_injury'].value_counts().head())

print("\nAnalysis Completed Successfully!")
