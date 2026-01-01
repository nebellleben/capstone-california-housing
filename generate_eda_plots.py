"""
Generate EDA plots for the California Housing dataset
Run this script to create visualization images for the README
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
df = pd.read_csv('housing.csv')

# Data type conversions
to_categorical = ['ocean_proximity']
df[to_categorical] = df[to_categorical].astype('category')

to_integer = ['housing_median_age', 'total_rooms', 'households', 'population']
df[to_integer] = df[to_integer].astype('int64')

numerical_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                      'total_bedrooms', 'population', 'households', 'median_income']

# 1. Histograms of Numerical Features
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 16))
for i, feature in enumerate(numerical_features):
    row = i // 2
    col = i % 2
    sns.histplot(df[feature], bins=30, kde=False, ax=axes[row, col])
    axes[row, col].set_title(f'Histogram of {feature}', fontsize=12, fontweight='bold')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Frequency')

plt.suptitle('Distribution of Numerical Features', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('images/eda_histograms.png', bbox_inches='tight', dpi=150)
plt.close()
print("✓ Saved: images/eda_histograms.png")

# 2. Geographic Distribution
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['longitude'], df['latitude'], 
                     c=df['median_house_value'], 
                     cmap='viridis', 
                     alpha=0.6, 
                     s=20)
plt.colorbar(scatter, label='Median House Value ($)')
plt.title('Geographic Distribution of House Values in California', 
          fontsize=14, fontweight='bold')
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.tight_layout()
plt.savefig('images/eda_geographic.png', bbox_inches='tight', dpi=150)
plt.close()
print("✓ Saved: images/eda_geographic.png")

# 3. Correlation Matrix
corr_matrix = df[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/eda_correlation.png', bbox_inches='tight', dpi=150)
plt.close()
print("✓ Saved: images/eda_correlation.png")

# 4. Ocean Proximity Distribution
plt.figure(figsize=(10, 6))
ocean_counts = df['ocean_proximity'].value_counts()
plt.bar(ocean_counts.index, ocean_counts.values, color='steelblue')
plt.title('Distribution of Ocean Proximity Categories', 
          fontsize=14, fontweight='bold')
plt.xlabel('Ocean Proximity', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('images/eda_ocean_proximity.png', bbox_inches='tight', dpi=150)
plt.close()
print("✓ Saved: images/eda_ocean_proximity.png")

# 5. Box plots for median_house_value by ocean_proximity
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='ocean_proximity', y='median_house_value')
plt.title('House Value Distribution by Ocean Proximity', 
          fontsize=14, fontweight='bold')
plt.xlabel('Ocean Proximity', fontsize=12)
plt.ylabel('Median House Value ($)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('images/eda_boxplot_ocean.png', bbox_inches='tight', dpi=150)
plt.close()
print("✓ Saved: images/eda_boxplot_ocean.png")

print("\n✅ All EDA plots generated successfully!")

