import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the data
df = pd.read_csv("PATIENTS.csv")

# Inspect the first few rows of the DataFrame
print("First few rows of the DataFrame:")
print(df.head())

# Check for missing values in the 'dob' column
print("\nMissing values in 'dob' column:")
print(df['dob'].isnull().sum())

# Check unique values in the 'dob' column
print("\nUnique values in 'dob' column:")
print(df['dob'].unique()[:10])  # Print the first 10 unique values

# Convert 'dob' to datetime and handle invalid dates
df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

# Drop rows with missing 'dob' values
df = df.dropna(subset=['dob'])
print(f"\nNumber of rows after dropping invalid/missing dates: {len(df)}")

# Calculate age using the reference year (MIMIC-III uses shifted years)
reference_year = 2100
df['age'] = reference_year - df['dob'].dt.year

# Inspect the first few rows of 'dob' and 'age'
print("\nFirst few rows of 'dob' and 'age':")
print(df[['dob', 'age']].head())

# Remove unrealistic ages (e.g., negative values or above 120)
df = df[(df['age'] > 0) & (df['age'] <= 120)]
print(f"\nNumber of rows after filtering unrealistic ages: {len(df)}")

# Prepare data for K-means clustering
X = df[['age']]  # Feature matrix (2D array)

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Use 3 clusters
df['cluster'] = kmeans.fit_predict(X)  # Add cluster labels to the DataFrame

# Inspect the first few rows with cluster labels
print("\nFirst few rows with cluster labels:")
print(df[['age', 'cluster']].head())

# Plot histogram of age distribution with clusters highlighted
plt.figure(figsize=(10, 5))
for cluster in sorted(df['cluster'].unique()):
    plt.hist(df[df['cluster'] == cluster]['age'], bins=30, alpha=0.7, label=f'Cluster {cluster}')

plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.title("Age Distribution of Patients (Clustered)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()