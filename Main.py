import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv('D:/Python/Costomer Segmentation Project/Customer_Segmentation.csv')
print(df)


# Display the first few rows in dataset
print("Dataset Head:")
print(df.head())

# Basic information about the dataset
print("\nDataset Info:")
df.info()

# Check for missing values
print("/n missing values:")
print(df.isnull().sum())

# Discriptive Statistics
print("/n Discriptive statistics:")
print(df.describe())

# Visualization the distribution of numerical columns
numerical_columns = ['Age', 'AnnualIncome', 'SpendingScore', 'Tenure', 'ProductsBought', 'AveragePurchaseValue']
for col in numerical_columns:
    plt.figure(figsize=(8,4))
    sns.histplot(df[col], kde=True,  bins=30, color='blue')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()
    
# Encoding categorical Variables
label_encoder = {}
for col in ['Gender','Region','PaymentMethod']:
    le =LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoder[col] = le
    
# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm',fmt=".2f")
plt.title("correlation heatmap")
plt.show()

# Standardizing the data
scaler =StandardScaler()
scaled_features = scaler.fit_transform(df[numerical_columns])
scaled_df = pd.DataFrame(scaled_features, columns = numerical_columns)

# Elbow Method to determine the optimal clusters
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_df)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bo-', markersize=8)
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.grid()
plt.show()

# Applying K-Means Clustering
optimal_k = 4  # Chosen based on the elbow method
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_df)

# Visualizing the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['AnnualIncome'], y=df['SpendingScore'], hue=df['Cluster'], palette='viridis', s=100)
plt.title("Customer Segments Based on Income and Spending Score")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend(title="Cluster")
plt.show()

# Cluster Characteristics
cluster_summary = df.groupby('Cluster').mean()
print("\nCluster Characteristics:")
print(cluster_summary)

# Save the clustered data
output_path = "Customer_Segmentation_Clusters.csv"
df.to_csv(output_path, index=False)
print(f"\nClustered data saved to {output_path}")

# Insights generation
for cluster in range(optimal_k):
    print(f"\nCluster {cluster} Insights:")
    cluster_data = df[df['Cluster'] == cluster]
    print(f"Number of Customers: {len(cluster_data)}")
    print(cluster_data.describe())
    print("Top Regions:")
    print(cluster_data['Region'].value_counts().head())
