import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ----------------------------
# Load data
# ----------------------------
df = pd.read_excel('Assignment_12/Online Retail.xlsx', sheet_name='Online Retail')
print(df.shape)
print(df.head())

# ----------------------------
# Clean data
# ----------------------------
df = df[df['Quantity'] > 0]
df = df[pd.notnull(df['CustomerID'])]
print(df.shape)

# ----------------------------
# Date filtering
# ----------------------------
print('Date Range: %s ~ %s' % (df['InvoiceDate'].min(), df['InvoiceDate'].max()))
df = df[df['InvoiceDate'] < '2011-12-01']

# ----------------------------
# Sales calculation
# ----------------------------
df['Sales'] = df['Quantity'] * df['UnitPrice']

# ----------------------------
# Customer-level aggregation
# ----------------------------
customer_df = df.groupby('CustomerID').agg({
    'Sales': sum,
    'InvoiceNo': lambda x: x.nunique()
})
customer_df.columns = ['TotalSales', 'OrderCount']
customer_df['AvgOrderValue'] = customer_df['TotalSales'] / customer_df['OrderCount']

print(customer_df.head(15))
print(customer_df.describe())

# ----------------------------
# Normalize features
# ----------------------------
normalized_df = (customer_df - customer_df.mean()) / customer_df.std()
print(normalized_df.head(15))

# ----------------------------
# K-Means clustering
# ----------------------------
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']])
normalized_df['Cluster'] = kmeans.labels_
print(normalized_df['Cluster'].value_counts())
print("Cluster centers:\n", kmeans.cluster_centers_)

# ----------------------------
# Plot clusters
# ----------------------------
colors = ['blue', 'red', 'orange', 'green']
for i in range(4):
    cluster_data = normalized_df[normalized_df['Cluster'] == i]
    plt.scatter(cluster_data['OrderCount'], cluster_data['TotalSales'], c=colors[i], label=f'Cluster {i}')

plt.title('TotalSales vs OrderCount Clusters')
plt.xlabel('Order Count')
plt.ylabel('Total Sales')
plt.legend()
plt.show()

for i in range(4):
    cluster_data = normalized_df[normalized_df['Cluster'] == i]
    plt.scatter(cluster_data['OrderCount'], cluster_data['AvgOrderValue'], c=colors[i], label=f'Cluster {i}')

plt.title('AvgOrderValue vs OrderCount Clusters')
plt.xlabel('Order Count')
plt.ylabel('Avg Order Value')
plt.legend()
plt.grid()
plt.show()

for i in range(4):
    cluster_data = normalized_df[normalized_df['Cluster'] == i]
    plt.scatter(cluster_data['TotalSales'], cluster_data['AvgOrderValue'], c=colors[i], label=f'Cluster {i}')

plt.title('AvgOrderValue vs TotalSales Clusters')
plt.xlabel('Total Sales')
plt.ylabel('Avg Order Value')
plt.legend()
plt.grid()
plt.show()

# ----------------------------
# Silhouette scores for different cluster counts
# ----------------------------
for n_clusters in range(4, 9):
    kmeans_test = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_test.fit(normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']])
    silhouette_avg = silhouette_score(normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']], kmeans_test.labels_)
    print(f'Silhouette Score for {n_clusters} Clusters: {silhouette_avg:.4f}')

# ----------------------------
# High-value cluster analysis
# ----------------------------
high_value_cluster = normalized_df[normalized_df['Cluster'] == 1]  # adjust cluster ID if necessary
print(high_value_cluster.head())

print(customer_df.loc[high_value_cluster.index].describe())

top_products_high_value = df[df['CustomerID'].isin(high_value_cluster.index)] \
    .groupby('Description').count()['StockCode'] \
    .sort_values(ascending=False).head()
print(top_products_high_value)

top_products_low_value = df[df['CustomerID'].isin(normalized_df[normalized_df['Cluster'] == 0].index)] \
    .groupby('Description').count()['StockCode'] \
    .sort_values(ascending=False).head()
print(top_products_low_value)
