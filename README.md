# Develop By : Nithya shree B
# Reg No : 212223220071
# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

# AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

# Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1)Choose the number of clusters (K): Decide how many clusters you want to identify in your data. This is a hyperparameter that you need to set in advance.

2)Initialize cluster centroids: Randomly select K data points from your dataset as the initial centroids of the clusters.

3)Assign data points to clusters: Calculate the distance between each data point and each centroid. Assign each data point to the cluster with the closest centroid. This step is typically done using Euclidean distance, but other distance metrics can also be used.

4)Update cluster centroids: Recalculate the centroid of each cluster by taking the mean of all the data points assigned to that cluster.

5)Repeat steps 3 and 4: Iterate steps 3 and 4 until convergence. Convergence occurs when the assignments of data points to clusters no longer change or change very minimally.

6)Evaluate the clustering results: Once convergence is reached, evaluate the quality of the clustering results. This can be done using various metrics such as the within-cluster sum of squares (WCSS), silhouette coefficient, or domain-specific evaluation criteria.

7)Select the best clustering solution: If the evaluation metrics allow for it, you can compare the results of multiple clustering runs with different K values and select the one that best suits your requirements


# Program:
```

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Mall_Customers.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
  kmeans = KMeans(n_clusters = i, init = "k-means++")
  kmeans.fit(data.iloc[:, 3:])
  wcss.append(kmeans.inertia_)
  
plt.plot(range(1, 11), wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

km = KMeans(n_clusters = 5)
km.fit(data.iloc[:, 3:])

y_pred = km.predict(data.iloc[:, 3:])
y_pred

data["cluster"] = y_pred
df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]
plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c = "red", label = "cluster0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c = "black", label = "cluster1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c = "blue", label = "cluster2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c = "green", label = "cluster3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c = "magenta", label = "cluster4")
plt.legend()
plt.title("Customer Segments")
```

# Output:
## data.head():
![Screenshot 2024-04-27 175246](https://github.com/Balunithu/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/161273477/1d07acd9-fdb7-46a0-a958-81b8d60df5a9)

## data.info():

![Screenshot 2024-04-27 175255](https://github.com/Balunithu/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/161273477/9e575f39-7be6-41e4-946f-dfa8a899b4c3)


## NULL VALUES:
![Screenshot 2024-04-27 175302](https://github.com/Balunithu/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/161273477/072834fd-9806-4c56-b25e-439f2a3d0bde)

## ELBOW GRAPH:
![Screenshot 2024-04-27 175314](https://github.com/Balunithu/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/161273477/53cb8b61-1ffe-4dad-988a-1efb667eacea)


## CLUSTER FORMATION:

![Screenshot 2024-04-27 175321](https://github.com/Balunithu/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/161273477/c8a7698a-398c-49fb-925f-890c40d106cc)

## PREDICICTED VALUE:
![Screenshot 2024-04-27 175328](https://github.com/Balunithu/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/161273477/9c7fad5b-8fef-4a24-9876-cc69a02354c0)


## FINAL GRAPH(D/O):
![Screenshot 2024-04-27 175338](https://github.com/Balunithu/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/161273477/02299c0f-8cc9-4ae7-9b0b-d72b732fe768)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
