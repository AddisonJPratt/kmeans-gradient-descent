# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Step 1: Load and Preprocess the Data
def load_and_preprocess_data(filepath):
    """
    Load the dataset and preprocess it by handling missing values,
    mapping categorical data, and standardizing numerical features.

    Parameters:
        filepath (str): The path to the dataset CSV file.

    Returns:
        pd.DataFrame: The original DataFrame with preprocessing.
        numpy.ndarray: The standardized feature matrix.
    """
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['Gender'])  # Drop rows with NaN Gender values
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})  # Map Gender to numerical
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender']
    X = df[features].values

    return df, X

# Step 2: Elbow Method
def elbow_method(X, max_clusters=10):
    """
    Calculate and plot the Elbow Method to determine the optimal number of clusters.

    Parameters:
        X (numpy.ndarray): The dataset to cluster.
        max_clusters (int): The maximum number of clusters to test.

    Returns:
        int: The optimal number of clusters (K).
    """
    wcss = []  # List to store the WCSS for each number of clusters

    # Calculate WCSS for each number of clusters
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Plot the Elbow Curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='-')
    plt.title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
    plt.xticks(range(1, max_clusters + 1))
    plt.grid(True)
    plt.show()

    return 4

# Step 3: K-means Gradient Descent Implementation
def kmeans_gradient_descent(X, K, num_epochs=100, learning_rate=0.01):
    """
    Perform K-means clustering using gradient descent.

    Parameters:
        X (numpy.ndarray): The dataset to cluster.
        K (int): The number of clusters.
        num_epochs (int): The number of iterations for gradient descent.
        learning_rate (float): The learning rate for centroid updates.

    Returns:
        numpy.ndarray: The final centroids.
        numpy.ndarray: The cluster assignments for each data point.
    """
    N, D = X.shape
    if N < K:
        raise ValueError("Number of clusters (K) cannot exceed the number of data points.")
    
    np.random.seed(42)  # Randomly initialize centroids
    random_indices = np.random.choice(N, K, replace=False)
    centroids = X[random_indices, :]

    for epoch in range(num_epochs):
        assignments = np.zeros(N, dtype=int)  # To store the cluster index for each point

        # Assign points to the nearest centroid
        for i in range(N):
            distances = np.sum((X[i] - centroids) ** 2, axis=1)
            assignments[i] = np.argmin(distances)  # Assign to the closest centroid

        # Update centroids using gradient descent
        for k in range(K):
            assigned_points = X[assignments == k]

            if len(assigned_points) > 0:  # Avoid division by zero
                gradient = -2 * np.sum(assigned_points - centroids[k], axis=0)
                centroids[k] = centroids[k] - learning_rate * gradient / len(assigned_points)

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Centroids:\n{centroids}")
    
    return centroids, assignments

# Step 4: Visualizations
def plot_clusters_2d(df, optimal_k):
    """
    Plot the clusters in 2D using Annual Income and Spending Score.

    Parameters:
        df (pd.DataFrame): The DataFrame containing cluster assignments.
        optimal_k (int): The number of clusters.
    """
    plt.figure(figsize=(8, 6))
    for k in range(optimal_k):
        cluster_data = df[df['Cluster'] == k]
        plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], label=f'Cluster {k}')
    plt.scatter(df['Centroid Annual Income'], df['Centroid Spending Score'], c='red', marker='X', s=200, label='Centroids')
    plt.xlabel('Annual Income ($k)')
    plt.ylabel('Spending Score')
    plt.title('Customer Segments')
    plt.legend()
    plt.show()

def plot_clusters_3d(df):
    """
    Plot the clusters in 3D using Plotly.

    Parameters:
        df (pd.DataFrame): The DataFrame containing cluster assignments.
    """
    distinct_colors = px.colors.qualitative.Set2  # Use Plotly's distinct color palette
    fig_3d = px.scatter_3d(
        df,
        x='Age',
        y='Annual Income (k$)',
        z='Spending Score (1-100)',
        color='Cluster',
        title="Enhanced 3D Visualization of Customer Clusters",
        opacity=0.7,
        labels={
            'Age': 'Age',
            'Annual Income (k$)': 'Annual Income ($k)',
            'Spending Score (1-100)': 'Spending Score',
            'Cluster': 'Cluster Group'
        },
        hover_data=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
        color_discrete_sequence=distinct_colors
    )
    fig_3d.show()

# Step 5: Main Workflow
def main():
    # Load and preprocess data
    filepath = "Mall_Customers 2.csv"
    df, X = load_and_preprocess_data(filepath)

    # Use Elbow Method to determine the optimal K
    print("Running Elbow Method...")
    optimal_k = elbow_method(X, max_clusters=10)

    # Perform K-means Clustering using Gradient Descent
    print(f"Clustering with K={optimal_k} using Gradient Descent...")
    centroids, assignments = kmeans_gradient_descent(X, optimal_k, num_epochs=500, learning_rate=0.01)

    # Add cluster assignments and centroids to the DataFrame
    df['Cluster'] = assignments
    df['Centroid Annual Income'] = [centroids[cluster][1] for cluster in assignments]
    df['Centroid Spending Score'] = [centroids[cluster][2] for cluster in assignments]

    # Visualizations
    print("Generating 2D Plot...")
    plot_clusters_2d(df, optimal_k)
    print("Generating 3D Plot...")
    plot_clusters_3d(df)

# Run the main function
if __name__ == "__main__":
    main()
# %%
