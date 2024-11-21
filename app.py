# %%
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Set number of threads for numpy (optional)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Title and Introduction
st.set_page_config(page_title="K-means Clustering with Gradient Descent", layout="wide")
st.title("K-means Clustering with Gradient Descent")

st.markdown("""
Welcome to this interactive exploration of **K-means clustering using gradient descent**! In this app, we'll delve into how the algorithm works, step by step, with mathematical explanations and visualizations.

---

### **Objective**

Cluster customers based on their demographics and purchasing behavior to identify distinct segments. This can help businesses tailor their marketing strategies and improve customer satisfaction.

### **Dataset Context**

We'll use the **Mall Customers** dataset, which contains information about:

- **CustomerID**: Unique identifier for each customer.
- **Gender**: Male or Female.
- **Age**: Age of the customer.
- **Annual Income (k$)**: Annual income of the customer in thousand dollars.
- **Spending Score (1-100)**: Score assigned by the mall based on customer behavior and spending nature.

---

Let's embark on this mathematical journey!
""")

# Step 1: Load and Preprocess the Data
st.header("1. Data Loading and Preprocessing")

def load_and_preprocess_data():
    """
    Load the sample dataset and preprocess it by handling missing values
    and encoding categorical data.

    Returns:
        pd.DataFrame: The original DataFrame with preprocessing.
        numpy.ndarray: The feature matrix.
    """
    # Load sample dataset
    df = pd.read_csv("Mall_Customers 2.csv")
    df.rename(columns={'Genre': 'Gender'}, inplace=True)
    df = df.dropna(subset=['Gender'])
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender']
    X = df[features].values.astype(np.float64)
    return df, X

df, X = load_and_preprocess_data()
st.write("**Data Preview:**")
st.dataframe(df.head())

# Step 2: Understanding K-means Clustering
st.header("2. Understanding K-means Clustering")

st.markdown(r"""
K-means clustering aims to partition **N** data points into **K** clusters by minimizing the within-cluster sum of squares (WCSS). The objective function is:

$$
J = \sum_{i=1}^{N} \sum_{k=1}^{K} r_{ik} \left\| \mathbf{x}_i - \boldsymbol{\mu}_k \right\|^2
$$

- $\mathbf{x}_i$ is the $i^{\text{th}}$ data point.
- $\boldsymbol{\mu}_k$ is the centroid of cluster $k$.
- $r_{ik}$ is a binary indicator (1 if $\mathbf{x}_i$ is assigned to cluster $k$, 0 otherwise).

The goal is to find the cluster assignments $r_{ik}$ and centroids $\boldsymbol{\mu}_k$ that minimize $J$.
""")

# Step 3: Elbow Method to Determine Optimal K
st.header("3. Determining the Optimal Number of Clusters (K)")

st.markdown("""
The **Elbow Method** helps in selecting the optimal number of clusters by plotting the WCSS against different values of K. The point where the rate of decrease sharply changes (forming an 'elbow') indicates the optimal K.
""")

def elbow_method(X, max_clusters=10):
    """
    Calculate and plot the Elbow Method to determine the optimal number of clusters.

    Parameters:
        X (numpy.ndarray): The dataset to cluster.
        max_clusters (int): The maximum number of clusters to test.

    Returns:
        None
    """
    from sklearn.cluster import KMeans

    wcss = []  # List to store the WCSS for each number of clusters

    # Calculate WCSS for each number of clusters
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Plot the Elbow Curve
    fig = px.line(
        x=range(1, max_clusters + 1),
        y=wcss,
        markers=True,
        title='Elbow Method for Optimal K',
        labels={'x': 'Number of Clusters (K)', 'y': 'WCSS (Within-Cluster Sum of Squares)'},
        width=800,
        height=500
    )
    fig.update_layout(title_font_size=20)
    st.plotly_chart(fig)

elbow_method(X, max_clusters=10)
st.markdown("""
From the elbow plot, we observe that the optimal number of clusters appears to be **K = 5**. This is where the rate of decrease in WCSS begins to slow down significantly.
""")

# Step 4: Gradient Descent Approach
st.header("4. Gradient Descent Approach")

st.markdown(r"""
Instead of the traditional approach, we'll use **gradient descent** to update the centroids.

**Gradient of the Objective Function w.r.t Centroids**:

$$
\frac{\partial J}{\partial \boldsymbol{\mu}_k} = -2 \sum_{i=1}^{N} r_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k)
$$

We update the centroids using:

$$
\boldsymbol{\mu}_k^{(t+1)} = \boldsymbol{\mu}_k^{(t)} - \eta \left( \frac{\partial J}{\partial \boldsymbol{\mu}_k} \right)
$$

- $\eta$ is the **learning rate**.
""")

# Add a button to show the detailed derivation
with st.expander("Show detailed derivation of the gradient"):
    st.markdown(r"""
    ### **Derivation of the Gradient**

    We want to compute the partial derivative of $J$ with respect to $\boldsymbol{\mu}_k$:

    $$
    \frac{\partial J}{\partial \boldsymbol{\mu}_k}
    $$

    **Step 1: Expand the Objective Function**

    Recall that the squared Euclidean norm can be expanded:

    $$
    \left\| \mathbf{x}_i - \boldsymbol{\mu}_k \right\|^2 = (\mathbf{x}_i - \boldsymbol{\mu}_k)^\top (\mathbf{x}_i - \boldsymbol{\mu}_k)
    $$

    **Step 2: Focus on Relevant Terms**

    Since $r_{ik}$ selects the terms where $\mathbf{x}_i$ belongs to cluster $k$, we can write:

    $$
    J = \sum_{i=1}^{N} r_{ik} \left\| \mathbf{x}_i - \boldsymbol{\mu}_k \right\|^2
    $$

    **Step 3: Compute the Gradient**

    Compute the derivative:

    $$
    \begin{align*}
    \frac{\partial J}{\partial \boldsymbol{\mu}_k} &= \frac{\partial}{\partial \boldsymbol{\mu}_k} \left( \sum_{i=1}^{N} r_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k)^\top (\mathbf{x}_i - \boldsymbol{\mu}_k) \right) \\
    &= \sum_{i=1}^{N} r_{ik} \frac{\partial}{\partial \boldsymbol{\mu}_k} \left( (\mathbf{x}_i - \boldsymbol{\mu}_k)^\top (\mathbf{x}_i - \boldsymbol{\mu}_k) \right)
    \end{align*}
    $$

    **Step 4: Differentiate the Squared Norm**

    The derivative of the squared norm is:

    $$
    \frac{\partial}{\partial \boldsymbol{\mu}_k} \left( (\mathbf{x}_i - \boldsymbol{\mu}_k)^\top (\mathbf{x}_i - \boldsymbol{\mu}_k) \right) = 2 (\boldsymbol{\mu}_k - \mathbf{x}_i)
    $$

    **Step 5: Substitute Back into the Gradient**

    Thus:

    $$
    \frac{\partial J}{\partial \boldsymbol{\mu}_k} = \sum_{i=1}^{N} r_{ik} \cdot 2 (\boldsymbol{\mu}_k - \mathbf{x}_i) = 2 \sum_{i=1}^{N} r_{ik} (\boldsymbol{\mu}_k - \mathbf{x}_i)
    $$

    **Step 6: Simplify the Expression**

    Factor out the negative sign:

    $$
    \frac{\partial J}{\partial \boldsymbol{\mu}_k} = -2 \sum_{i=1}^{N} r_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k)
    $$

    **Conclusion**

    We have derived:

    $$
    \frac{\partial J}{\partial \boldsymbol{\mu}_k} = -2 \sum_{i=1}^{N} r_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k)
    $$

    This gradient indicates the direction to update the centroids to minimize the objective function.
    """)

# Display the kmeans_gradient_descent function code
st.subheader("Implementation of Gradient Descent for K-means")
st.markdown("Below is the code for the `kmeans_gradient_descent` function that we will use to perform clustering:")
st.code('''
def kmeans_gradient_descent(X, K, num_epochs=500, learning_rate=0.001):
    # Step 1: Extract the dimensions of the input data
    N, D = X.shape  # N: Number of points, D: Dimensions of each point
    
    # Step 2: Check if the number of clusters is valid
    if N < K:
        raise ValueError("Number of clusters (K) cannot exceed the number of data points.")

    # Step 3: Initialize centroids randomly from the data points
    np.random.seed(42)  # Set random seed for reproducibility
    random_indices = np.random.choice(N, K, replace=False)  # Randomly select K unique indices
    centroids = X[random_indices, :].astype(np.float64)  # Initialize centroids using selected points

    # Step 4: Initialize a history list to store centroids at each epoch
    centroids_history = [centroids.copy()]  # Copy current centroids to avoid overwriting
    
    # Step 5: Main optimization loop
    for epoch in range(num_epochs):  # Repeat for the specified number of epochs
        
        # Step 5.1: Calculate distances of each point to all centroids
        # Broadcasting: X[:, np.newaxis] adds a new axis for element-wise operations
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  
        
        # Step 5.2: Assign each point to the nearest centroid
        # Get the index of the closest centroid (minimum distance)
        assignments = np.argmin(distances, axis=1)

        # Step 5.3: Update centroids using gradient descent
        for k in range(K):  # Iterate over each cluster
            # Extract points assigned to the current cluster
            assigned_points = X[assignments == k]
            
            # Skip centroid update if no points are assigned
            if len(assigned_points) > 0:
                # Compute the gradient for the centroid (sum of differences between points and centroid)
                gradient = -2 * np.sum(assigned_points - centroids[k], axis=0)
                
                # Update centroid using gradient descent formula
                centroids[k] -= learning_rate * gradient / len(assigned_points)
        
        # Step 5.4: Append a copy of the updated centroids to the history
        centroids_history.append(centroids.copy())

    # Step 6: Return the final centroids, assignments, and the history of centroids
    return centroids, assignments, centroids_history

''', language='python')

# Step 5: Implementing K-means with Gradient Descent
st.header("5. Implementing K-means with Gradient Descent")

# Parameters
K = 5
num_epochs = 500
learning_rate = 0.001

def kmeans_gradient_descent(X, K, num_epochs=500, learning_rate=0.001):
    
    # Step 1: Extract the dimensions of the input data
    N, D = X.shape  # N: Number of points, D: Dimensions of each point
    
    # Step 2: Check if the number of clusters is valid
    if N < K:
        raise ValueError("Number of clusters (K) cannot exceed the number of data points.")

    # Step 3: Initialize centroids randomly from the data points
    np.random.seed(42)  # Set random seed for reproducibility
    random_indices = np.random.choice(N, K, replace=False)  # Randomly select K unique indices
    centroids = X[random_indices, :].astype(np.float64)  # Initialize centroids using selected points

    # Step 4: Initialize a history list to store centroids at each epoch
    centroids_history = [centroids.copy()]  # Copy current centroids to avoid overwriting
    
    # Step 5: Main optimization loop
    for epoch in range(num_epochs):  # Repeat for the specified number of epochs
        
        # Step 5.1: Calculate distances of each point to all centroids
        # Broadcasting: X[:, np.newaxis] adds a new axis for element-wise operations
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  
        
        # Step 5.2: Assign each point to the nearest centroid
        # Get the index of the closest centroid (minimum distance)
        assignments = np.argmin(distances, axis=1)

        # Step 5.3: Update centroids using gradient descent
        for k in range(K):  # Iterate over each cluster
            # Extract points assigned to the current cluster
            assigned_points = X[assignments == k]
            
            # Skip centroid update if no points are assigned
            if len(assigned_points) > 0:
                # Compute the gradient for the centroid (sum of differences between points and centroid)
                gradient = -2 * np.sum(assigned_points - centroids[k], axis=0)
                
                # Update centroid using gradient descent formula
                centroids[k] -= learning_rate * gradient / len(assigned_points)
        
        # Step 5.4: Append a copy of the updated centroids to the history
        centroids_history.append(centroids.copy())

    # Step 6: Return the final centroids, assignments, and the history of centroids
    return centroids, assignments, centroids_history



st.markdown(f"""
**Parameters**:

- **Number of clusters (K)**: {K}
- **Number of epochs**: {num_epochs}
- **Learning rate (η)**: {learning_rate}

We will now perform clustering using these parameters.
""")

# Perform clustering
centroids, assignments, centroids_history = kmeans_gradient_descent(
    X, K, num_epochs=num_epochs, learning_rate=learning_rate
)

# Step 6: Visualizing the Results
st.header("6. Visualizing the Results")

df['Cluster'] = assignments.astype(str)

st.subheader("6.1 2D Visualization")

st.markdown("""
In this 2D plot, we visualize the clusters using **Annual Income** and **Spending Score**. It might appear that some clusters overlap, but this is because we're projecting high-dimensional data onto two dimensions. The clustering algorithm considers all features, including **Age** and **Gender**, which we will explore in the 3D visualization.
""")

def plot_clusters_2d(df, centroids):
    fig = px.scatter(
        df,
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        color='Cluster',
        title='Customer Segments in 2D',
        labels={'Annual Income (k$)': 'Annual Income ($k)', 'Spending Score (1-100)': 'Spending Score'},
        opacity=0.8,
        width=800,
        height=600,
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    fig.update_traces(marker=dict(size=12))
    fig.add_trace(
        go.Scatter(
            x=centroids[:, 1],
            y=centroids[:, 2],
            mode='markers',
            marker=dict(color='white', size=18, symbol='x'),
            name='Centroids'
        )
    )
    st.plotly_chart(fig)

plot_clusters_2d(df, centroids)

st.subheader("6.2 3D Visualization")

def plot_clusters_3d(df):
    fig_3d = px.scatter_3d(
        df,
        x='Age',
        y='Annual Income (k$)',
        z='Spending Score (1-100)',
        color='Cluster',
        title="3D Visualization of Customer Clusters",
        opacity=0.8,
        labels={
            'Age': 'Age',
            'Annual Income (k$)': 'Annual Income ($k)',
            'Spending Score (1-100)': 'Spending Score',
            'Cluster': 'Cluster'
        },
        width=800,
        height=600,
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    fig_3d.update_traces(marker=dict(size=5))
    st.plotly_chart(fig_3d)

plot_clusters_3d(df)

# Step 7: Animating Centroid Updates
st.header("7. Animating Centroid Updates")

st.markdown("""
Observe how the centroids move over iterations as they minimize the objective function.
""")

def animate_centroid_updates(df, centroids_history):
    import plotly.express as px  # Make sure to import px
    epochs = len(centroids_history)
    K = len(centroids_history[0])  # Number of clusters
    
    # Prepare color palette
    colors = px.colors.qualitative.Dark24  # Qualitative color palette
    if K > len(colors):
        st.error("Number of clusters exceeds the number of available colors.")
        return
    
    # Prepare data for animation
    frames = []
    for i in range(0, epochs, 10):  # Skip frames for faster animation
        centroids = centroids_history[i]
        # Assignments for current centroids
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        assignments = np.argmin(distances, axis=1)
        df['Cluster'] = assignments.astype(int)
        
        # Map cluster indices to colors
        cluster_colors = [colors[cluster_idx] for cluster_idx in df['Cluster']]
    
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=df['Annual Income (k$)'],
                    y=df['Spending Score (1-100)'],
                    mode='markers',
                    marker=dict(size=8, color=cluster_colors),
                    showlegend=False
                ),
                go.Scatter(
                    x=centroids[:, 1],
                    y=centroids[:, 2],
                    mode='markers',
                    marker=dict(color='white', size=18, symbol='x'),
                    name='Centroids'
                )
            ],
            name=f'Frame {i}'
        )
        frames.append(frame)
    
    # Initial plot
    centroids = centroids_history[0]
    df['Cluster'] = np.zeros(len(df), dtype=int)
    initial_colors = ['gray'] * len(df)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=df['Annual Income (k$)'],
                y=df['Spending Score (1-100)'],
                mode='markers',
                marker=dict(size=8, color=initial_colors),
                showlegend=False
            ),
            go.Scatter(
                x=centroids[:, 1],
                y=centroids[:, 2],
                mode='markers',
                marker=dict(color='white', size=18, symbol='x'),
                name='Centroids'
            )
        ],
        frames=frames
    )
    
    # Update layout with sliders and buttons
    fig.update_layout(
        title='Centroid Updates Over Iterations',
        width=800,
        height=600,
        xaxis_title='Annual Income ($k)',
        yaxis_title='Spending Score',
        updatemenus=[dict(
            type='buttons',
            buttons=[dict(label='Play',
                          method='animate',
                          args=[None, {'frame': {'duration': 50, 'redraw': True},
                                       'fromcurrent': True,
                                       'transition': {'duration': 0}}])]
        )],
        sliders=[dict(
            steps=[dict(method='animate',
                        args=[[f'Frame {k}'], {'frame': {'duration': 50, 'redraw': True},
                                                'mode': 'immediate'}],
                        label=f'{k}') for k in range(0, epochs, 10)],
            transition={'duration': 0},
            x=0,
            y=0,
            currentvalue=dict(font=dict(size=12), prefix='Epoch ', visible=True, xanchor='center'),
            len=1.0)
        ]
    )
    
    st.plotly_chart(fig)

animate_centroid_updates(df, centroids_history)

# Step 8: Conclusion
st.header("8. Conclusion")

st.markdown("""
In this journey, we've explored how K-means clustering works and how gradient descent can be applied to optimize the centroids. By visualizing the centroid movements, we gain insights into the convergence process.

**Key Takeaways**:

- **K-means clustering** minimizes the within-cluster sum of squares to find cohesive clusters.
- **Gradient descent** updates centroids iteratively by moving them in the direction that reduces the objective function.
- **Elbow Method** helps determine the optimal number of clusters by identifying the point where adding more clusters doesn't significantly reduce WCSS.
- **Visualizations** are powerful tools to understand the clustering results and the optimization process.

---


""")