from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from utils import save_plot

def perform_kmeans(df, num_cols, n_clusters = 8):

    # Initialize Pipeline
    scaler= StandardScaler()
    preprocessor = make_column_transformer((scaler, num_cols), remainder='drop')
    pipeline = make_pipeline(
        preprocessor,
        KMeans(n_clusters=n_clusters, random_state=0))

    pipeline.fit(df)
    inertia = pipeline.named_steps['kmeans'].inertia_
    return pipeline, inertia

def predict_clusters(df, pipeline):
    return pipeline.predict(df)

def find_optimal_k(df, num_cols, k_range = range(1, 11)):
    """
    Perform elbow method to find optimal number of clusters
    Returns a list of inertias for each k value
    """

    inertias = []
    for k in range(1, 11):
        pipeline, i = perform_kmeans(df, num_cols, k)
        inertias.append(i)
    return inertias

def plot_elbow_curve(k_range, inertias):
    """
    Plot the elbow curve to find the optimal number of clusters.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bx-')
    plt.xlabel('k (number of clusters)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.grid(True)
    save_plot(plt, 'elbow_curve.png')
