from sklearn.cluster import KMeans

def perform_kmeans(df, num_cols):
    X = df[num_cols]
    kmeans = KMeans()
    kmeans.fit(X)
    return kmeans

def predict_clusters(df, kmeans, num_cols):
    X = df[num_cols]
    return kmeans.predict(X)