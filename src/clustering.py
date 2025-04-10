from sklearn.cluster import KMeans
import logging

def apply_kmeans(data, n_clusters=5):
    try:
        model = KMeans(n_clusters=n_clusters, random_state=42)
        model.fit(data)
        logging.info(f"KMeans clustering completed with {n_clusters} clusters.")
        return model
    except Exception as e:
        logging.error(f"KMeans clustering failed: {e}")
        raise
