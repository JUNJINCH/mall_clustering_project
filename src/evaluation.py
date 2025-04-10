from sklearn.metrics import silhouette_score
import logging

def evaluate_model(model, data):
    try:
        labels = model.labels_
        score = silhouette_score(data, labels)
        logging.info(f"Silhouette Score: {score}")
        return score
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise