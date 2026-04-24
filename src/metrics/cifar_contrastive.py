from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


def grouped_recall_at_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
    n_groups: int = 4,
) -> float:
    n_samples = len(labels)
    samples_per_group = n_samples // n_groups
    
    grouped_rk_scores = []
    
    for i in range(n_groups):
        start_idx = i * samples_per_group
        end_idx = start_idx + samples_per_group
        
        group_embeddings = embeddings[start_idx:end_idx]
        group_labels = labels[start_idx:end_idx]
        
        distances = np.linalg.norm(
            group_embeddings[:, np.newaxis] - group_embeddings[np.newaxis, :],
            axis=2
        )
        
        recall_scores = []
        for j in range(len(group_embeddings)):
            row_distances = distances[j]
            row_labels = group_labels
            
            neighbor_indices = np.argsort(row_distances)[1:k+1]
            neighbor_labels = row_labels[neighbor_indices]
            
            correct = np.sum(neighbor_labels == group_labels[j])
            recall = correct / min(k, len(neighbor_labels))
            recall_scores.append(recall)
        
        grouped_rk_scores.append(np.mean(recall_scores))
    
    return float(np.mean(grouped_rk_scores))


def compute_opis(
    embeddings: np.ndarray,
    labels: np.ndarray,
    threshold_range: np.ndarray | None = None,
) -> float:
    if threshold_range is None:
        threshold_range = np.linspace(0.1, 2.0, 50)
    
    n_samples = len(labels)
    distances = np.linalg.norm(
        embeddings[:, np.newaxis] - embeddings[np.newaxis, :],
        axis=2
    )
    
    f1_scores = []
    for thresh in threshold_range:
        predictions = (distances < thresh).astype(int)
        
        tp = np.sum((predictions == 1) & (labels[:, np.newaxis] == labels[np.newaxis, :]))
        fp = np.sum((predictions == 1) & (labels[:, np.newaxis] != labels[np.newaxis, :]))
        fn = np.sum((predictions == 0) & (labels[:, np.newaxis] == labels[np.newaxis, :]))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)
    
    f1_scores = np.array(f1_scores)
    mean_f1 = np.mean(f1_scores)
    
    opis = np.mean(np.abs(f1_scores - mean_f1))
    
    return float(opis)


def compute_ami(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    return float(adjusted_mutual_info_score(labels_true, labels_pred))


def compute_ari(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    return float(adjusted_rand_score(labels_true, labels_pred))


def compute_nmi(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    return float(normalized_mutual_info_score(labels_true, labels_pred))


def compute_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    return float(silhouette_score(embeddings, labels))


def cluster_and_evaluate(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_clusters: int | None = None,
) -> dict[str, float]:
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    return {
        "ami": compute_ami(labels, cluster_labels),
        "ari": compute_ari(labels, cluster_labels),
        "nmi": compute_nmi(labels, cluster_labels),
        "silhouette": compute_silhouette(embeddings, labels),
    }
