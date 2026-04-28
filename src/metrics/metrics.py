"""
Metrics module: Advanced evaluation metrics for DML
"""

import torch
import numpy as np
import sys
from sklearn.metrics import (
    normalized_mutual_info_score, adjusted_mutual_info_score,
    adjusted_rand_score, silhouette_score
)
from typing import List, Tuple, Dict
import faiss


class AdvancedMetrics:
    """Advanced evaluation metrics for deep metric learning"""
    
    def __init__(self, config):
        self.config = config
        self.k = config.evaluation.k
        self.threshold_range = config.evaluation.threshold_range
        self.num_groups = config.evaluation.num_groups
        
    def grouped_recall_at_k(self, embeddings: torch.Tensor, 
                            labels: torch.Tensor,
                            k: int = 5) -> float:
        """
        Grouped Recall@K: Partition test set into non-overlapping groups
        to make metric invariant to dataset size
        """
        if len(embeddings) == 0:
            return 0.0
            
        # Convert to numpy for FAISS
        embeddings_np = embeddings.cpu().detach().numpy()
        labels_np = labels.cpu().detach().numpy()
        
        # Use deterministic random generator
        rng = np.random.default_rng(self.config.data.seed)
        
        # Partition into groups
        unique_labels = np.unique(labels_np)
        rng.shuffle(unique_labels)
        
        group_size = max(1, len(unique_labels) // self.num_groups)
        groups = [unique_labels[i:i+group_size] for i in range(0, len(unique_labels), group_size)]
        
        recalls = []
        
        for group in groups:
            # Filter embeddings and labels for this group
            mask = np.isin(labels_np, group)
            group_embeddings = embeddings_np[mask]
            group_labels = labels_np[mask]
            
            if len(group_embeddings) == 0:
                continue
                
            # Build FAISS index
            dimension = group_embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(group_embeddings.astype("float32"))
            
            # Query for each sample
            distances, indices = index.search(
                group_embeddings.astype("float32"), k + 1
            )
            
            # Calculate recall@k
            recall_count = 0
            for i in range(len(group_embeddings)):
                neighbor_labels = group_labels[indices[i][1:k+1]]  # Skip self
                if group_labels[i] in neighbor_labels:
                    recall_count += 1
                    
            recalls.append(recall_count / len(group_embeddings))
            
        return np.mean(recalls) if recalls else 0.0
    
    def opis(self, embeddings: torch.Tensor,
             labels: torch.Tensor) -> float:
        """
        Operating-Point-Inconsistency Score: Measure threshold consistency
        across the embedding space
        """
        embeddings_np = embeddings.cpu().detach().numpy()
        labels_np = labels.cpu().detach().numpy()
        
        # Compute pairwise distances
        distances = np.zeros((len(embeddings), len(embeddings)))
        for i in range(len(embeddings)):
            diff = embeddings_np - embeddings_np[i]
            distances[i] = np.sqrt(np.sum(diff ** 2, axis=1))
        
        # Compute F1 score for each threshold
        f1_scores = []
        
        for threshold in self.threshold_range:
            tp = 0
            fp = 0
            fn = 0
            
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    is_same_class = labels_np[i] == labels_np[j]
                    is_close = distances[i, j] <= threshold
                    
                    if is_same_class and is_close:
                        tp += 1
                    elif not is_same_class and is_close:
                        fp += 1
                    elif is_same_class and not is_close:
                        fn += 1
                        
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
            
        # Compute OPIS: average deviation from mean F1
        if len(f1_scores) == 0:
            opis = 0.0
        else:
            mean_f1 = np.mean(f1_scores)
            opis = np.mean(np.abs(np.array(f1_scores) - mean_f1))
        
        return opis
    
    def _predict_clusters(self, embeddings: torch.Tensor, n_clusters: int) -> np.ndarray:
        """Predict cluster assignments using k-means"""
        try:
            from sklearn.cluster import KMeans
            embeddings_np = embeddings.cpu().detach().numpy()
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3, max_iter=50)
            labels_pred = kmeans.fit_predict(embeddings_np.astype("float32"))
            return labels_pred
        except Exception as e:
            print(f"KMeans failed: {e}", file=sys.stderr)
            return np.zeros(len(embeddings), dtype=int)
    
    def compute_all_metrics(self, embeddings: torch.Tensor,
                            labels: torch.Tensor) -> Dict[str, float]:
        """Compute all metrics at once"""
        results = {}
        
        # Basic metrics
        embeddings_np = embeddings.cpu().detach().numpy()
        labels_np = labels.cpu().detach().numpy()
        
        # Get number of unique classes for clustering
        n_clusters = len(np.unique(labels_np))
        # Limit clusters to avoid issues with small datasets
        min_samples_per_cluster = 20
        n_clusters = min(n_clusters, max(2, len(embeddings) // min_samples_per_cluster))
        
        # Predict clusters for clustering-based metrics
        labels_pred = self._predict_clusters(embeddings, n_clusters)
        
        # Clustering metrics: compare predicted clusters to true labels
        # Use long names as the canonical contract
        results["normalized_mutual_info"] = normalized_mutual_info_score(labels_np, labels_pred)
        results["adjusted_mutual_info"] = adjusted_mutual_info_score(labels_np, labels_pred)
        results["adjusted_rand_index"] = adjusted_rand_score(labels_np, labels_pred)
        results["silhouette_score"] = silhouette_score(embeddings_np, labels_np)
        
        # Advanced metrics
        results["grouped_recall_at_k"] = self.grouped_recall_at_k(
            embeddings, labels, k=self.k
        )
        results["opis"] = self.opis(embeddings, labels)
        
        # Optionally include short aliases for backward compatibility
        results["nmi"] = results["normalized_mutual_info"]
        results["ami"] = results["adjusted_mutual_info"]
        results["ari"] = results["adjusted_rand_index"]
        results["silhouette"] = results["silhouette_score"]
        
        # Compute composite_score only after required long names are present
        results["composite_score"] = round(
            (
                results.get("grouped_recall_at_k", 0)
                + (1.0 - results.get("opis", 0))
                + results.get("adjusted_mutual_info", 0)
                + results.get("adjusted_rand_index", 0)
                + results.get("normalized_mutual_info", 0)
                + results.get("silhouette_score", 0)
            )
            / 6.0,
            4,
        )
        
        return results


class StatisticalTests:
    """Statistical significance testing for model comparison"""
    
    @staticmethod
    def five_x_two_cv_paired_ttest(scores1: List[float], 
                                   scores2: List[float]) -> Tuple[float, float]:
        """
        5x2cv Paired t-test: Circumvent overlapping training set biases
        """
        if len(scores1) != len(scores2):
            raise ValueError("Scores must have same length")
            
        n = len(scores1)
        differences = [scores1[i] - scores2[i] for i in range(n)]
        
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        if std_diff == 0:
            t_stat = 0
        else:
            t_stat = mean_diff / (std_diff / np.sqrt(n))
            
        # Degrees of freedom for 5x2cv
        df = 2 * n - 2
        
        # Compute p-value using t-distribution
        from scipy import stats
        p_value = 2 * stats.t.sf(np.abs(t_stat), df)
        
        return t_stat, p_value
    
    @staticmethod
    def mc_nemar(test1_correct: List[bool], 
                 test2_correct: List[bool]) -> Tuple[float, float]:
        """
        McNemar's test: Compare classifiers on same test set
        """
        if len(test1_correct) != len(test2_correct):
            raise ValueError("Test results must have same length")
            
        n = len(test1_correct)
        
        # Contingency table
        both_correct = sum([1 for i in range(n) 
                           if test1_correct[i] and test2_correct[i]])
        only_test1 = sum([1 for i in range(n) 
                         if test1_correct[i] and not test2_correct[i]])
        only_test2 = sum([1 for i in range(n) 
                         if not test1_correct[i] and test2_correct[i]])
        neither_correct = sum([1 for i in range(n) 
                              if not test1_correct[i] and not test2_correct[i]])
        
        # McNemar's statistic
        b = only_test1
        c = only_test2
        
        if b + c == 0:
            chi2 = 0
        else:
            chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
            
        # Degrees of freedom = 1
        from scipy import stats
        p_value = stats.chi2.sf(chi2, 1)
        
        return chi2, p_value
