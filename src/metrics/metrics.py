"""
Metrics module: Advanced evaluation metrics for DML
"""

import os
import sys
import time
from math import comb
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

# Only import FAISS on non-macOS platforms (FAISS has OpenMP deadlock on macOS).
# Kubernetes cluster runs on Linux where FAISS works fine.
import_platform_faiss = None
if os.uname().sysname != "Darwin":
    import faiss

    import_platform_faiss = faiss


class AdvancedMetrics:
    """Advanced evaluation metrics for deep metric learning."""

    def __init__(self, config):
        self.config = config
        self.k = config.evaluation.k
        self.threshold_range = config.evaluation.threshold_range
        self.num_groups = config.evaluation.num_groups

    @staticmethod
    def _as_numpy(embeddings: torch.Tensor, labels: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        embeddings_np = embeddings.detach().cpu().numpy().astype("float32", copy=False)
        labels_np = labels.detach().cpu().numpy()
        return embeddings_np, labels_np

    @staticmethod
    def _chance_at_k_for_labels(labels: np.ndarray, k: int) -> float:
        """Exact chance of at least one same-label neighbor under random ranking."""
        if labels.size <= 1 or k <= 0:
            return 0.0

        unique_labels, counts = np.unique(labels, return_counts=True)
        class_counts = dict(zip(unique_labels, counts))
        total_neighbors = labels.size - 1
        k_eff = min(k, total_neighbors)
        if k_eff <= 0:
            return 0.0

        total_draws = comb(total_neighbors, k_eff)
        per_sample = []
        for label in labels:
            positives = int(class_counts[label]) - 1
            if positives <= 0:
                per_sample.append(0.0)
                continue
            negatives = total_neighbors - positives
            no_positive_draws = comb(negatives, k_eff) if negatives >= k_eff else 0
            per_sample.append(1.0 - (no_positive_draws / total_draws))
        return float(np.mean(per_sample))

    @staticmethod
    def recall_chance_estimate(k: int, group_class_count: int) -> float:
        """Approximate chance using only K and the group class count."""
        if k <= 0 or group_class_count <= 1:
            return 0.0
        return float(1.0 - ((group_class_count - 1) / group_class_count) ** k)

    def _class_groups(self, labels_np: np.ndarray, group_size: int) -> list[np.ndarray]:
        rng = np.random.default_rng(self.config.data.seed)
        unique_labels = np.unique(labels_np)
        rng.shuffle(unique_labels)
        actual_group_size = max(1, min(group_size, len(unique_labels)))
        return [unique_labels[i : i + actual_group_size] for i in range(0, len(unique_labels), actual_group_size)]

    def grouped_recall_at_k_details(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        k: int = 5,
        group_size: int = 10,
    ) -> Dict[str, float | bool]:
        """
        Grouped Recall@K using fixed-size class groups and FAISS IndexFlatL2.

        The gallery for each group is all samples from the group's classes; it is
        not evaluated batch-by-batch. Split-level metadata in trainer.py records
        whether that gallery came from a partial max_eval_batches extraction.
        """
        print(f"Time {time.time()}: grouped_recall_at_k starting", file=sys.stderr, flush=True)
        if len(embeddings) == 0:
            return {
                "grouped_recall_at_k": 0.0,
                "grouped_recall_chance_at_k": 0.0,
                "grouped_recall_class_count_chance_at_k": 0.0,
                "grouped_recall_k": float(k),
                "grouped_recall_num_groups": 0.0,
                "grouped_recall_mean_group_num_classes": 0.0,
                "grouped_recall_min_group_num_classes": 0.0,
                "grouped_recall_max_group_num_classes": 0.0,
                "grouped_recall_mean_group_num_samples": 0.0,
                "grouped_recall_at_k_num_groups": 0.0,
                "grouped_recall_at_k_group_size": float(group_size),
                "grouped_recall_at_k_gallery_size": 0.0,
                "grouped_recall_at_k_gallery_classes": 0.0,
                "grouped_recall_at_k_partial": True,
            }

        if import_platform_faiss is None:
            raise RuntimeError("FAISS is required for grouped_recall_at_k on this platform")

        embeddings_np, labels_np = self._as_numpy(embeddings, labels)
        unique_labels = np.unique(labels_np)
        groups = self._class_groups(labels_np, group_size)
        recalls = []
        chance_estimates = []
        group_class_counts = []
        group_sample_counts = []

        for group in groups:
            mask = np.isin(labels_np, group)
            group_embeddings = np.ascontiguousarray(embeddings_np[mask])
            group_labels = labels_np[mask]
            if len(group_embeddings) <= 1:
                continue

            search_k = min(k + 1, len(group_embeddings))
            index = import_platform_faiss.IndexFlatL2(group_embeddings.shape[1])
            index.add(group_embeddings)
            _, indices = index.search(group_embeddings, search_k)

            recall_count = 0
            for row_idx, neighbor_indices in enumerate(indices):
                nearest = [idx for idx in neighbor_indices if idx != row_idx][:k]
                if len(nearest) == 0:
                    continue
                neighbor_labels = group_labels[nearest]
                if group_labels[row_idx] in neighbor_labels:
                    recall_count += 1

            recalls.append(recall_count / len(group_embeddings))
            chance_estimates.append(self._chance_at_k_for_labels(group_labels, k))
            group_class_counts.append(len(np.unique(group_labels)))
            group_sample_counts.append(len(group_labels))

        result = float(np.mean(recalls)) if recalls else 0.0
        chance = float(np.mean(chance_estimates)) if chance_estimates else 0.0
        mean_group_classes = float(np.mean(group_class_counts)) if group_class_counts else 0.0
        class_count_chance = self.recall_chance_estimate(k, int(round(mean_group_classes)))
        effective_group_size = min(group_size, len(unique_labels)) if len(unique_labels) else group_size

        details = {
            "grouped_recall_at_k": result,
            "grouped_recall_chance_at_k": chance,
            "grouped_recall_class_count_chance_at_k": class_count_chance,
            "grouped_recall_k": float(k),
            "grouped_recall_num_groups": float(len(recalls)),
            "grouped_recall_mean_group_num_classes": mean_group_classes,
            "grouped_recall_min_group_num_classes": float(np.min(group_class_counts)) if group_class_counts else 0.0,
            "grouped_recall_max_group_num_classes": float(np.max(group_class_counts)) if group_class_counts else 0.0,
            "grouped_recall_mean_group_num_samples": float(np.mean(group_sample_counts)) if group_sample_counts else 0.0,
            "grouped_recall_at_k_num_groups": float(len(recalls)),
            "grouped_recall_at_k_group_size": float(effective_group_size),
            "grouped_recall_at_k_gallery_size": float(len(labels_np)),
            "grouped_recall_at_k_gallery_classes": float(len(unique_labels)),
            "grouped_recall_at_k_partial": False,
        }
        print(
            f"Time {time.time()}: grouped_recall_at_k done: {result}, chance={chance}",
            file=sys.stderr,
            flush=True,
        )
        return details

    def grouped_recall_at_k(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        k: int = 5,
        group_size: int = 10,
    ) -> float:
        return float(self.grouped_recall_at_k_details(embeddings, labels, k, group_size)["grouped_recall_at_k"])

    def opis_details(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float | bool]:
        """
        Operating-Point-Inconsistency Score: measure threshold consistency.
        """
        embeddings_np, labels_np = self._as_numpy(embeddings, labels)
        sample_limit = int(os.environ.get("GLASSLAB_OPIS_MAX_SAMPLES", "512"))
        original_num_samples = len(labels_np)
        sampled = False
        if len(labels_np) > sample_limit:
            rng = np.random.default_rng(self.config.data.seed)
            idx = np.sort(rng.choice(len(labels_np), size=sample_limit, replace=False))
            embeddings_np = embeddings_np[idx]
            labels_np = labels_np[idx]
            sampled = True

        distances = np.zeros((len(embeddings_np), len(embeddings_np)), dtype=np.float32)
        for i in range(len(embeddings_np)):
            diff = embeddings_np - embeddings_np[i]
            distances[i] = np.sqrt(np.sum(diff**2, axis=1))

        f1_scores = []
        for threshold in self.threshold_range:
            tp = 0
            fp = 0
            fn = 0

            for i in range(len(embeddings_np)):
                for j in range(i + 1, len(embeddings_np)):
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

        if len(f1_scores) == 0:
            opis = 0.0
        else:
            mean_f1 = np.mean(f1_scores)
            opis = np.mean(np.abs(np.array(f1_scores) - mean_f1))

        print(f"Time {time.time()}: opis done: {opis} on n={len(labels_np)}", file=sys.stderr)
        return {
            "opis": float(opis),
            "opis_sampled": sampled,
            "opis_num_samples": float(len(labels_np)),
            "opis_original_num_samples": float(original_num_samples),
        }

    def opis(self, embeddings: torch.Tensor, labels: torch.Tensor) -> float:
        return float(self.opis_details(embeddings, labels)["opis"])

    def _predict_clusters(self, embeddings: torch.Tensor, n_clusters: int) -> np.ndarray:
        """Predict cluster assignments using k-means."""
        try:
            from sklearn.cluster import KMeans

            embeddings_np = embeddings.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3, max_iter=50)
            labels_pred = kmeans.fit_predict(embeddings_np.astype("float32"))
            return labels_pred
        except Exception as e:
            print(f"KMeans failed: {e}", file=sys.stderr)
            return np.zeros(len(embeddings), dtype=int)

    def compute_all_metrics(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute all metrics at once."""
        results = {}

        embeddings_np, labels_np = self._as_numpy(embeddings, labels)

        n_clusters = len(np.unique(labels_np))
        min_samples_per_cluster = 20
        n_clusters = min(n_clusters, max(2, len(embeddings) // min_samples_per_cluster))

        labels_pred = self._predict_clusters(embeddings, n_clusters)

        results["normalized_mutual_info"] = normalized_mutual_info_score(labels_np, labels_pred)
        results["adjusted_mutual_info"] = adjusted_mutual_info_score(labels_np, labels_pred)
        results["adjusted_rand_index"] = adjusted_rand_score(labels_np, labels_pred)

        silhouette_sample_limit = int(os.environ.get("GLASSLAB_SILHOUETTE_MAX_SAMPLES", "2048"))
        if len(labels_np) > silhouette_sample_limit:
            rng = np.random.default_rng(self.config.data.seed)
            idx = np.sort(rng.choice(len(labels_np), size=silhouette_sample_limit, replace=False))
            results["silhouette_score"] = silhouette_score(embeddings_np[idx], labels_np[idx])
            results["silhouette_sampled"] = True
            results["silhouette_num_samples"] = float(silhouette_sample_limit)
        else:
            results["silhouette_score"] = silhouette_score(embeddings_np, labels_np)
            results["silhouette_sampled"] = False
            results["silhouette_num_samples"] = float(len(labels_np))

        results.update(self.grouped_recall_at_k_details(embeddings, labels, k=self.k, group_size=10))
        results.update(self.opis_details(embeddings, labels))

        results["nmi"] = results["normalized_mutual_info"]
        results["ami"] = results["adjusted_mutual_info"]
        results["ari"] = results["adjusted_rand_index"]
        results["silhouette"] = results["silhouette_score"]

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

        results["grouped_recall_at_k"] = results.get("grouped_recall_at_k", 0)
        results["opis"] = results.get("opis", 0)
        results["adjusted_mutual_info"] = results.get("adjusted_mutual_info", 0)
        results["adjusted_rand_index"] = results.get("adjusted_rand_index", 0)
        results["normalized_mutual_info"] = results.get("normalized_mutual_info", 0)
        results["silhouette_score"] = results.get("silhouette_score", 0)

        return results


class StatisticalTests:
    """Statistical significance testing for model comparison."""

    @staticmethod
    def five_x_two_cv_paired_ttest(scores1: List[float], scores2: List[float]) -> Tuple[float, float]:
        """
        5x2cv Paired t-test: Circumvent overlapping training set biases.
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

        df = 2 * n - 2

        from scipy import stats

        p_value = 2 * stats.t.sf(np.abs(t_stat), df)

        return t_stat, p_value

    @staticmethod
    def mc_nemar(test1_correct: List[bool], test2_correct: List[bool]) -> Tuple[float, float]:
        """
        McNemar's test: Compare classifiers on same test set.
        """
        if len(test1_correct) != len(test2_correct):
            raise ValueError("Test results must have same length")

        both_correct = sum([1 for i in range(len(test1_correct)) if test1_correct[i] and test2_correct[i]])
        only_test1 = sum([1 for i in range(len(test1_correct)) if test1_correct[i] and not test2_correct[i]])
        only_test2 = sum([1 for i in range(len(test1_correct)) if not test1_correct[i] and test2_correct[i]])
        neither_correct = sum([1 for i in range(len(test1_correct)) if not test1_correct[i] and not test2_correct[i]])

        b = only_test1
        c = only_test2

        if b + c == 0:
            chi2 = 0
        else:
            chi2 = ((abs(b - c) - 1) ** 2) / (b + c)

        from scipy import stats

        p_value = stats.chi2.sf(chi2, 1)

        return chi2, p_value
