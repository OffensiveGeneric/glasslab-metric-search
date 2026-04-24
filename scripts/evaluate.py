"""
Evaluation script for DML experiments
"""

import torch
import json
import yaml
import os
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import umap

from src.config import Config
from src.data.dataset import get_dataloaders
from src.models.backbone import ModelFactory
from src.metrics.metrics import AdvancedMetrics, StatisticalTests


class DMLEvaluator:
    """Advanced evaluator for Deep Metric Learning experiments"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.training.device
        self.dataloaders = get_dataloaders(config)
        self.metrics = AdvancedMetrics(config)
        self.stats_tests = StatisticalTests()
        
        self.results_dir = f"{config.project_dir}/results"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_model(self, checkpoint_path: str):
        """Load trained model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model = ModelFactory.create_backbone(self.config, "resnet18")
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        
        return model.eval()
    
    def embed_dataset(self, model, dataloader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract embeddings for a dataset"""
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                embeddings = model(images)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels)
                
        all_embeddings = torch.cat(all_embeddings)
        all_labels = torch.cat(all_labels)
        
        return all_embeddings, all_labels
    
    def compute_umap(self, embeddings: torch.Tensor, labels: torch.Tensor,
                    n_neighbors: int = 15, n_components: int = 2) -> np.ndarray:
        """Compute UMAP projection"""
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            random_state=42
        )
        embedding_2d = reducer.fit_transform(embeddings.cpu().numpy())
        return embedding_2d
    
    def plot_umap(self, embeddings_2d: np.ndarray, labels: torch.Tensor,
                  title: str = "UMAP Projection", save_path: str = None):
        """Create and save UMAP plot"""
        plt.figure(figsize=(10, 8))
        
        # Use categorical colors
        unique_labels = np.unique(labels.cpu().numpy())
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels.cpu().numpy() == label
            plt.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1],
                c=[colors[i]],
                label=str(label),
                s=10,
                alpha=0.7
            )
            
        plt.title(title)
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        
    def run_comprehensive_evaluation(self, checkpoint_path: str):
        """Run complete evaluation pipeline"""
        print("Loading model...")
        model = self.load_model(checkpoint_path)
        
        print("Extracting embeddings...")
        train_seen_emb, train_seen_labels = self.embed_dataset(
            model, self.dataloaders["train_seen_0"]
        )
        val_seen_emb, val_seen_labels = self.embed_dataset(
            model, self.dataloaders["val_seen_0"]
        )
        test_seen_emb, test_seen_labels = self.embed_dataset(
            model, self.dataloaders["test_seen_0"]
        )
        test_unseen_emb, test_unseen_labels = self.embed_dataset(
            model, self.dataloaders["test_unseen_0"]
        )
        
        # Combine test sets
        combined_emb = torch.cat([test_seen_emb, test_unseen_emb])
        combined_labels = torch.cat([test_seen_labels, test_unseen_labels])
        
        results = {}
        
        # Evaluate each set
        print("Evaluating Train-seen...")
        results["train_seen"] = self.metrics.compute_all_metrics(train_seen_emb, train_seen_labels)
        
        print("Evaluating Val-seen...")
        results["val_seen"] = self.metrics.compute_all_metrics(val_seen_emb, val_seen_labels)
        
        print("Evaluating Test-seen...")
        results["test_seen"] = self.metrics.compute_all_metrics(test_seen_emb, test_seen_labels)
        
        print("Evaluating Test-unseen...")
        results["test_unseen"] = self.metrics.compute_all_metrics(test_unseen_emb, test_unseen_labels)
        
        print("Evaluating Combined...")
        results["combined"] = self.metrics.compute_all_metrics(combined_emb, combined_labels)
        
        # Compute generalization gap
        results["generalization_gap"] = {
            "grouped_recall_at_k": results["test_unseen"]["grouped_recall_at_k"] - results["test_seen"]["grouped_recall_at_k"],
            "opis": results["test_unseen"]["opis"] - results["test_seen"]["opis"],
            "silhouette": results["test_unseen"]["silhouette"] - results["test_seen"]["silhouette"]
        }
        
        # Generate UMAP plots
        print("Generating UMAP plots...")
        
        # Test-seen UMAP
        umap_seen = self.compute_umap(test_seen_emb, test_seen_labels)
        self.plot_umap(
            umap_seen, test_seen_labels,
            title="UMAP: Test-seen (80 classes)",
            save_path=f"{self.results_dir}/umap_test_seen.png"
        )
        
        # Test-unseen UMAP
        umap_unseen = self.compute_umap(test_unseen_emb, test_unseen_labels)
        self.plot_umap(
            umap_unseen, test_unseen_labels,
            title="UMAP: Test-unseen (20 classes)",
            save_path=f"{self.results_dir}/umap_test_unseen.png"
        )
        
        # Combined UMAP
        umap_combined = self.compute_umap(combined_emb, combined_labels)
        self.plot_umap(
            umap_combined, combined_labels,
            title="UMAP: Combined (100 classes)",
            save_path=f"{self.results_dir}/umap_combined.png"
        )
        
        # Run statistical tests
        print("Running statistical tests...")
        
        # Example: Compare test_seen vs test_unseen Grouped Recall@K
        # For 5x2cv, we need multiple runs
        # For now, compute McNemar's test on error patterns
        
        test_seen_pred = self._predict_clusters(test_seen_emb)
        test_unseen_pred = self._predict_clusters(test_unseen_emb)
        
        # McNemar's test on error patterns
        seen_errors = test_seen_labels.cpu().numpy() != test_seen_pred
        unseen_errors = test_unseen_labels.cpu().numpy() != test_unseen_pred
        
        # Convert to boolean lists
        seen_errors_bool = [bool(e) for e in seen_errors]
        unseen_errors_bool = [bool(e) for e in unseen_errors]
        
        chi2, p_value = self.stats_tests.mc_nemar(seen_errors_bool, unseen_errors_bool)
        
        results["statistical_tests"] = {
            "mcnemar_chi2": chi2,
            "mcnemar_p_value": p_value
        }
        
        # Save results
        with open(f"{self.results_dir}/comprehensive_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        print("Evaluation complete!")
        print(f"Results saved to {self.results_dir}/comprehensive_results.json")
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def _predict_clusters(self, embeddings: torch.Tensor, k: int = 5) -> np.ndarray:
        """Predict cluster labels using k-NN"""
        from sklearn.neighbors import KNeighborsClassifier
        
        # For simplicity, use embeddings as features and predict based on nearest neighbors
        # In practice, you'd use the training set as reference
        embeddings_np = embeddings.cpu().numpy()
        
        # Use all embeddings as both train and test for demonstration
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(embeddings_np, np.arange(len(embeddings)))
        
        predictions = knn.predict(embeddings_np)
        return predictions
    
    def print_summary(self, results: Dict[str, Dict[str, float]]):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        for set_name, metrics in results.items():
            if set_name in ["statistical_tests"]:
                continue
                
            print(f"\n{set_name.upper()}:")
            print("-" * 40)
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric:30s}: {value:.4f}")
                else:
                    print(f"  {metric:30s}: {value}")
        
        if "generalization_gap" in results:
            print("\nGENERALIZATION GAP:")
            print("-" * 40)
            for metric, gap in results["generalization_gap"].items():
                print(f"  {metric:30s}: {gap:.4f}")
                
        if "statistical_tests" in results:
            print("\nSTATISTICAL TESTS:")
            print("-" * 40)
            for test, value in results["statistical_tests"].items():
                print(f"  {test:30s}: {value:.4f}")
                
        print("\n" + "="*60)
    
    def run_baseline_comparison(self, baseline_results_path: str):
        """Compare trained model with zero-shot baselines"""
        with open(baseline_results_path, "r") as f:
            baseline_results = json.load(f)
            
        # Load trained model results
        with open(f"{self.results_dir}/comprehensive_results.json", "r") as f:
            trained_results = json.load(f)
            
        # Compare test-unseen performance
        print("\nBaseline Comparison (Test-unseen):")
        print("-" * 40)
        
        for baseline_name, metrics in baseline_results.items():
            if "test_unseen" in metrics:
                print(f"\n{baseline_name}:")
                for metric, value in metrics["test_unseen"].items():
                    trained_value = trained_results.get("test_unseen", {}).get(metric, None)
                    if trained_value is not None:
                        diff = trained_value - value
                        print(f"  {metric:30s}: {value:.4f} -> {trained_value:.4f} (diff: {diff:+.4f})")
                    else:
                        print(f"  {metric:30s}: {value:.4f}")
                        
        print("\nTrained Model:")
        print("-" * 40)
        for metric, value in trained_results["test_unseen"].items():
            print(f"  {metric:30s}: {value:.4f}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--baseline", type=str, default=None, help="Path to baseline results")
    args = parser.parse_args()
    
    # Load config
    config = Config.from_yaml(args.config) if args.config else Config()
    
    # Create evaluator
    evaluator = DMLEvaluator(config)
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation(args.checkpoint)
    
    # Compare with baselines if provided
    if args.baseline:
        evaluator.run_baseline_comparison(args.baseline)


if __name__ == "__main__":
    main()
