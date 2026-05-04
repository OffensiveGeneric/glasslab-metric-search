from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from search.run_spec import RunSpec
from src.config import Config
from src.data.dataset import get_dataloaders
from src.losses.losses import SupervisedContrastiveLoss, TripletLoss
from src.models.backbone import ModelFactory
from src.metrics.metrics import AdvancedMetrics


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    max_batches: int = 100,
) -> float:
    """Train for one epoch with early stopping for speed."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        embeddings = model(images)

        if isinstance(loss_fn, SupervisedContrastiveLoss):
            loss = loss_fn(embeddings, labels)
        elif isinstance(loss_fn, TripletLoss):
            batch_size = embeddings.shape[0]
            anchor = embeddings
            positive = torch.zeros_like(embeddings)
            negative = torch.zeros_like(embeddings)

            for i in range(batch_size):
                same_class = torch.where(labels == labels[i])[0]
                diff_class = torch.where(labels != labels[i])[0]

                if len(same_class) > 1:
                    same_class = same_class[same_class != i]
                    positive[i] = embeddings[same_class[0]]
                else:
                    positive[i] = embeddings[i]

                if len(diff_class) > 0:
                    negative[i] = embeddings[diff_class[0]]
                else:
                    negative[i] = embeddings[(i + 1) % batch_size]

            loss = loss_fn(anchor, positive, negative)
        else:
            loss = loss_fn(embeddings, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate_metrics(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    device: str,
    config: Config,
) -> Dict[str, float]:
    """Evaluate on test set and compute metrics."""
    model.eval()
    all_embeddings = []
    all_labels = []

    test_loader = dataloaders.get("test_seen_0")
    if test_loader is None:
        return {}

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.detach().cpu())
            all_labels.append(labels.detach().cpu())

    embeddings = torch.cat(all_embeddings)
    labels = torch.cat(all_labels)

    metrics_fn = AdvancedMetrics(config)
    return metrics_fn.compute_all_metrics(embeddings, labels)


METRIC_ALIASES = {
    "ami": "adjusted_mutual_info",
    "ari": "adjusted_rand_index",
    "nmi": "normalized_mutual_info",
    "silhouette": "silhouette_score",
}


SUMMARY_ALIAS_SOURCES = {
    "grouped_recall_at_k": "test_unseen_grouped_recall_at_k",
    "opis": "test_unseen_opis",
    "adjusted_mutual_info": "test_unseen_adjusted_mutual_info",
    "adjusted_rand_index": "test_unseen_adjusted_rand_index",
    "normalized_mutual_info": "test_unseen_normalized_mutual_info",
    "silhouette_score": "test_unseen_silhouette_score",
    "composite_score": "test_unseen_composite_score",
}


def composite_score(metrics: dict[str, Any]) -> float | None:
    required = (
        "grouped_recall_at_k",
        "opis",
        "adjusted_mutual_info",
        "adjusted_rand_index",
        "normalized_mutual_info",
        "silhouette_score",
    )
    if any(metrics.get(key) is None for key in required):
        return None
    return round(
        (
            float(metrics.get("grouped_recall_at_k", 0.0))
            + (1.0 - float(metrics.get("opis", 0.0)))
            + float(metrics.get("adjusted_mutual_info", 0.0))
            + float(metrics.get("adjusted_rand_index", 0.0))
            + float(metrics.get("normalized_mutual_info", 0.0))
            + float(metrics.get("silhouette_score", 0.0))
        )
        / 6.0,
        4,
    )


def normalize_metric_keys(metrics: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in metrics.items():
        if isinstance(value, bool) or value is None:
            normalized[METRIC_ALIASES.get(key, key)] = value
        else:
            normalized[METRIC_ALIASES.get(key, key)] = float(value)
    normalized["composite_score"] = composite_score(normalized)
    return normalized


def prefix_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def class_count_metadata(labels: torch.Tensor, prefix: str) -> dict[str, int | float | None]:
    labels_cpu = labels.detach().cpu()
    if labels_cpu.numel() == 0:
        return {
            f"{prefix}_num_samples": 0,
            f"{prefix}_num_classes": 0,
            f"{prefix}_min_samples_per_class": None,
            f"{prefix}_max_samples_per_class": None,
            f"{prefix}_mean_samples_per_class": None,
        }

    _, counts = torch.unique(labels_cpu, return_counts=True)
    counts_f = counts.to(torch.float32)
    return {
        f"{prefix}_num_samples": int(labels_cpu.numel()),
        f"{prefix}_num_classes": int(counts.numel()),
        f"{prefix}_min_samples_per_class": int(counts.min().item()),
        f"{prefix}_max_samples_per_class": int(counts.max().item()),
        f"{prefix}_mean_samples_per_class": round(float(counts_f.mean().item()), 4),
    }


def gallery_metadata(
    labels: torch.Tensor,
    prefix: str,
    loader: DataLoader | None,
    max_batches: int | None,
) -> dict[str, int | bool | None]:
    labels_cpu = labels.detach().cpu()
    total_samples = len(loader.dataset) if loader is not None and hasattr(loader, "dataset") else None
    batch_size = getattr(loader, "batch_size", None)
    partial = False
    if max_batches is not None:
        partial = True
        if total_samples is not None and batch_size is not None:
            partial = int(labels_cpu.numel()) < int(total_samples)
    return {
        f"{prefix}_gallery_num_samples": int(labels_cpu.numel()),
        f"{prefix}_gallery_num_classes": int(torch.unique(labels_cpu).numel()) if labels_cpu.numel() else 0,
        f"{prefix}_gallery_total_dataset_samples": int(total_samples) if total_samples is not None else None,
        f"{prefix}_gallery_max_eval_batches": int(max_batches) if max_batches is not None else None,
        f"{prefix}_gallery_partial": partial,
    }


def evaluate_embeddings(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    config: Config,
    warnings: list[str],
    context: str,
) -> dict[str, Any]:
    if embeddings.numel() == 0 or labels.numel() == 0:
        warnings.append(f"{context} has no embeddings or labels; metrics are null")
        return normalize_metric_keys(
            {
                "grouped_recall_at_k": None,
                "opis": None,
                "ami": None,
                "ari": None,
                "nmi": None,
                "silhouette": None,
            }
        )

    num_classes = int(torch.unique(labels.detach().cpu()).numel())
    if num_classes < 2:
        warnings.append(f"{context} has fewer than two classes; metrics are null")
        return normalize_metric_keys(
            {
                "grouped_recall_at_k": None,
                "opis": None,
                "ami": None,
                "ari": None,
                "nmi": None,
                "silhouette": None,
            }
        )

    try:
        return normalize_metric_keys(AdvancedMetrics(config).compute_all_metrics(embeddings, labels))
    except Exception as exc:
        warnings.append(f"{context} metrics could not be computed: {exc}")
        return normalize_metric_keys(
            {
                "grouped_recall_at_k": None,
                "opis": None,
                "ami": None,
                "ari": None,
                "nmi": None,
                "silhouette": None,
            }
        )


def collect_embeddings(
    model: nn.Module,
    loader: DataLoader | None,
    device: str,
    max_batches: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if loader is None:
        return torch.empty(0), torch.empty(0, dtype=torch.long)

    model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.detach().cpu())
            all_labels.append(labels.detach().cpu())

    if not all_embeddings:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(all_embeddings), torch.cat(all_labels)


def shuffled_label_baseline(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    config: Config,
    warnings: list[str],
    context: str,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(int(config.data.seed))
    shuffled = labels[torch.randperm(labels.numel(), generator=generator)]
    return evaluate_embeddings(embeddings, shuffled, config, warnings, context)


def random_embedding_baseline(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    config: Config,
    warnings: list[str],
    context: str,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(int(config.data.seed) + 17)
    random_embeddings = torch.randn(
        embeddings.shape,
        generator=generator,
        dtype=embeddings.dtype if embeddings.numel() else torch.float32,
    )
    metrics = evaluate_embeddings(random_embeddings, labels, config, warnings, context)
    random_global_recall = metrics.get("global_recall_at_1")
    random_global_chance_exact = metrics.get("global_recall_at_1_chance_exact")
    random_global_chance_approx = metrics.get("global_recall_at_1_chance_approx")
    random_grouped_recall = metrics.get("grouped_recall_at_k")
    random_grouped_chance = metrics.get("grouped_recall_chance_at_k")
    
    metrics["global_recall_at_1_expected_target"] = random_global_chance_exact
    metrics["global_recall_at_1_abs_error"] = (
        None
        if random_global_recall is None or random_global_chance_exact is None
        else round(float(abs(random_global_recall - random_global_chance_exact)), 4)
    )
    metrics["global_recall_at_1_sanity_pass"] = (
        False
        if metrics["global_recall_at_1_abs_error"] is None
        else metrics["global_recall_at_1_abs_error"] <= 0.03
    )
    metrics["grouped_recall_at_k_expected_target"] = random_grouped_chance
    metrics["grouped_recall_at_k_abs_error"] = (
        None
        if random_grouped_recall is None or random_grouped_chance is None
        else round(float(abs(random_grouped_recall - random_grouped_chance)), 4)
    )
    metrics["global_recall_at_1_sanity_target_approx"] = random_global_chance_approx
    metrics["global_recall_at_1_chance_approx"] = random_global_chance_approx
    return metrics


def baseline_sane(
    observed: Any,
    chance: Any,
    *,
    absolute_tolerance: float = 0.10,
    relative_tolerance: float = 0.50,
) -> bool:
    if observed is None or chance is None:
        return False
    observed_f = float(observed)
    chance_f = float(chance)
    tolerance = max(absolute_tolerance, abs(chance_f) * relative_tolerance)
    return abs(observed_f - chance_f) <= tolerance


def append_runner_log(output_dir: Path, message: str) -> None:
    log_path = output_dir / "logs" / "runner.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def equalized_seen_subset(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    target_num_classes: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    unique = torch.unique(labels.detach().cpu()).sort().values
    if target_num_classes <= 0 or unique.numel() < target_num_classes:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    generator = torch.Generator().manual_seed(int(seed))
    selected = unique[torch.randperm(unique.numel(), generator=generator)[:target_num_classes]]
    mask = torch.isin(labels.detach().cpu(), selected)
    return embeddings[mask], labels[mask]


def add_summary_aliases(metrics: dict[str, Any]) -> None:
    for alias, source in SUMMARY_ALIAS_SOURCES.items():
        metrics[alias] = metrics.get(source)


def apply_run_config_overrides(config: Config, run_config: dict[str, Any], budget: Any) -> None:
    if "backbone_name" in run_config:
        config.model.backbones = [run_config["backbone_name"]]
    if "batch_size" in run_config:
        config.training.batch_size = int(run_config["batch_size"])
    if "learning_rate" in run_config:
        config.training.learning_rate = float(run_config["learning_rate"])
    if "max_epochs" in run_config:
        config.training.epochs = int(run_config["max_epochs"])
    if hasattr(budget, "max_epochs") and budget.max_epochs is not None:
        config.training.epochs = int(budget.max_epochs)


def run_real_experiment(run_spec: RunSpec, output_dir: Path) -> Dict[str, Any]:
    """Run a real training experiment on CIFAR-100."""
    import time as t
    print(f"Starting run_real_experiment", file=sys.stderr)
    print(f"Time {t.time()}: Starting", file=sys.stderr)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Time {t.time()}: Output dir created", file=sys.stderr)

    config_path = output_dir / "config.yaml"
    if config_path.exists():
        config = Config.from_yaml(str(config_path))
    else:
        config = Config.from_yaml("configs/search_spaces/cifar100_contrastive_v0.yaml")

    if run_spec.config:
        from src.config import DataConfig, AugmentationConfig, LossConfig, ModelConfig, TrainingConfig, HPOConfig, EvaluationConfig, L2ANCConfig
        
        for key, value in run_spec.config.items():
            if key == "data":
                if isinstance(config.data, DataConfig):
                    config.data = DataConfig(**value)
                else:
                    config.data = DataConfig(**value)
            elif key == "augmentation":
                if isinstance(config.augmentation, AugmentationConfig):
                    config.augmentation = AugmentationConfig(**value)
                else:
                    config.augmentation = AugmentationConfig(**value)
            elif key == "loss":
                loss_name = run_spec.config.get("loss_name", value.get("name", "contrastive"))
                if loss_name == "contrastive":
                    config.loss = LossConfig(contrastive=value)
                elif loss_name == "triplet":
                    config.loss = LossConfig(triplet=value)
                else:
                    config.loss = LossConfig(**{loss_name: value})
            elif key == "model":
                if isinstance(config.model, ModelConfig):
                    config.model = ModelConfig(**value)
                else:
                    config.model = ModelConfig(**value)
            elif key == "training":
                if isinstance(config.training, TrainingConfig):
                    config.training = TrainingConfig(**value)
                else:
                    config.training = TrainingConfig(**value)
            elif key == "hpo":
                if isinstance(config.hpo, HPOConfig):
                    config.hpo = HPOConfig(**value)
                else:
                    config.hpo = HPOConfig(**value)
            elif key == "evaluation":
                if isinstance(config.evaluation, EvaluationConfig):
                    config.evaluation = EvaluationConfig(**value)
                else:
                    config.evaluation = EvaluationConfig(**value)
            elif key == "l2anc":
                if isinstance(config.l2anc, L2ANCConfig):
                    config.l2anc = L2ANCConfig(**value)
                else:
                    config.l2anc = L2ANCConfig(**value)
            elif hasattr(config, key):
                setattr(config, key, value)
        flat_config_map = {
            "backbone_name": lambda v: setattr(config.model, "backbones", [v] if isinstance(v, str) else v),
            "loss_name": lambda v: None,
            "batch_size": lambda v: setattr(config.training, "batch_size", v),
            "learning_rate": lambda v: setattr(config.training, "learning_rate", v),
            "max_epochs": lambda v: setattr(config.training, "epochs", v),
            "temperature": lambda v: config.loss.contrastive.__setitem__("temperature", v) if isinstance(config.loss.contrastive, dict) else None,
        }
        
        for key, value in run_spec.config.items():
            if key not in ["data", "augmentation", "loss", "model", "training", "hpo", "evaluation", "l2anc"]:
                if key in flat_config_map:
                    flat_config_map[key](value)
        
        loss_name = run_spec.config.get("loss_name", "contrastive")
        if loss_name != "contrastive":
            raise NotImplementedError(f"Only contrastive loss is supported in real mode, got: {loss_name}")
    apply_run_config_overrides(config, run_spec.config or {}, run_spec.budget)

    dataloaders = get_dataloaders(config)
    train_loader = dataloaders["train_seen_0"]

    device = config.training.device
    if not torch.cuda.is_available():
        device = "cpu"

    backbone_name = config.model.backbones[0] if config.model.backbones else "resnet18"
    model = ModelFactory.create_backbone(config, backbone_name)
    model = model.to(device)

    loss_config = config.loss.contrastive if hasattr(config.loss.contrastive, "get") else {}
    loss_fn = SupervisedContrastiveLoss(temperature=loss_config.get("temperature", 0.1))

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    num_epochs = min(run_spec.budget.max_epochs if run_spec.budget.max_epochs is not None else config.training.epochs, 2)
    max_train_batches = run_spec.budget.max_train_batches if run_spec.budget.max_train_batches is not None else 100
    for epoch in range(num_epochs):
        epoch_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, max_train_batches)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    max_eval_batches = run_spec.budget.max_eval_batches if run_spec.budget.max_eval_batches is not None else 100

    # Save checkpoints
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoints_dir / "final_model.pt")
    
    # Save embeddings dir
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    max_eval_batches = run_spec.budget.max_eval_batches
    split_loaders = {
        "val_seen": dataloaders.get("val_seen_0"),
        "test_seen": dataloaders.get("test_seen_0"),
        "test_unseen": dataloaders.get("test_unseen_0"),
    }
    split_payloads: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    metrics: dict[str, Any] = {}
    sanity_warnings: list[str] = []

    for split_name, loader in split_loaders.items():
        print(f"Time {t.time()}: Collecting embeddings for {split_name}", file=sys.stderr)
        embeddings, labels = collect_embeddings(model, loader, device, max_eval_batches)
        print(f"Time {t.time()}: Collected embeddings for {split_name}", file=sys.stderr)
        split_payloads[split_name] = (embeddings, labels)
        if embeddings.numel():
            torch.save(embeddings, embeddings_dir / f"{split_name}_embeddings.pt")
            torch.save(labels, embeddings_dir / f"{split_name}_labels.pt")

        metrics.update(class_count_metadata(labels, split_name))
        metrics.update(gallery_metadata(labels, split_name, loader, max_eval_batches))
        print(f"Time {t.time()}: Computing metrics for {split_name}", file=sys.stderr)
        split_metrics = evaluate_embeddings(embeddings, labels, config, sanity_warnings, split_name)
        metrics.update(prefix_metrics(split_name, split_metrics))

        shuffled_metrics = shuffled_label_baseline(
            embeddings,
            labels,
            config,
            sanity_warnings,
            f"{split_name} shuffled-label baseline",
        )
        metrics.update(prefix_metrics(f"{split_name}_shuffled_label", shuffled_metrics))

        random_metrics = random_embedding_baseline(
            embeddings,
            labels,
            config,
            sanity_warnings,
            f"{split_name} random-embedding baseline",
        )
        metrics.update(prefix_metrics(f"{split_name}_random_embedding", random_metrics))

        if split_name == "test_unseen":
            random_recall = random_metrics.get("grouped_recall_at_k")
            random_chance = random_metrics.get("grouped_recall_chance_at_k")
            message = (
                "INFO test_unseen random baseline "
                f"grouped_recall_at_k={random_recall} chance={random_chance}"
            )
            print(message, file=sys.stderr, flush=True)
            append_runner_log(output_dir, message)

    if metrics.get("test_unseen_grouped_recall_at_k") is None:
        raise RuntimeError(
            "grouped_recall_at_k metric is missing. "
            "The evaluation pipeline did not produce expected metrics. "
            "Please check the dataset and model configuration."
        )

    for split_name in ("test_unseen", "test_seen"):
        real = metrics.get(f"{split_name}_grouped_recall_at_k")
        shuffled = metrics.get(f"{split_name}_shuffled_label_grouped_recall_at_k")
        random_metric = metrics.get(f"{split_name}_random_embedding_grouped_recall_at_k")
        metrics[f"{split_name}_grouped_recall_lift_vs_shuffled_labels"] = (
            None if real is None or shuffled is None else round(float(real) - float(shuffled), 4)
        )
        metrics[f"{split_name}_grouped_recall_lift_vs_random_embeddings"] = (
            None if real is None or random_metric is None else round(float(real) - float(random_metric), 4)
        )

    test_seen_embeddings, test_seen_labels = split_payloads["test_seen"]
    test_unseen_embeddings, test_unseen_labels = split_payloads["test_unseen"]
    unseen_num_classes = int(metrics.get("test_unseen_num_classes") or 0)
    equalized_embeddings, equalized_labels = equalized_seen_subset(
        test_seen_embeddings,
        test_seen_labels,
        unseen_num_classes,
        config.data.seed,
    )
    equalized_metadata = class_count_metadata(equalized_labels, "test_seen_equalized")
    metrics.update(
        {
            "test_seen_equalized_num_classes": equalized_metadata["test_seen_equalized_num_classes"],
            "test_seen_equalized_num_samples": equalized_metadata["test_seen_equalized_num_samples"],
            "test_unseen_equalized_reference_num_classes": metrics.get("test_unseen_num_classes"),
            "test_unseen_equalized_reference_grouped_recall_at_k": metrics.get(
                "test_unseen_grouped_recall_at_k"
            ),
        }
    )
    equalized_metrics = evaluate_embeddings(
        equalized_embeddings,
        equalized_labels,
        config,
        sanity_warnings,
        "test_seen equalized",
    )
    metrics["test_seen_equalized_grouped_recall_at_k"] = equalized_metrics.get("grouped_recall_at_k")
    metrics["test_seen_equalized_composite_score"] = equalized_metrics.get("composite_score")
    if equalized_embeddings.numel():
        torch.save(equalized_embeddings, embeddings_dir / "test_seen_equalized_embeddings.pt")
        torch.save(equalized_labels, embeddings_dir / "test_seen_equalized_labels.pt")
    else:
        sanity_warnings.append("test_seen could not be equalized to test_unseen class count")
    metrics["generalization_gap_equalized_grouped_recall_at_k"] = (
        None
        if metrics.get("test_seen_equalized_grouped_recall_at_k") is None
        or metrics.get("test_unseen_grouped_recall_at_k") is None
        else round(
            float(metrics["test_seen_equalized_grouped_recall_at_k"])
            - float(metrics["test_unseen_grouped_recall_at_k"]),
            4,
        )
    )

    if metrics.get("test_unseen_num_classes") != metrics.get("test_seen_num_classes"):
        sanity_warnings.append(
            "test_unseen has fewer classes than test_seen; raw grouped recall is not directly comparable"
        )
    if max_eval_batches is not None and max_eval_batches <= 5:
        sanity_warnings.append("max_eval_batches is small; metrics may be noisy")
    for split_name in ("test_seen", "test_unseen"):
        real = metrics.get(f"{split_name}_grouped_recall_at_k")
        shuffled = metrics.get(f"{split_name}_shuffled_label_grouped_recall_at_k")
        random_metric = metrics.get(f"{split_name}_random_embedding_grouped_recall_at_k")
        if real is not None and shuffled is not None and abs(float(real) - float(shuffled)) < 0.05:
            sanity_warnings.append(
                f"{split_name} shuffled-label baseline is close to real metric; metric may not reflect learned class structure"
            )
        if real is not None and random_metric is not None and abs(float(real) - float(random_metric)) < 0.05:
            sanity_warnings.append(
                f"{split_name} random-embedding baseline is close to real metric; model may not be learning useful embeddings"
            )
        silhouette = metrics.get(f"{split_name}_silhouette_score")
        if silhouette is not None and float(silhouette) < 0:
            sanity_warnings.append(f"{split_name} has negative silhouette score; embedding clusters are not well separated")

    model_quality_interpretable = True
    for split_name in ("test_seen", "test_unseen"):
        random_recall = metrics.get(f"{split_name}_random_embedding_grouped_recall_at_k")
        random_chance = metrics.get(f"{split_name}_random_embedding_grouped_recall_chance_at_k")
        if not baseline_sane(random_recall, random_chance):
            model_quality_interpretable = False
            sanity_warnings.append(
                f"{split_name} random-embedding grouped recall is not near analytic chance; model quality is not interpretable"
            )

        random_global_recall = metrics.get(f"{split_name}_random_embedding_global_recall_at_1")
        random_global_chance = metrics.get(f"{split_name}_random_embedding_global_recall_at_1_chance_exact")
        if random_global_recall is not None and random_global_chance is not None:
            random_global_abs_error = abs(float(random_global_recall) - float(random_global_chance))
            if random_global_abs_error > 0.03:
                sanity_warnings.append(
                    f"{split_name} random-embedding global Recall@1 ({random_global_recall:.4f}) "
                    f"not near exact chance ({random_global_chance:.4f}); error={random_global_abs_error:.4f}"
                )

    metrics["model_quality_interpretable"] = model_quality_interpretable

    add_summary_aliases(metrics)

    metrics["run_id"] = run_spec.run_id
    metrics["dataset_id"] = run_spec.dataset.dataset_id
    metrics["mode"] = "real"
    metrics["simulated"] = False
    metrics["warning"] = None
    metrics["sanity_warnings"] = sorted(set(sanity_warnings))

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return metrics


def run_contrastive_experiment(run_spec: RunSpec, output_dir: Path) -> Dict[str, Any]:
    """Run a real contrastive learning experiment."""
    return run_real_experiment(run_spec, output_dir)
