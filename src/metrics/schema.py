"""
Metrics schema: Define and validate metrics.json structure
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
from pathlib import Path


@dataclass
class MetricsField:
    """Definition of a metrics field"""
    name: str
    field_type: type
    required: bool = False
    description: str = ""
    min_value: float | None = None
    max_value: float | None = None
    
    def validate(self, value: Any) -> tuple[bool, str]:
        """Validate a single value"""
        if value is None:
            if self.required:
                return False, f"Field '{self.name}' is required but missing"
            else:
                return True, ""
        
        if not isinstance(value, self.field_type):
            return False, f"Field '{self.name}' must be {self.field_type.__name__}, got {type(value).__name__}"
        
        if self.min_value is not None and value < self.min_value:
            return False, f"Field '{self.name}' must be >= {self.min_value}, got {value}"
        
        if self.max_value is not None and value > self.max_value:
            return False, f"Field '{self.name}' must be <= {self.max_value}, got {value}"
        
        return True, ""


# Define metrics schema
METRICS_SCHEMA: Dict[str, MetricsField] = {
    "run_id": MetricsField(
        name="run_id",
        field_type=str,
        required=True,
        description="Unique identifier for the run"
    ),
    "dataset_id": MetricsField(
        name="dataset_id",
        field_type=str,
        required=True,
        description="Dataset identifier (e.g., cifar100-unseen-classes)"
    ),
    "mode": MetricsField(
        name="mode",
        field_type=str,
        required=True,
        description="Mode: 'real', 'baseline', or 'smoke'"
    ),
    "simulated": MetricsField(
        name="simulated",
        field_type=bool,
        required=True,
        description="Whether this was a simulated run"
    ),
    "test_seen_grouped_recall_at_k": MetricsField(
        name="test_seen_grouped_recall_at_k",
        field_type=float,
        required=False,
        min_value=0.0,
        max_value=100.0,
        description="Grouped Recall@K for test seen set"
    ),
    "test_unseen_grouped_recall_at_k": MetricsField(
        name="test_unseen_grouped_recall_at_k",
        field_type=float,
        required=True,
        min_value=0.0,
        max_value=100.0,
        description="Grouped Recall@K for test unseen set"
    ),
    "test_unseen_global_recall_at_1": MetricsField(
        name="test_unseen_global_recall_at_1",
        field_type=float,
        required=False,
        min_value=0.0,
        max_value=100.0,
        description="Global Recall@1 for test unseen set"
    ),
    "test_unseen_random_embedding_global_recall_at_1": MetricsField(
        name="test_unseen_random_embedding_global_recall_at_1",
        field_type=float,
        required=False,
        min_value=0.0,
        max_value=100.0,
        description="Global Recall@1 for random embedding baseline"
    ),
    "model_quality_interpretable": MetricsField(
        name="model_quality_interpretable",
        field_type=bool,
        required=False,
        description="Whether model quality metrics are interpretable"
    ),
    "sanity_warnings": MetricsField(
        name="sanity_warnings",
        field_type=list,
        required=False,
        description="List of sanity check warnings"
    ),
    "composite_score": MetricsField(
        name="composite_score",
        field_type=float,
        required=False,
        min_value=0.0,
        max_value=100.0,
        description="Composite evaluation score"
    ),
}


def validate_metrics(metrics: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate metrics against schema
    
    Args:
        metrics: Dictionary of metrics to validate
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    # Check required fields
    for field_name, field_def in METRICS_SCHEMA.items():
        if field_def.required and field_name not in metrics:
            errors.append(f"Missing required field: {field_name}")
    
    # Validate each field
    for field_name, value in metrics.items():
        if field_name in METRICS_SCHEMA:
            is_valid, error_msg = METRICS_SCHEMA[field_name].validate(value)
            if not is_valid:
                errors.append(error_msg)
        else:
            # Optional: warn about unknown fields
            pass
    
    return len(errors) == 0, errors


def load_metrics(path: Path) -> tuple[Dict[str, Any], bool, List[str]]:
    """Load and validate metrics from JSON file
    
    Args:
        path: Path to metrics.json
        
    Returns:
        Tuple of (metrics_dict, is_valid, error_messages)
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception as e:
        return {}, False, [f"Failed to load metrics file: {e}"]
    
    is_valid, errors = validate_metrics(metrics)
    return metrics, is_valid, errors


def create_empty_metrics(run_id: str, dataset_id: str) -> Dict[str, Any]:
    """Create empty metrics structure with required fields
    
    Args:
        run_id: Run identifier
        dataset_id: Dataset identifier
        
    Returns:
        Dictionary with empty metrics structure
    """
    return {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "mode": "baseline",
        "simulated": False,
        "test_seen_grouped_recall_at_k": None,
        "test_unseen_grouped_recall_at_k": None,
        "test_unseen_global_recall_at_1": None,
        "test_unseen_random_embedding_global_recall_at_1": None,
        "model_quality_interpretable": False,
        "sanity_warnings": [],
        "composite_score": None,
    }
