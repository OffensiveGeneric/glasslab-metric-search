"""
Metrics schema tests: Validate metrics.json structure
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest

from src.metrics.schema import (
    METRICS_SCHEMA,
    validate_metrics,
    load_metrics,
    create_empty_metrics,
)


class TestMetricsSchema:
    """Test metrics schema validation"""
    
    def test_empty_metrics_valid(self):
        """Empty metrics with required fields should validate"""
        metrics = create_empty_metrics("test-run-1", "cifar100-unseen-classes")
        is_valid, errors = validate_metrics(metrics)
        
        assert is_valid, f"Empty metrics should be valid, got errors: {errors}"
        assert metrics["run_id"] == "test-run-1"
        assert metrics["dataset_id"] == "cifar100-unseen-classes"
        assert metrics["mode"] == "baseline"
        assert metrics["simulated"] is False
    
    def test_missing_required_field(self):
        """Missing required field should fail validation"""
        metrics = {
            "run_id": "test-run-1",
            # missing dataset_id
            "mode": "baseline",
            "simulated": False,
        }
        
        is_valid, errors = validate_metrics(metrics)
        
        assert not is_valid
        assert any("dataset_id" in error for error in errors)
    
    def test_invalid_field_type(self):
        """Invalid field type should fail validation"""
        metrics = create_empty_metrics("test-run-1", "cifar100-unseen-classes")
        metrics["simulated"] = "yes"  # Should be bool, not string
        
        is_valid, errors = validate_metrics(metrics)
        
        assert not is_valid
        assert any("simulated" in error for error in errors)
    
    def test_out_of_range_value(self):
        """Value out of range should fail validation"""
        metrics = create_empty_metrics("test-run-1", "cifar100-unseen-classes")
        metrics["test_unseen_grouped_recall_at_k"] = 150.0  # > 100
        
        is_valid, errors = validate_metrics(metrics)
        
        assert not is_valid
        assert any("grouped_recall_at_k" in error for error in errors)
    
    def test_valid_metrics(self):
        """Valid metrics should pass validation"""
        metrics = create_empty_metrics("test-run-1", "cifar100-unseen-classes")
        metrics["test_unseen_grouped_recall_at_k"] = 85.5
        metrics["test_unseen_global_recall_at_1"] = 45.2
        metrics["composite_score"] = 0.85
        metrics["model_quality_interpretable"] = True
        metrics["sanity_warnings"] = []
        
        is_valid, errors = validate_metrics(metrics)
        
        assert is_valid, f"Valid metrics should pass validation, got errors: {errors}"


class TestMetricsFileLoading:
    """Test loading and validating metrics from files"""
    
    def test_load_valid_metrics(self):
        """Load valid metrics file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.json"
            metrics = create_empty_metrics("test-run-1", "cifar100-unseen-classes")
            metrics["test_unseen_grouped_recall_at_k"] = 85.5
            
            with open(metrics_path, "w") as f:
                json.dump(metrics, f)
            
            loaded, is_valid, errors = load_metrics(metrics_path)
            
            assert is_valid, f"Loaded metrics should be valid: {errors}"
            assert loaded["test_unseen_grouped_recall_at_k"] == 85.5
            assert loaded["run_id"] == "test-run-1"
    
    def test_load_invalid_metrics(self):
        """Load invalid metrics file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.json"
            metrics = create_empty_metrics("test-run-1", "cifar100-unseen-classes")
            metrics["test_unseen_grouped_recall_at_k"] = 150.0  # Invalid range
            
            with open(metrics_path, "w") as f:
                json.dump(metrics, f)
            
            loaded, is_valid, errors = load_metrics(metrics_path)
            
            assert not is_valid
            assert any("grouped_recall_at_k" in error for error in errors)
    
    def test_load_missing_required_field(self):
        """Load metrics file missing required field"""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.json"
            metrics = {"run_id": "test-run-1"}  # Missing required fields
            
            with open(metrics_path, "w") as f:
                json.dump(metrics, f)
            
            loaded, is_valid, errors = load_metrics(metrics_path)
            
            assert not is_valid
            assert any("dataset_id" in error for error in errors)


class TestMetricsSchemaFields:
    """Test individual metrics schema fields"""
    
    def test_required_fields(self):
        """Check all required fields are defined"""
        required_fields = {
            "run_id",
            "dataset_id",
            "mode",
            "simulated",
        }
        
        actual_required = {
            name for name, field_def in METRICS_SCHEMA.items()
            if field_def.required
        }
        
        assert required_fields == actual_required, (
            f"Required fields mismatch. Expected {required_fields}, "
            f"got {actual_required}"
        )
    
    def test_range_validations(self):
        """Check range validations are correct"""
        grouped_recall_field = METRICS_SCHEMA["test_unseen_grouped_recall_at_k"]
        assert grouped_recall_field.min_value == 0.0
        assert grouped_recall_field.max_value == 100.0
    
    def test_boolean_field(self):
        """Check boolean fields are defined correctly"""
        simulated_field = METRICS_SCHEMA["simulated"]
        assert simulated_field.field_type is bool
        assert simulated_field.required is True
