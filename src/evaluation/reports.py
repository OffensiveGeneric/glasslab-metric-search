"""
Report utilities: JSON serialization and report generation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


def serialize_metrics(metrics: Dict[str, Any]) -> str:
    """Serialize metrics to JSON, handling numpy types
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        JSON string
    """
    def convert(obj):
        """Convert numpy types to Python types"""
        if hasattr(obj, "dtype"):
            if obj.dtype.kind == "f":
                return float(obj)
            elif obj.dtype.kind in ("i", "u"):
                return int(obj)
            elif obj.dtype.kind == "b":
                return bool(obj)
            else:
                return str(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj
    
    return json.dumps(convert(metrics), indent=2, sort_keys=True)


def generate_report(
    metrics: Dict[str, Any],
    output_dir: Path,
    filename: str = "metrics.json",
) -> Path:
    """Generate metrics report file
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Output directory
        filename: Output filename
        
    Returns:
        Path to generated file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    output_path.write_text(
        serialize_metrics(metrics),
        encoding="utf-8"
    )
    
    return output_path
