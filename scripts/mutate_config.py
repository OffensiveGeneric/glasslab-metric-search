#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search.mutation_ops import mutate_config


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    mutated = dict(config)
    mutated["experiment"] = mutate_config(
        config["experiment"],
        config["search_space"],
        seed=args.seed,
    )
    args.output.write_text(yaml.safe_dump(mutated, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
