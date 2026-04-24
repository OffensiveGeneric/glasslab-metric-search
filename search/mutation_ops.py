from __future__ import annotations

import copy
import random
from typing import Any


def _pick(rng: random.Random, values: list[Any], current: Any) -> Any:
    choices = [value for value in values if value != current]
    return rng.choice(choices) if choices else current


def mutate_config(
    config: dict[str, Any],
    search_space: dict[str, Any],
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    mutated = copy.deepcopy(config)
    fields = search_space.get("mutable_fields", {})

    for field_path, choices in fields.items():
        if rng.random() > 0.5:
            continue
        keys = field_path.split(".")
        cursor = mutated
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        leaf = keys[-1]
        cursor[leaf] = _pick(rng, list(choices), cursor.get(leaf))

    return mutated

