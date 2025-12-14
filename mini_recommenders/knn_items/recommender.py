from __future__ import annotations

import json
import math
from pathlib import Path


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def load_items(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def recommend_similar(items: list[dict], item_id: str, top_k: int = 3) -> list[tuple[str, float]]:
    lookup = {item["id"]: item for item in items}
    if item_id not in lookup:
        raise ValueError(f"unknown item_id: {item_id}")
    target = lookup[item_id]["features"]
    scored: list[tuple[str, float]] = []
    for item in items:
        if item["id"] == item_id:
            continue
        score = cosine_similarity(target, item["features"])
        scored.append((item["id"], score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]
