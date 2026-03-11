from __future__ import annotations

from pathlib import Path

from recommender import explain_recommendation, load_items, recommend_similar


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    items = load_items(base_dir / "synthetic_items.json")
    print("k-NN Item Recommender (cosine similarity)")
    print("Available item ids:")
    print(", ".join(item["id"] for item in items))
    item_id = input("Select base item id: ").strip()
    recommendations = recommend_similar(items, item_id, top_k=3)
    print("Top recommendations:")
    for candidate_id, score in recommendations:
        print(f"- {candidate_id}: {score:.4f}")
        details = explain_recommendation(items, item_id, candidate_id)
        print(f"  shared_dimensions={details['shared_dimensions']}")


if __name__ == "__main__":
    main()
