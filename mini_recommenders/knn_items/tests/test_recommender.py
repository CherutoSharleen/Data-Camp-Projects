import unittest

from recommender import (
    cosine_similarity,
    explain_recommendation,
    recommend_similar,
    recommend_with_min_similarity,
    vector_length,
)


class RecommenderTests(unittest.TestCase):
    def test_cosine_similarity_bounds(self) -> None:
        score = cosine_similarity([1.0, 0.0], [0.5, 0.5])
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_recommend_returns_ranked_items(self) -> None:
        items = [
            {"id": "a", "features": [1.0, 0.0, 0.0]},
            {"id": "b", "features": [0.9, 0.1, 0.0]},
            {"id": "c", "features": [0.1, 0.9, 0.0]},
        ]
        recs = recommend_similar(items, "a", top_k=2)
        self.assertEqual(recs[0][0], "b")

    def test_explanation_contains_similarity_metrics(self) -> None:
        items = [
            {"id": "a", "features": [1.0, 0.2, 0.0]},
            {"id": "b", "features": [0.9, 0.1, 0.1]},
        ]
        explanation = explain_recommendation(items, "a", "b")
        self.assertIn("cosine_similarity", explanation)
        self.assertGreater(explanation["shared_dimensions"], 0)

    def test_recommend_with_min_similarity(self) -> None:
        items = [
            {"id": "a", "features": [1.0, 0.0, 0.0]},
            {"id": "b", "features": [0.9, 0.1, 0.0]},
            {"id": "c", "features": [0.1, 0.9, 0.0]},
        ]
        recs = recommend_with_min_similarity(items, "a", min_similarity=0.5, top_k=3)
        self.assertEqual([r[0] for r in recs], ["b"])

    def test_vector_length(self) -> None:
        self.assertEqual(vector_length([3.0, 4.0]), 5.0)


if __name__ == "__main__":
    unittest.main()
