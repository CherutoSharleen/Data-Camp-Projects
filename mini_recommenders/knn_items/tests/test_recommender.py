import unittest

from recommender import cosine_similarity, recommend_similar


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


if __name__ == "__main__":
    unittest.main()
