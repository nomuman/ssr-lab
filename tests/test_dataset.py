"""Dataset validation for data/*.json — pure stdlib, no model needed.

Every concept file data/<slug>.json must have:
  - concept (str), prices (exactly 2, ascending), personas, responses
  - one response per persona x price, no duplicates
  - data/<slug>.gold.json covering exactly those responses, values in 1..5
  - data/<slug>.extracts.json covering exactly those responses, non-empty
"""
import glob
import json
import os
import unittest

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

def concept_files():
    return sorted(f for f in glob.glob(os.path.join(DATA, "*.json"))
                  if ".gold." not in f and ".extracts." not in f)

class TestDataset(unittest.TestCase):
    def test_concepts_exist(self):
        self.assertGreaterEqual(len(concept_files()), 2)

    def test_each_concept(self):
        for path in concept_files():
            with self.subTest(concept=path):
                data = json.load(open(path, encoding="utf-8"))
                self.assertIsInstance(data["concept"], str)
                prices = data["prices"]
                self.assertEqual(len(prices), 2)
                self.assertLess(prices[0], prices[1])
                pids = [p["id"] for p in data["personas"]]
                self.assertEqual(len(pids), len(set(pids)))
                keys = set()
                for r in data["responses"]:
                    self.assertIn(r["persona_id"], pids)
                    self.assertIn(r["price"], prices)
                    self.assertTrue(r["text"].strip())
                    keys.add(f"{r['persona_id']}@{r['price']}")
                self.assertEqual(len(keys), len(pids) * 2)
                gold = json.load(open(path.replace(".json", ".gold.json"), encoding="utf-8"))
                self.assertEqual(set(gold), keys)
                for v in gold.values():
                    self.assertIn(v, (1, 2, 3, 4, 5))
                ext = json.load(open(path.replace(".json", ".extracts.json"), encoding="utf-8"))
                self.assertEqual(set(ext), keys)
                for v in ext.values():
                    self.assertTrue(v.strip())

if __name__ == "__main__":
    unittest.main()
