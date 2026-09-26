"""Validate baseline metrics and train-only vocabulary fitting."""
import unittest
from training.train_baseline import build_model, evaluate


class BaselineTests(unittest.TestCase):
    def test_validation_does_not_change_vocabulary(self):
        model = build_model()
        model.fit(["cats sleep", "dogs run", "cats rest", "dogs walk"], [0, 1, 0, 1])
        before = dict(model.named_steps["tfidf"].vocabulary_)
        model.predict(["validationonlyword cats"])
        self.assertEqual(before, model.named_steps["tfidf"].vocabulary_)
        self.assertNotIn("validationonlyword", before)

    def test_six_class_report(self):
        report = evaluate([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 0])
        self.assertAlmostEqual(report["accuracy"], 5 / 6)
        self.assertAlmostEqual(report["macro_f1"], (2 / 3 + 4) / 6)
        self.assertEqual(report["per_class"]["pants-fire"]["recall"], 0)
        self.assertEqual(report["confusion_matrix"][5][0], 1)
