"""Game-note features: extractors read what they claim, and the text score never sees its own season's outcomes."""
import math
from pathlib import Path
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import note_features as nf


class ExtractorTests(unittest.TestCase):
    def test_status_counts_read_only_the_status_section(self):
        text = ("Timeout out of bounds. INJURY REPORT Tatum (ankle) is OUT. Brown (knee) is questionable; "
                "Smith (G League - Two-Way) out\nJones probable")
        self.assertEqual(nf.status_counts(text), {"out": 1, "questionable": 1, "probable": 1})
        self.assertEqual(nf.status_counts("No section here: out, questionable."), {"out": 0, "questionable": 0, "probable": 0})

    def test_numbers(self):
        text = ("The Celtics have won five straight and are 7-3 in their last 10. Boston leads the season series, 2-0. "
                "Earlier they had a 9-game losing streak. They have won 8 of their last 12.")
        out = nf.numbers(text)
        self.assertEqual(out["streak"], 5.0)               # first stated streak
        self.assertEqual(out["last10_win_share"], 0.7)
        self.assertEqual(out["series_margin"], 2.0)
        self.assertAlmostEqual(out["won_of_last_share"], 8 / 12)
        self.assertEqual(nf.numbers("Detroit enters on a 4-game losing streak and trails the season series 0-1.")["streak"], -4.0)
        self.assertEqual(nf.numbers("trails the season series 0-1")["series_margin"], -1.0)
        self.assertTrue(math.isnan(nf.numbers("nothing to see")["streak"]))

    def test_prose_filter_drops_table_rows(self):
        text = "Jayson Tatum leads the team in scoring this season with big games.\n12 34 56 78 90 11 22 33\nPTS REB AST"
        self.assertEqual(nf.prose_sentences(text), ["Jayson Tatum leads the team in scoring this season with big games."])


class TextScoreTests(unittest.TestCase):
    def docs(self, flip_season=None):
        rng = random.Random(0)
        vocab = ["streak", "injury", "momentum", "defense", "rebound", "return", "slump", "victory", "travel", "rookie"]
        docs, won = [], {}
        for i in range(240):
            season = 2012 + i % 6
            split = "train" if season < 2017 else "dev"
            text = " ".join(rng.choice(vocab) for _ in range(40)) + (" victory streak" if (i // 6) % 2 else " slump injury")
            docs.append({"text": text, "split": split, "season": season})
            if split == "train":
                outcome = (i // 6) % 2
                won[i] = 1 - outcome if season == flip_season else outcome
        return docs, won

    def test_train_scores_do_not_depend_on_their_own_season(self):
        docs, won = self.docs()
        _, base = nf.text_models(docs, won)
        flipped_docs, flipped = self.docs(flip_season=2013)
        _, after = nf.text_models(flipped_docs, flipped)
        same = np.array([d["season"] == 2013 for d in docs])
        np.testing.assert_allclose(base[same], after[same])        # own season's labels never used
        self.assertFalse(np.allclose(base[~same & np.array([d["split"] == "train" for d in docs])],
                                     after[~same & np.array([d["split"] == "train" for d in docs])]))
        self.assertTrue(np.isfinite(base).all())

    def test_learns_a_real_signal(self):
        docs, won = self.docs()
        _, score = nf.text_models(docs, won)
        dev = np.array([d["split"] == "dev" for d in docs])
        positive = np.array(["victory streak" in d["text"] for d in docs])
        self.assertGreater(score[dev & positive].mean(), score[dev & ~positive].mean())


if __name__ == "__main__":
    unittest.main()
