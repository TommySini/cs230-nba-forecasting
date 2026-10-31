import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_text_ablation import (ARMS, CONFIG_PATH, FEATURE_NAMES, assign_splits,
                               make_model, paired_brier_interval, prompt_for,
                               select_shared_epoch, trainable_digest, verify_injury)


class TextAblationTests(unittest.TestCase):
    def test_chronology_keeps_complete_dates_and_disjoint_membership(self):
        rows = [{"game_id": str(i), "game_date_et": f"2026-01-0{1 + i // 2}",
                 "scheduled_start_utc": f"2026-01-0{1 + i // 2}T22:00:00Z"} for i in range(8)]
        result = assign_splits(rows[::-1], {"train": 4, "dev": 2, "test": 2})
        self.assertEqual([r["game_id"] for r in result], list(map(str, range(8))))
        self.assertEqual([r["split"] for r in result], ["train"] * 4 + ["dev"] * 2 + ["test"] * 2)
        with self.assertRaisesRegex(ValueError, "calendar date"):
            assign_splits(rows, {"train": 3, "dev": 3, "test": 2})

    def test_prompts_exclude_outcomes_dates_and_unapproved_content(self):
        row = {"features": {n: float(i) for i, n in enumerate(FEATURE_NAMES)},
               "pregame_text": "HOME Player Example Out Ankle injury", "game_id": "DO_NOT_INCLUDE_ID",
               "game_date_et": "DO_NOT_INCLUDE_DATE", "home_win": "DO_NOT_INCLUDE_TARGET",
               "narrative_text": "DO_NOT_INCLUDE_NARRATIVE", "injury_report": {"source": "DO_NOT_INCLUDE_SOURCE"}}
        stats, text = (prompt_for(row, arm) for arm in ARMS)
        self.assertEqual(text.replace(row["pregame_text"], "[withheld]"), stats)
        self.assertNotIn("DO_NOT_INCLUDE", text)
        self.assertNotIn("Player Example", stats)
        row["home_win"] = "CHANGED_TARGET"
        self.assertEqual(prompt_for(row, ARMS[1]), text)

    def test_post_cutoff_and_unapproved_text_rejected_before_source_use(self):
        numeric = {"game_date_et": "2026-01-01", "scheduled_start_utc": "2026-01-01T20:00:00Z",
                   "cutoff_utc": "2026-01-01T19:00:00Z", "home_abbr": "BOS", "away_abbr": "NYK"}
        row = dict(numeric, narrative_text=None, injury_report={"text": "HOME injury", "not_yet_submitted": False,
                                                                "report_time_utc": "2026-01-01T19:01:00Z"})
        with self.assertRaisesRegex(ValueError, "Post-cutoff"):
            verify_injury(row, numeric, {})
        row["narrative_text"] = "An unapproved preview"
        with self.assertRaisesRegex(ValueError, "Unapproved"):
            verify_injury(row, numeric, {})

    def test_common_epoch_uses_both_arms_and_not_one_models_best_epoch(self):
        rows = [{"arm": ARMS[0], "epoch": 1, "dev_brier": .10},
                {"arm": ARMS[1], "epoch": 1, "dev_brier": .40},
                {"arm": ARMS[0], "epoch": 2, "dev_brier": .20},
                {"arm": ARMS[1], "epoch": 2, "dev_brier": .20}]
        self.assertEqual(select_shared_epoch(rows)[0]["epoch"], 2)
        with self.assertRaises(ValueError):
            select_shared_epoch(rows[:-1])

    def test_bootstrap_is_paired_and_reports_direction_correctly(self):
        config = {"bootstrap_seed": 230, "bootstrap_replicates": 100}
        y = np.array([0, 1, 1, 0])
        stats = np.full(4, .5)
        dates = ["a", "a", "b", "c"]
        same = paired_brier_interval(y, stats, stats, dates, config)
        self.assertEqual(same["paired_date_bootstrap_ci95"], [0, 0])
        perfect = paired_brier_interval(y, stats, y.astype(float), dates, config)
        self.assertEqual(perfect["text_minus_stats"], -.25)
        self.assertEqual(perfect["paired_date_bootstrap_ci95"], [-.25, -.25])

    def test_real_lora_binary_head_initialization_training_and_reload(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForSequenceClassification
        config = json.loads(CONFIG_PATH.read_text())
        tiny = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                           num_attention_heads=2, num_key_value_heads=2, num_labels=1, pad_token_id=0)
        class Tokenizer:
            pad_token_id = 0
        def fresh(*args, **kwargs):
            return Qwen2ForSequenceClassification(tiny)
        inputs = {"input_ids": torch.tensor([[2, 3, 4], [2, 5, 0]]),
                  "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]])}
        with patch("transformers.AutoModelForSequenceClassification.from_pretrained", side_effect=fresh):
            model = make_model(config, Tokenizer())
            other = make_model(config, Tokenizer())
            self.assertEqual(trainable_digest(model), trainable_digest(other))
            original_digest = trainable_digest(model)
            self.assertTrue(all(not p.requires_grad for n, p in model.named_parameters() if "lora_" not in n and "score" not in n))
            logits = model(**inputs).logits[:, 0]
            np.testing.assert_array_equal(torch.sigmoid(logits).detach().numpy(), [.5, .5])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.]))
            loss.backward()
            torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1).step()
            self.assertNotEqual(trainable_digest(model), original_digest)
            model.eval()
            expected = model(**inputs).logits.detach()
            with tempfile.TemporaryDirectory() as directory:
                model.save_pretrained(directory)
                loaded = make_model(config, Tokenizer(), directory).eval()
                torch.testing.assert_close(loaded(**inputs).logits.detach(), expected)


if __name__ == "__main__":
    unittest.main()
