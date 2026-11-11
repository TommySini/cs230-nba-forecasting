"""Rating features: earlier dates only, injury-report names matched to the roster, and lineups adjusted."""
from datetime import date, timedelta
import copy
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rating_features as rf

TEAMS = ["BOS", "DEN", "LAL", "MIA"]


def synthetic(days=60, seed=5):
    rng = random.Random(seed)
    strength = {t: rng.gauss(0, 5) for t in TEAMS}
    games, minutes = [], {}
    for d in range(days):
        order = TEAMS[:]
        rng.shuffle(order)
        for k in (0, 2):
            gid = f"g{d}-{k}"
            home, away = order[k], order[k + 1]
            margin = round(strength[home] - strength[away] + 3 + rng.gauss(0, 10)) or 1
            games.append({"game_id": gid, "season": 2023, "game_date_et": (date(2023, 11, 1) + timedelta(days=d)).isoformat(),
                          "home_abbr": home, "away_abbr": away, "home_score": 110 + max(margin, 0), "away_score": 110 + max(-margin, 0)})
            for side, team in (("home", home), ("away", away)):
                minutes[gid, side] = {f"{team}{i}": 48.0 for i in range(5)}
    return games, minutes


def same(a, b):
    return all((isinstance(a[k], float) and math.isnan(a[k]) and math.isnan(b[k])) or a[k] == b[k] for k in a)


class RatingTests(unittest.TestCase):
    def test_later_games_never_change_earlier_features(self):
        games, minutes = synthetic()
        before = rf.build(games, minutes)
        cut = "2023-12-10"
        changed = copy.deepcopy(games)
        for g in changed:
            if g["game_date_et"] >= cut:
                g["home_score"], g["away_score"] = 160, 80
        after = rf.build(changed, minutes)
        for g in games:
            if g["game_date_et"] <= cut:
                self.assertTrue(same(before[g["game_id"]], after[g["game_id"]]), g["game_id"])
        self.assertTrue(any(not same(before[g["game_id"]], after[g["game_id"]]) for g in games if g["game_date_et"] > "2023-12-24"))

    def test_ratings_learn_strength(self):
        games, minutes = synthetic(days=120)
        f = rf.build(games, minutes)
        late = [g for g in games if g["game_date_et"] > "2024-02-01"]
        margin = [g["home_score"] - g["away_score"] for g in late]
        for key in ("apm_margin_full", "diff_elo"):
            pred = [f[g["game_id"]][key] for g in late]
            self.assertGreater(sum(p * m for p, m in zip(pred, margin)), 0, key)


class InjuryMatchTests(unittest.TestCase):
    names = {"1": {"first": "Jayson", "last": "Tatum"}, "2": {"first": "Jaylen", "last": "Brown"},
             "3": {"first": "Jaren", "last": "Jackson Jr."}, "4": {"first": "Nikola", "last": "Jokić"}}

    def test_rows_and_matching_tolerate_glued_reason_words_suffixes_and_accents(self):
        segment = ("Boston Celtics Tatum, Jayson Out Injury/Illness - Left Ankle; Sprain Brown, Jaylen Questionable Rest "
                   "Jackson Jr., Jaren Doubtful Injury/Illness - Knee Jokic, Nikola Probable Injury/Illness - Wrist")
        status = rf.match(rf.report_rows(segment), ["1", "2", "3", "4"], self.names)
        self.assertEqual(status, {"1": "Out", "2": "Questionable", "3": "Doubtful", "4": "Probable"})

    def test_star_out_lowers_injury_adjusted_strength(self):
        recent = [(date(2024, 1, d), {"1": 36.0, "2": 34.0, "5": 20.0, "6": 20.0, "7": 20.0}) for d in range(1, 6)]
        ratings = {"1": 8.0, "2": 2.0, "5": -1.0, "6": -1.0, "7": -1.0}
        segments = {"home": "Boston Celtics Tatum, Jayson Out Injury/Illness - Ankle"}
        out = rf._team_strength("home", recent, ratings, segments, self.names)
        self.assertLess(out["home_apm_injury"], out["home_apm_full"])
        self.assertEqual(out["home_injury_matched"], 1.0)
        self.assertGreater(out["home_injury_value_out"], 0)


if __name__ == "__main__":
    unittest.main()
