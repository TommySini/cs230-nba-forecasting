"""Situational features use earlier ET dates only, and the injury parser reads team segments."""
from datetime import date, timedelta
import copy
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import situational_features as sf

TEAMS = ["BOS", "DEN", "LAL", "MIA"]


def synthetic(days=12, seed=3):
    rng = random.Random(seed)
    games, team_boxes, player_boxes = [], {}, {}
    start = date(2023, 11, 1)
    for d in range(days):
        order = TEAMS[:]
        rng.shuffle(order)
        for k in range(0, 4, 2):
            gid = f"g{d}-{k}"
            home, away = order[k], order[k + 1]
            hs, as_ = rng.randint(90, 130), rng.randint(90, 130)
            games.append({"game_id": gid, "season": 2023, "game_date_et": (start + timedelta(days=d)).isoformat(),
                          "home_abbr": home, "away_abbr": away, "home_score": hs, "away_score": as_ + (as_ == hs)})
            for side, team, pts in (("home", home, hs), ("away", away, as_)):
                team_boxes[gid, side] = {"complete": True, "pts": pts, "fga": 85, "fgm": 40, "fg3m": rng.randint(8, 18),
                                         "fg3a": 35, "fta": 20, "oreb": 10, "tov": 13}
                player_boxes[gid, side] = [{"player_id": f"{team}{i}", "minutes": rng.choice([0, 12, 24, 36]),
                                            "pts": rng.randint(0, 30), "fgm": 5, "fga": 10, "ftm": 2, "fta": 3, "oreb": 1,
                                            "dreb": 3, "stl": 1, "ast": 2, "blk": 0, "pf": 2, "tov": 1} for i in range(10)]
    return games, team_boxes, player_boxes


def same(a, b):
    return all((math.isnan(a[k]) and math.isnan(b[k])) or a[k] == b[k] for k in a)


class LeakageTests(unittest.TestCase):
    def test_changing_a_date_leaves_that_date_and_earlier_unchanged(self):
        games, teams, players = synthetic()
        before = sf.build(games, teams, players)
        cut = "2023-11-07"
        g2, t2, p2 = copy.deepcopy(games), copy.deepcopy(teams), copy.deepcopy(players)
        for g in g2:
            if g["game_date_et"] >= cut:
                g["home_score"], g["away_score"] = 200, 50
                for side in ("home", "away"):
                    t2[g["game_id"], side]["pts"] = 200
                    for p in p2[g["game_id"], side]:
                        p["minutes"], p["pts"] = 48, 99
        after = sf.build(g2, t2, p2)
        for g in games:
            if g["game_date_et"] <= cut:
                self.assertTrue(same(before[g["game_id"]], after[g["game_id"]]), g["game_id"])
        later = [g["game_id"] for g in games if g["game_date_et"] > cut]
        self.assertTrue(any(not same(before[i], after[i]) for i in later))

    def test_first_game_has_no_history_and_columns_are_complete(self):
        games, teams, players = synthetic(days=2)
        features = sf.build(games, teams, players)
        first = features[games[0]["game_id"]]
        self.assertEqual(set(first), set(sf.FEATURE_NAMES))
        self.assertTrue(math.isnan(first["home_ewm_net"]))
        self.assertEqual(first["home_miles_since_last"], 0.0)

    def test_travel_altitude_and_rematch(self):
        games = [
            {"game_id": "a", "season": 2023, "game_date_et": "2023-11-01", "home_abbr": "BOS", "away_abbr": "DEN", "home_score": 110, "away_score": 100},
            {"game_id": "b", "season": 2023, "game_date_et": "2023-11-02", "home_abbr": "DEN", "away_abbr": "BOS", "home_score": 100, "away_score": 90},
        ]
        f = sf.build(games, {}, {})["b"]
        self.assertEqual(f["away_rematch"], 1.0)
        self.assertEqual(f["away_rematch_prev_margin"], 10.0)      # BOS won game one by 10
        self.assertEqual(f["away_altitude_short_rest"], 1.0)
        self.assertEqual(f["home_altitude_short_rest"], 0.0)
        self.assertAlmostEqual(f["away_miles_since_last"], 1750, delta=60)
        self.assertEqual(f["away_tz_shift"], -2.0)


class InjuryParseTests(unittest.TestCase):
    def test_counts_split_by_team_and_skip_g_league(self):
        text = ("Boston Celtics Tatum, Jayson Out Injury/Illness - Ankle Brown, Jaylen Questionable Rest "
                "Smith, Joe Out G League - Two-Way Denver Nuggets Jokic, Nikola Doubtful Injury/Illness - Wrist")
        row = {"home_team": "Boston Celtics", "away_team": "Denver Nuggets", "injury_report": {"text": text}}
        counts = sf.injury_counts(row)
        self.assertEqual(counts["home"], {"Out": 1, "Doubtful": 0, "Questionable": 1})
        self.assertEqual(counts["away"], {"Out": 0, "Doubtful": 1, "Questionable": 0})
        self.assertIsNone(sf.injury_counts({"injury_report": None}))

    def test_note_counts_read_only_the_status_section(self):
        text = ("Timeout out of bounds. INJURY REPORT Tatum (ankle) is OUT. Brown (knee) is questionable; "
                "Smith (G League - Two-Way) out\nJones probable")
        self.assertEqual(sf.note_counts(text), {"out": 1, "questionable": 1, "probable": 1})
        self.assertEqual(sf.note_counts("No section here: out, questionable."), {"out": 0, "questionable": 0, "probable": 0})

    def test_notes_are_credited_to_the_publishing_side(self):
        games = [{"game_id": "g", "season": 2023, "game_date_et": "2023-11-01", "home_abbr": "BOS", "away_abbr": "DEN",
                  "home_score": 100, "away_score": 90}]
        notes = {"g": {"away": ["STATUS REPORT Murray is out. Gordon is out."]}}
        f = sf.build(games, {}, {}, notes=notes)["g"]
        self.assertEqual((f["away_note_found"], f["away_note_out"]), (1.0, 2.0))
        self.assertEqual(f["home_note_found"], 0.0)
        self.assertTrue(math.isnan(f["home_note_out"]))


if __name__ == "__main__":
    unittest.main()
