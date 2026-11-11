"""Stints: starters inferred from play-by-play, substitutions applied, margins add up to the score."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stint_features as st

HOME = {f"h{i}": 40.0 for i in range(5)} | {"h5": 8.0}
AWAY = {f"a{i}": 48.0 for i in range(5)}


def event(period, seconds_left, order, home, away, a1=None, a2=None, sub=False):
    return {"period": period, "seconds_left": seconds_left, "order": order, "home_score": home, "away_score": away,
            "sub": sub, "a1": a1, "a2": a2, "a3": None}


def game():
    rows = [event(1, 700, 1, 2, 0, "h0"), event(1, 650, 2, 2, 3, "a0"), event(1, 600, 3, 4, 3, "h1"),
            event(1, 480, 4, 4, 3, "h5", "h4", sub=True),          # h5 in for h4, who never touched the ball
            event(1, 300, 5, 7, 3, "h5"), event(1, 100, 6, 7, 5, "a1")]
    for period in (2, 3, 4):
        rows.append(event(period, 400, 10 * period, 7 + period, 5, "h0"))
    return rows


class StintTests(unittest.TestCase):
    def test_stints_cover_the_game_and_add_up_to_the_final_margin(self):
        found = st.stints(game(), HOME, AWAY)
        self.assertEqual(sum(s[0] for s in found), 4 * 720)
        self.assertEqual(sum(s[1] for s in found), 11 - 5)
        self.assertTrue(all(len(s[2]) == 5 and len(s[3]) == 5 for s in found))

    def test_silent_starter_is_placed_and_substitution_changes_the_five(self):
        first, second = st.stints(game(), HOME, AWAY)[:2]
        self.assertIn("h4", first[2])                              # inferred from being substituted out
        self.assertNotIn("h5", first[2])
        self.assertEqual((first[0], first[1]), (240.0, 1))         # 720 -> 480 seconds left, 4-3
        self.assertIn("h5", second[2])
        self.assertNotIn("h4", second[2])

    def test_plus_minus_follows_who_was_on_court(self):
        pm = st.player_plus_minus(st.stints(game(), HOME, AWAY))
        self.assertEqual(pm["h0"], 6)
        self.assertEqual(pm["h4"], 5)                               # first stint, then periods 2-4 as a top-minutes starter
        self.assertEqual(pm["h5"], 1)
        self.assertEqual(pm["a0"], -6)

    def test_unplaceable_lineup_is_rejected(self):
        self.assertIsNone(st.stints(game(), {"h0": 40.0, "h1": 40.0}, AWAY))


if __name__ == "__main__":
    unittest.main()
