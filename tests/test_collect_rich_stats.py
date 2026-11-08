import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from collect_rich_stats import check_counters, digest, normalize_summary, rebuild_from_record


def fixture():
    game = dict(game_id="espn:123", espn_id="123", season=2023,
                scheduled_start_utc="2023-11-02T00:00:00Z", game_date_et="2023-11-01",
                home_abbr="BOS", away_abbr="CLE", home_score=20, away_score=30)
    provenance = dict(status=200, sha256="a" * 64, url="https://example.test/123",
                      raw_path="raw/123.json", retrieved_at_utc="2026-11-03T00:00:00Z")
    teams, groups, competitors = [], [], []
    for index, side in enumerate(("home", "away")):
        abbr = game[side + "_abbr"]
        made = index + 2
        players = []
        for number in range(5):
            players.append({"athlete": {"id": str(100 * (index + 1) + number),
                                         "active": False, "currentTeam": "POISON"},
                            "starter": True, "didNotPlay": False,
                            "stats": ["48", str(made * 2), f"{made}-5", "0-1", "0-0", "2",
                                      "1", "2", "0", "0", "1", "1", "2", "0"]})
        groups.append({"team": {"abbreviation": abbr}, "statistics": [{
            "labels": ["MIN", "PTS", "FG", "3PT", "FT", "REB", "AST", "TO", "STL", "BLK", "OREB", "DREB", "PF", "+/-"],
            "athletes": players}]})
        values = {"fieldGoalsMade-fieldGoalsAttempted": f"{made * 5}-25",
                  "threePointFieldGoalsMade-threePointFieldGoalsAttempted": "0-5",
                  "freeThrowsMade-freeThrowsAttempted": "0-0",
                  "offensiveRebounds": "5", "defensiveRebounds": "5", "totalRebounds": "10",
                  "assists": "5", "steals": "0", "blocks": "0", "fouls": "10",
                  "turnovers": "10", "teamTurnovers": "1", "totalTurnovers": "11"}
        teams.append({"team": {"abbreviation": abbr}, "statistics": [
            {"name": k, "displayValue": v} for k, v in values.items()]})
        competitors.append({"homeAway": side, "team": {"abbreviation": abbr}, "score": str(game[side + "_score"])})
    document = {"header": {"id": "123", "competitions": [{"date": game["scheduled_start_utc"],
                  "status": {"type": {"completed": True}, "period": 4}, "competitors": competitors}]},
                "boxscore": {"teams": teams, "players": groups},
                "news": "POISON", "injuries": "POISON", "odds": "POISON"}
    return game, document, provenance


class CollectorTests(unittest.TestCase):
    def test_numeric_whitelist_and_team_turnovers(self):
        out = normalize_summary(*fixture())
        self.assertTrue(out["complete"])
        self.assertTrue(out["players_complete"])
        self.assertEqual(out["game_minutes"], 48)
        self.assertEqual(out["teams"]["home"]["minutes"], 240)
        self.assertEqual(out["teams"]["home"]["tov"], 11)
        self.assertEqual(sum(p["tov"] for p in out["teams"]["home"]["players"]), 10)
        self.assertNotIn("POISON", json.dumps(out))

    def test_all_star_mini_game_not_treated_as_nba_franchise_history(self):
        game, doc, source = fixture()
        game.update(home_abbr="STARS", away_abbr="WORLD")
        with self.assertRaisesRegex(ValueError, "two NBA franchises"):
            normalize_summary(game, doc, source)

    def test_game_identity_date_scores_and_completed_required(self):
        for mutation in ("identity", "date", "score", "complete", "team"):
            with self.subTest(mutation=mutation):
                game, doc, source = fixture()
                comp = doc["header"]["competitions"][0]
                if mutation == "identity": doc["header"]["id"] = "999"
                if mutation == "date": comp["date"] = "2023-11-03T00:00:00Z"
                if mutation == "score": comp["competitors"][0]["score"] = "99"
                if mutation == "complete": comp["status"]["type"]["completed"] = False
                if mutation == "team": comp["competitors"][0]["team"]["abbreviation"] = "LAL"
                with self.assertRaises(ValueError): normalize_summary(game, doc, source)

    def test_missing_minutes_remain_unknown(self):
        game, doc, source = fixture()
        bench = {"athlete": {"id": "999"}, "starter": False, "didNotPlay": False,
                 "stats": ["--", "0", "0-0", "0-0", "0-0", "0", "0", "0", "0", "0", "0", "0", "0", "0"]}
        doc["boxscore"]["players"][0]["statistics"][0]["athletes"].append(bench)
        out = normalize_summary(game, doc, source)
        self.assertTrue(out["complete"])
        self.assertFalse(out["players_complete"])
        self.assertEqual(len(out["teams"]["home"]["players"]), 5)
        self.assertEqual(out["teams"]["home"]["player_issues"][0]["player_id"], "999")

    def test_explicit_dnp_not_imputed_as_participation(self):
        game, doc, source = fixture()
        doc["boxscore"]["players"][0]["statistics"][0]["athletes"].append(
            {"athlete": {"id": "999"}, "didNotPlay": True, "stats": []})
        out = normalize_summary(game, doc, source)
        self.assertTrue(out["players_complete"])
        self.assertEqual(len(out["teams"]["home"]["players"]), 5)

    def test_shooting_arithmetic_rejects_corruption(self):
        game, doc, source = fixture()
        doc["boxscore"]["teams"][0]["statistics"][0]["displayValue"] = "11-25"
        with self.assertRaisesRegex(ValueError, "Points do not reconcile"):
            normalize_summary(game, doc, source)

    def test_null_player_identity_is_not_accepted(self):
        game, doc, source = fixture()
        doc["boxscore"]["players"][0]["statistics"][0]["athletes"][0]["athlete"]["id"] = None
        out = normalize_summary(game, doc, source)
        self.assertTrue(out["complete"])
        self.assertFalse(out["players_complete"])
        self.assertEqual(out["teams"]["home"]["minutes"], 240)
        self.assertEqual(out["teams"]["home"]["observed_player_minutes"], 192)
        self.assertNotIn("None", [p["player_id"] for p in out["teams"]["home"]["players"]])

    def test_turnover_total_must_reconcile(self):
        game, doc, source = fixture()
        stats = doc["boxscore"]["teams"][0]["statistics"]
        next(r for r in stats if r["name"] == "totalTurnovers")["displayValue"] = "99"
        with self.assertRaisesRegex(ValueError, "Total turnovers do not reconcile"):
            normalize_summary(game, doc, source)

    def test_impossible_two_point_split_is_rejected(self):
        row = normalize_summary(*fixture())["teams"]["home"]
        row.update(fga=10, fg3a=9)
        with self.assertRaisesRegex(ValueError, "Two-point makes exceed"):
            check_counters(row)

    def test_rebuild_checks_raw_provenance_and_bytes(self):
        game, doc, source = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "data/raw/summaries/example.json"
            path.parent.mkdir(parents=True)
            body = json.dumps(doc).encode()
            path.write_bytes(body)
            source.update(raw_path="data/raw/summaries/example.json", sha256=digest(body))
            path.with_suffix(".meta.json").write_text(json.dumps(source))
            record = normalize_summary(game, doc, source)
            self.assertEqual(rebuild_from_record(game, record, root), record)
            path.write_bytes(body + b" ")
            with self.assertRaisesRegex(ValueError, "checksum/status mismatch"):
                rebuild_from_record(game, record, root)

    def test_overtime_duration_uses_period(self):
        game, doc, source = fixture()
        doc["header"]["competitions"][0]["status"]["period"] = 5
        for group in doc["boxscore"]["players"]:
            for athlete in group["statistics"][0]["athletes"]:
                athlete["stats"][0] = "53"
        out = normalize_summary(game, doc, source)
        self.assertEqual(out["game_minutes"], 53)
        self.assertTrue(out["players_complete"])

    def test_quarter_scores_supply_duration_without_status_period(self):
        game, doc, source = fixture()
        comp = doc["header"]["competitions"][0]
        del comp["status"]["period"]
        for team, values in zip(comp["competitors"], ([5, 5, 5, 5], [6, 8, 8, 8])):
            team["linescores"] = [{"displayValue": str(v)} for v in values]
        # Missing participant identity must not erase trustworthy team minutes.
        doc["boxscore"]["players"][0]["statistics"][0]["athletes"][0]["athlete"]["id"] = None
        out = normalize_summary(game, doc, source)
        self.assertEqual(out["game_minutes"], 48)
        self.assertEqual(out["teams"]["home"]["minutes"], 240)
        self.assertFalse(out["players_complete"])
        self.assertIn("quarter linescores", out["duration_basis"])

    def test_overtime_quarter_scores_supply_duration(self):
        game, doc, source = fixture()
        comp = doc["header"]["competitions"][0]
        del comp["status"]["period"]
        for team, values in zip(comp["competitors"], ([2, 3, 4, 5, 6], [5, 6, 7, 8, 4])):
            team["linescores"] = [{"period": i, "value": v, "displayValue": str(v)} for i, v in enumerate(values, 1)]
        out = normalize_summary(game, doc, source)
        self.assertEqual(out["game_minutes"], 53)
        self.assertEqual(out["teams"]["away"]["minutes"], 265)

    def test_inconsistent_quarter_scores_rejected(self):
        for kind in ("sum", "count", "index", "status", "values", "one_side"):
            with self.subTest(kind=kind):
                game, doc, source = fixture()
                comp = doc["header"]["competitions"][0]
                for team, values in zip(comp["competitors"], ([5, 5, 5, 5], [6, 8, 8, 8])):
                    team["linescores"] = [{"period": i, "value": v, "displayValue": str(v)} for i, v in enumerate(values, 1)]
                lines = comp["competitors"][0]["linescores"]
                if kind == "sum": lines[0].update(value=6, displayValue="6")
                if kind == "count": lines.append({"displayValue": "0"})
                if kind == "index": lines[1]["period"] = 1
                if kind == "status": comp["status"]["period"] = 5
                if kind == "values": lines[0]["value"] = 6
                if kind == "one_side": del comp["competitors"][0]["linescores"]
                with self.assertRaises(ValueError): normalize_summary(game, doc, source)


if __name__ == "__main__":
    unittest.main()
