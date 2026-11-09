import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import collect_pilot
import collect_rich_stats
from collect_rich_stats import (ESPN_SUMMARY, NbaGameIds, check_counters, collect, digest, nba_box_url,
                                normalize_nba_page, normalize_summary, rebuild_from_record)


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


def team_stat(doc, side, name, value):
    stats = doc["boxscore"]["teams"][0 if side == "home" else 1]["statistics"]
    next(r for r in stats if r["name"] == name)["displayValue"] = value


def nba_page(nba_game_id, game, *, home_tricode=None, home_score=None):
    """Minimal NBA.com box-score page whose players mirror fixture()."""
    def team(side, made, team_id):
        players = [{"firstName": "Player", "familyName": f"{side}{n}", "comment": "",
                    "statistics": {"minutes": "48:00", "points": 2 * made, "fieldGoalsMade": made,
                                   "fieldGoalsAttempted": 5, "threePointersMade": 0, "threePointersAttempted": 1,
                                   "freeThrowsMade": 0, "freeThrowsAttempted": 0, "reboundsOffensive": 1,
                                   "reboundsDefensive": 1, "reboundsTotal": 2, "assists": 1, "steals": 0,
                                   "blocks": 0, "foulsPersonal": 2, "turnovers": 2}} for n in range(5)]
        totals = {k: sum(p["statistics"][k] for p in players) for k in players[0]["statistics"] if k != "minutes"}
        return {"teamId": team_id, "teamTricode": game[side + "_abbr"], "score": game[side + "_score"],
                "statistics": totals, "players": players}
    home, away = team("home", 2, 1), team("away", 3, 2)
    home["teamTricode"] = home_tricode or home["teamTricode"]
    home["score"] = home_score if home_score is not None else home["score"]
    charts = {side: {"teamId": t["teamId"], "statistics": {"turnovers": 10, "turnoversTotal": 11, "turnoversTeam": 1}}
              for side, t in (("homeTeam", home), ("awayTeam", away))}
    data = {"props": {"pageProps": {"game": {"gameId": nba_game_id, "gameStatus": 3, "period": 4,
            "gameTimeUTC": game["scheduled_start_utc"], "homeTeam": home, "awayTeam": away, "postgameCharts": charts}}}}
    return f'<html><script id="__NEXT_DATA__" type="application/json">{json.dumps(data)}</script></html>'.encode()


def empty_espn_box(doc):
    """ESPN summaries with no box: empty team stats, placeholder player rows."""
    for group in doc["boxscore"]["players"]:
        side = "home" if group is doc["boxscore"]["players"][0] else "away"
        for n, athlete in enumerate(group["statistics"][0]["athletes"]):
            athlete["athlete"]["displayName"] = f"Player {side}{n}"
            athlete["stats"] = ["--"] + ["0"] * 13
    for team in doc["boxscore"]["teams"]:
        team["statistics"] = []
    return doc


def cache_put(root, url, kind, suffix, body):
    path = root / "data/raw" / kind / (hashlib.sha256(url.encode()).hexdigest()[:24] + suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    meta = dict(url=url, retrieved_at_utc="2026-11-08T00:00:00Z", status=200, sha256=digest(body),
                bytes=len(body), raw_path=str(path.relative_to(root)))
    path.with_suffix(".meta.json").write_text(json.dumps(meta))
    return meta


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

    def test_legacy_team_rebounds_are_separated_when_players_confirm_components(self):
        game, doc, source = fixture()
        team_stat(doc, "home", "totalRebounds", "13")
        out = normalize_summary(game, doc, source)
        home, away = out["teams"]["home"], out["teams"]["away"]
        self.assertEqual((home["reb"], home["oreb"], home["dreb"], home["team_rebounds"]), (10, 5, 5, 3))
        self.assertIn("team_rebounds", home["rebounds_definition"])
        self.assertNotIn("team_rebounds", away)
        self.assertTrue(out["players_complete"])

    def test_legacy_team_rebounds_require_matching_player_rows(self):
        for kind in ("player_oreb", "total_below_components"):
            with self.subTest(kind=kind):
                game, doc, source = fixture()
                if kind == "player_oreb":
                    team_stat(doc, "home", "totalRebounds", "13")
                    team_stat(doc, "home", "offensiveRebounds", "6")
                    team_stat(doc, "home", "defensiveRebounds", "4")
                else:
                    team_stat(doc, "home", "totalRebounds", "9")
                with self.assertRaisesRegex(ValueError, "Rebound counts do not reconcile"):
                    normalize_summary(game, doc, source)

    def test_current_records_carry_only_the_schema_tag_change(self):
        out = normalize_summary(*fixture())
        self.assertEqual(out["schema_version"], "rich-stats-v2")
        self.assertFalse({"team_rebounds", "rebounds_definition"} & set(out["teams"]["home"]))

    def test_team_counters_below_player_rows_are_a_team_failure(self):
        for name, value in (("assists", "3"), ("steals", "0"), ("blocks", "0"), ("turnovers", "0"), ("fouls", "4")):
            with self.subTest(name=name):
                game, doc, source = fixture()
                rows = doc["boxscore"]["players"][0]["statistics"][0]["athletes"]
                if name in ("steals", "blocks"):
                    rows[0]["stats"][8 if name == "steals" else 9] = "2"
                team_stat(doc, "home", name, value)
                if name == "turnovers":
                    team_stat(doc, "home", "totalTurnovers", "1")
                with self.assertRaisesRegex(ValueError, "below the sum of its player rows"):
                    normalize_summary(game, doc, source)

    def test_team_counters_above_player_rows_stay_a_player_issue(self):
        game, doc, source = fixture()
        team_stat(doc, "home", "assists", "6")
        out = normalize_summary(game, doc, source)
        self.assertEqual(out["teams"]["home"]["ast"], 6)
        self.assertFalse(out["players_complete"])

    def test_nba_page_replaces_empty_espn_box_with_legacy_tricode(self):
        game, doc, source = fixture()
        game.update(home_abbr="NOP")
        doc["header"]["competitions"][0]["competitors"][0]["team"]["abbreviation"] = "NO"
        for part in ("players", "teams"):
            doc["boxscore"][part][0]["team"]["abbreviation"] = "NO"
        empty_espn_box(doc)
        page = nba_page("0021200007", game, home_tricode="NOH")
        nba_source = dict(source, url=nba_box_url("0021200007"))
        record = normalize_nba_page(game, page, nba_source, doc, source, "0021200007")
        self.assertEqual(record["provenance"]["provider"], "NBA")
        self.assertEqual(record["teams"]["home"]["team_abbr"], "NOP")
        self.assertEqual(record["teams"]["home"]["tov"], 11)
        self.assertEqual([p["player_id"] for p in record["teams"]["home"]["players"]], [str(n) for n in range(100, 105)])
        self.assertIn("fieldGoalsMade-fieldGoalsAttempted", record["provenance"]["rejected_original_reason"])
        with self.assertRaisesRegex(ValueError, "identity/status mismatch"):
            normalize_nba_page(game, page, nba_source, doc, source, "0021200008")
        valid_game, valid_doc, _ = fixture()
        with self.assertRaisesRegex(ValueError, "only for invalid ESPN boxes"):
            normalize_nba_page(valid_game, nba_page("0021200007", valid_game), nba_source, valid_doc, source, "0021200007")

    def test_nba_ids_are_found_from_verified_page_identities(self):
        game, _, _ = fixture()
        first = dict(game, game_id="espn:500", espn_id="500", season=2012,
                     scheduled_start_utc="2012-11-01T00:00:00Z", game_date_et="2012-10-31")
        second = dict(game, game_id="espn:510", espn_id="510", season=2012,
                      scheduled_start_utc="2012-11-03T00:00:00Z", game_date_et="2012-11-02")
        makeup = dict(game, game_id="espn:900000", espn_id="900000", season=2012,
                      scheduled_start_utc="2012-11-05T00:00:00Z", game_date_et="2012-11-04")
        pages = {nba_box_url("0021200002"): nba_page("0021200002", first),
                 nba_box_url("0021200012"): nba_page("0021200012", second),
                 nba_box_url("0021200030"): nba_page("0021200030", dict(second, home_score=21))}
        statuses = {nba_box_url("0021200004"): 503}

        def served(url, *args, **kwargs):
            if url in pages:
                return pages[url], {"url": url, "status": 200}
            return None, {"url": url, "status": statuses.get(url, 404)}

        with tempfile.TemporaryDirectory() as temp, mock.patch.object(collect_rich_stats, "fetch", served):
            ids = NbaGameIds([first, second, makeup], fetch_missing=True, delay=0, retries=0, root=temp)
            nba_id, body, _, probed = ids.find(second)
            # The chronological guess lands on another verified game, whose ESPN-ID
            # anchor then gives the right sequence.
            self.assertEqual((nba_id, probed), ("0021200012", ["0021200002", "0021200012"]))
            self.assertEqual(body, pages[nba_box_url("0021200012")])
            self.assertEqual(ids.find(first)[:1], ("0021200002",))
            # A far ESPN ID never yields out-of-season requests; one 503 does not block.
            missing = ids.find(makeup)
            self.assertIsNone(missing[1])
            self.assertEqual(len(missing[3]), collect_rich_stats.NBA_MAX_PROBES)
            self.assertTrue(all(0 < int(p[5:]) <= collect_rich_stats.NBA_MAX_SEQUENCE for p in missing[3]))
            self.assertIn("0021200004", missing[3])
            self.assertFalse(ids.blocked.is_set())
            offline = NbaGameIds([first, second], fetch_missing=False, delay=0, retries=0, root=temp)
            self.assertEqual(offline.find(second), (None, None, None, []))
        statuses.update({nba_box_url(f"00212{n:05d}"): 429 for n in range(1, 20)})
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(collect_rich_stats, "fetch", served):
            ids = NbaGameIds([first, second, makeup], fetch_missing=True, delay=0, retries=0, root=temp)
            self.assertEqual(len(ids.find(makeup)[3]), 1)
            self.assertTrue(ids.blocked.is_set())

    def test_ambiguous_espn_names_leave_players_unmapped_not_guessed(self):
        game, doc, source = fixture()
        empty_espn_box(doc)
        twin = copy.deepcopy(doc["boxscore"]["players"][0]["statistics"][0]["athletes"][0])
        twin["athlete"]["id"] = "999"
        doc["boxscore"]["players"][0]["statistics"][0]["athletes"].append(twin)
        nba_source = dict(source, url=nba_box_url("0021200007"))
        record = normalize_nba_page(game, nba_page("0021200007", game), nba_source, doc, source, "0021200007")
        home = record["teams"]["home"]
        self.assertTrue(record["complete"])
        self.assertFalse(home["players_complete"])
        self.assertEqual([i["reason"] for i in home["player_issues"]][:1], ["missing_participated_player_id"])
        self.assertNotIn("999", [p["player_id"] for p in home["players"]])

    def test_collect_combines_histories_and_repairs_from_cache_only(self):
        game, doc, _ = fixture()
        game.update(season=2012)
        valid = dict(game, game_id="espn:124", espn_id="124", scheduled_start_utc="2023-11-03T00:00:00Z",
                     game_date_et="2023-11-02")
        valid_doc = copy.deepcopy(doc)
        valid_doc["header"].update(id="124")
        valid_doc["header"]["competitions"][0]["date"] = valid["scheduled_start_utc"]
        mismatch = dict(game, game_id="espn:125", espn_id="125", scheduled_start_utc="2023-11-04T00:00:00Z",
                        game_date_et="2023-11-03")
        mismatch_doc = empty_espn_box(copy.deepcopy(valid_doc))
        mismatch_doc["header"].update(id="125")
        mismatch_doc["header"]["competitions"][0]["date"] = mismatch["scheduled_start_utc"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for item, body in ((game, empty_espn_box(copy.deepcopy(doc))), (valid, valid_doc), (mismatch, mismatch_doc)):
                cache_put(root, ESPN_SUMMARY + item["espn_id"], "summaries", ".json", json.dumps(body).encode())
            cache_put(root, nba_box_url("0021200007"), "rich_stats_nba", ".html", nba_page("0021200007", game))
            cache_put(root, nba_box_url("0021200009"), "rich_stats_nba", ".html",
                      nba_page("0021200009", mismatch, home_score=21))
            histories = []
            for name, rows in (("a.jsonl", [game, mismatch]), ("b.jsonl", [valid])):
                histories.append(root / name)
                histories[-1].write_text("".join(json.dumps(r) + "\n" for r in rows))
            with mock.patch.object(collect_pilot, "ROOT", root), mock.patch.object(collect_rich_stats, "ROOT", root):
                records, failures, manifest = collect(histories, root / "out", workers=1)
                by_id = {r["game_id"]: r for r in records}
                self.assertEqual(by_id["espn:123"]["provenance"]["source_game_id"], "0021200007")
                self.assertEqual(by_id["espn:124"]["provenance"]["provider"], "ESPN")
                self.assertEqual([(f["game_id"], f["reason"]) for f in failures],
                                 [("espn:125", "espn_invalid_no_verified_nba_page")])
                self.assertEqual([h["path"] for h in manifest["histories"]], ["a.jsonl", "b.jsonl"])
                self.assertEqual(manifest["season_coverage"]["2012"]["nba_repaired"], 1)
                self.assertEqual(manifest["nba_page_probes"], 0)
                self.assertEqual(rebuild_from_record(game, by_id["espn:123"], root), by_id["espn:123"])
                with self.assertRaisesRegex(ValueError, "Duplicate history game"):
                    collect(histories + histories[:1], root / "dup", workers=1)


if __name__ == "__main__":
    unittest.main()
