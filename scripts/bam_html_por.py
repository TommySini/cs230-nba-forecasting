"""Publisher-specific Portland opening checks; canonical article text unchanged."""
from datetime import datetime
import re
from zoneinfo import ZoneInfo

from bam_html_notes import MONTHS, TEAM_NAMES, parse_html_note, _stamp


def parse_por_html_note(raw, provenance, game, prior_records):
    result = parse_html_note(raw, provenance, game)
    reasons = set(result["eligibility"]["reasons"])
    errors, warnings, checks = [], [], {}
    if provenance.get("publisher") != "POR" or not provenance["publisher_url"].startswith("https://www.nba.com/blazers/news/game-notes-"):
        errors.append("unsupported_portland_publisher_url")
    if "POR" not in {game["home_abbr"], game["away_abbr"]}:
        errors.append("portland_not_in_matchup")
        other = None
    else:
        other = game["away_abbr"] if game["home_abbr"] == "POR" else game["home_abbr"]
    opening = result["sections"][1]["text"] if len(result["sections"]) > 1 and result["sections"][1]["label"] in {"paragraph", "heading"} else ""
    first = opening.splitlines()[0] if opening else ""
    opponent = TEAM_NAMES.get(other, "NO_VALID_OPPONENT")
    headline = re.fullmatch(r"\s*Portland Trail Blazers\s*\((\d+)-(\d+),\s*\d+-\d+\s+(home|away)\){1,2}\s*"
                            r"(@|at|vs\.?)\s*[^()\n]*\b" + re.escape(opponent) +
                            r"\s*\((\d+)-(\d+),\s*\d+-\d+\s+(home|away)\)\s*", first, re.I)
    if not headline:
        errors.append("missing_literal_portland_matchup_and_records")
    else:
        if re.search(r"\b(?:home|away)\)\)", first, re.I):
            warnings.append("extra_closing_parenthesis_in_literal_header")
        home = headline[4].lower().startswith("vs")
        if home != (game["home_abbr"] == "POR") or headline[3].lower() != ("home" if home else "away") or headline[7].lower() != ("away" if home else "home"):
            errors.append("portland_header_home_away_disagrees")
        for team, actual in (("POR", (int(headline[1]), int(headline[2]))), (other, (int(headline[5]), int(headline[6])))):
            expected = prior_records.get(team)
            checks[team] = {"actual": list(actual), "expected": expected}
            if expected is None or list(actual) != list(expected):
                errors.append("portland_header_record_disagrees_" + str(team))
    dates = list(re.finditer(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s*([A-Za-z]+)\.?\s+(\d{1,2}),\s*(20\d{2})\b", opening, re.I))
    if len(dates) != 1:
        errors.append("missing_or_ambiguous_full_portland_opening_date")
    else:
        value = dates[0]
        try:
            stated = datetime(int(value[4]), MONTHS[value[2].lower()], int(value[3]))
            if stated.date().isoformat() != game["game_date_et"] or stated.strftime("%A").lower() != value[1].lower():
                errors.append("portland_opening_date_disagrees")
        except (KeyError, ValueError):
            errors.append("invalid_portland_opening_date")
        checks["literal_date"] = value[0]
    times = list(re.finditer(r"\b(\d{1,2}):(\d{2})\s*(am|pm)\s*(PST|PDT|PT)\b", opening, re.I))
    if len(times) != 1:
        errors.append("missing_or_ambiguous_portland_opening_tipoff")
    else:
        value = times[0]
        scheduled = _stamp(game["scheduled_start_utc"]).astimezone(ZoneInfo("America/Los_Angeles"))
        hour, minute = int(value[1]), int(value[2])
        if not 1 <= hour <= 12 or not 0 <= minute <= 59 or ((hour % 12) + (12 if value[3].lower() == "pm" else 0), minute) != (scheduled.hour, scheduled.minute):
            errors.append("portland_opening_wallclock_disagrees")
        if value[4].upper() not in ("PT", scheduled.tzname()):
            warnings.append("literal_" + value[4].upper() + "_differs_from_verified_schedule_" + scheduled.tzname())
        checks["literal_tipoff"] = value[0]
        checks["schedule_pacific_wallclock"] = scheduled.isoformat()
    if not errors:
        reasons.discard("game_date_missing_from_title")
        if all(re.search(r"\b" + re.escape(TEAM_NAMES[abbr]) + r"\b", result["title"], re.I) for abbr in (game["home_abbr"], game["away_abbr"])):
            reasons.discard("home_team_missing_from_title_or_opening")
            reasons.discard("away_team_missing_from_title_or_opening")
    reasons.update(errors)
    result["eligibility"].update(status="quarantined" if reasons else "verified_pregame", reasons=sorted(reasons))
    contexts, marker, recap = {}, None, None
    for section in result["sections"]:
        value = section["text"].strip()
        if value in {"SERIES HISTORY", "SERIES NOTES", "INJURY REPORT", "LAST FIVE GAMES", "BROADCAST NOTES"}:
            marker, recap = section["section"], None
        dated_recap = bool(re.search(r"\b20\d{2}\b.*\bGAME\s*#\d+", value) and re.search(r"\b(?:HOME|AWAY)\)", value))
        ids = [1, 2] + ([marker] if marker else []) + ([recap] if recap and not dated_recap else [])
        contexts[str(section["section"])] = list(dict.fromkeys(i for i in ids if i != section["section"]))
        if dated_recap:
            recap = section["section"]
    result["publisher_adapter"] = {"version": "portland_literal_opening_v1", "checks": checks,
                                   "warnings": warnings, "context_sections": contexts,
                                   "cutoff_source": "Verified scheduled UTC tipoff minus 60 minutes; never the article timezone label."}
    return result
