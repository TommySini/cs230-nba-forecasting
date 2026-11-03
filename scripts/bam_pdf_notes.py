"""Narrow, source-preserving header checks for readable Warriors packets."""
from datetime import datetime
import re

from bam_html_notes import TEAM_NAMES
from review_game_notes import prior_record


def warriors_header_checks(text, game, history):
    """Require the literal reviewed headline, full date and game-number line.

    This checks original extracted text; it never rewrites team names or creates
    source lines. The caller retains every other original content-check failure.
    """
    checks, reasons = {}, []
    if "GSW" not in {game["home_abbr"], game["away_abbr"]}:
        return ["warriors_not_in_matchup"], checks
    other = game["away_abbr"] if game["home_abbr"] == "GSW" else game["home_abbr"]
    nickname = TEAM_NAMES[other]
    headlines = list(re.finditer(r"(?mi)^\s*WARRIORS\s*\(\s*(\d+)\s*-\s*(\d+)\s*\)\s*"
                                r"(VS\.?|AT|@)\s*" + re.escape(nickname) +
                                r"\s*\(\s*(\d+)\s*-\s*(\d+)\s*\)\s*$", text))
    if len(headlines) != 1:
        return ["missing_or_ambiguous_literal_warriors_matchup_line"], checks
    headline = headlines[0]
    home = headline[3].upper().startswith("VS")
    if home != (game["home_abbr"] == "GSW"):
        reasons.append("warriors_headline_home_away_disagrees")
    for team, record in (("GSW", (int(headline[1]), int(headline[2]))),
                         (other, (int(headline[4]), int(headline[5])))):
        expected = prior_record(game, team, history)
        side = "home" if game["home_abbr"] == team else "away"
        checks[side + "_record"] = {"actual": record, "expected": expected}
        if record != expected:
            reasons.append(side + "_record_disagrees_with_prior_history")
    dates = list(re.finditer(r"(?mi)^\s*(MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY),?\s+"
                            r"([A-Z]+)\.?\s+(\d{1,2}),?\s+(20\d{2})\s*$", text))
    if len(dates) != 1:
        reasons.append("missing_or_ambiguous_full_warriors_header_date")
    else:
        date = dates[0]
        try:
            stated = datetime.strptime(date[2].title() + " " + date[3] + " " + date[4], "%B %d %Y")
            if stated.date().isoformat() != game["game_date_et"] or stated.strftime("%A").upper() != date[1].upper():
                reasons.append("warriors_full_header_date_disagrees")
        except ValueError:
            reasons.append("invalid_warriors_header_date")
        checks["literal_header_date"] = date[0].strip()
    numbers = list(re.finditer(r"(?mi)^\s*GAME\s*#\s*(\d+)\s*[/|]\s*(HOME|ROAD|AWAY)\s+GAME\s*#\s*\d+\s*$", text))
    if len(numbers) != 1:
        reasons.append("missing_or_ambiguous_warriors_game_number")
    else:
        expected = sum(prior_record(game, "GSW", history)) + 1
        checks["publisher_game_number"] = {"actual": int(numbers[0][1]), "expected": expected}
        if int(numbers[0][1]) != expected:
            reasons.append("publisher_game_number_disagrees_with_history")
        if (numbers[0][2].upper() == "HOME") != (game["home_abbr"] == "GSW"):
            reasons.append("warriors_game_number_home_away_disagrees")
    if f"{game['season']}-{(game['season'] + 1) % 100:02}" not in text:
        reasons.append("season_mismatch_or_missing")
    checks["literal_matchup_line"] = headline[0].strip()
    checks["validator"] = "literal_warriors_nickname_header_v1"
    return reasons, checks
