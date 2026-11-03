"""Reject pregame documents that contain their own game's result.

Policy (user decision, 2026-11-01): documents proven only by the publisher's own
timestamps are allowed, but any document showing the outcome of the game it is
used for is excluded. A document is flagged when it contains:
  * the game's final score as a pair (e.g. "112-105", either order) on the same line
    as this game's date, or as both teams with no other date (long packets list many
    earlier scores and repeat the game date in page headers, so a bare match is not enough);
  * a team's post-game record without its pre-game record, on the opening page;
  * result language about this game: a result verb within the same sentence as
    the opponent and this game's date, or "tonight's win/loss"-style phrasing.
Scores and records from earlier games are normal pregame content and pass.
Flags are conservative: a coincidental match drops a document rather than
risking a leak.
"""
from datetime import date
import re

DASH = r"\s*[-–—]\s*"
RESULT_VERBS = (r"beat|beats|defeated|defeats|downed|topped|edged|routed|outlasted|"
                r"fell to|falls to|lost to|loses to|won|win over|victory over|loss to|"
                r"final score|recap|postgame|post-game")
TONIGHT_RESULT = re.compile(r"\b(tonight'?s|today'?s) (win|loss|victory|defeat) (over|to|against)\b", re.I)
HEADER_CHARS = 4000  # records are checked on the opening page only; stat tables repeat W-L-like pairs
SENTENCE = re.compile(r"[^.!?\n]+")
MONTH_DOT = re.compile(r"\b(Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.")  # "Oct. 30" is not a sentence end
ANY_DATE = re.compile(r"\b\d{1,2}/\d{1,2}\b|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.? \d{1,2}\b")


def nickname(team):
    return "Trail Blazers" if team.endswith("Trail Blazers") else team.split()[-1]


def date_patterns(game_date):
    """This game's month and day, unless followed by a different year ("November 3, 1995")."""
    day = date.fromisoformat(game_date)
    month, short = day.strftime("%B"), day.strftime("%b")
    other_year = rf"(?!\s*,?\s*(?!{day.year})\d{{4}})"
    return [rf"\b{month}\.? {day.day}\b{other_year}", rf"\b{short}\.? {day.day}\b{other_year}",
            rf"\b{day.month}/{day.day}\b(?!/(?!{day.year % 100}\b)\d)"]


def record_flags(text, team, before, won, min_games=5):
    """Post-game record present while the pre-game record is absent (skipped very early in a
    season, when short records like 1-0 are common in other contexts and 0-0 is rarely printed)."""
    wins, losses = before
    if wins + losses < min_games:
        return []
    after = (wins + 1, losses) if won else (wins, losses + 1)
    has = lambda w, l: re.search(rf"(?<![\d-]){w}{DASH}{l}(?![\d-])", text) is not None
    return [f"post_game_record:{team}:{after[0]}-{after[1]}"] if has(*after) and not has(*before) else []


def about_this_game(window, dates, teams):
    """Text near a score refers to this game: its date, or both teams with no other date."""
    if any(re.search(p, window) for p in dates):
        return True
    both = all(any(re.search(rf"\b{re.escape(n)}\b", window, re.I) for n in names) for names in teams)
    return both and not ANY_DATE.search(window)


def postgame_flags(text, game):
    """Return reasons this document shows the result of `game`; empty means it passes.

    `game` needs home_team, away_team, home_score, away_score, game_date_et, optionally
    home_abbr / away_abbr and, for the record check, optional home_record_before / away_record_before as (wins, losses).
    """
    flags = []
    home, away = game["home_score"], game["away_score"]
    dates = date_patterns(game["game_date_et"])
    teams = [[nickname(game[f"{side}_team"]), game.get(f"{side}_abbr") or "\0"] for side in ("home", "away")]
    for a, b in ((home, away), (away, home)):
        for match in re.finditer(rf"(?<!\d){a}{DASH}{b}(?!\d)", text):
            start, end = text.rfind("\n", 0, match.start()) + 1, text.find("\n", match.end())
            window = text[max(start, match.start() - 150):min(end if end >= 0 else len(text), match.end() + 150)]
            if about_this_game(window, dates, teams):
                flags.append(f"final_score:{a}-{b}:" + " ".join(window.split())[:160])
                break
    home_won = home > away
    header = text[:HEADER_CHARS]
    for side, won in (("home", home_won), ("away", not home_won)):
        if game.get(f"{side}_record_before"):
            flags += record_flags(header, game[f"{side}_team"], tuple(game[f"{side}_record_before"]), won)
    opponents = [nickname(game["home_team"]), nickname(game["away_team"])]
    for sentence in SENTENCE.findall(MONTH_DOT.sub(r"\1", text)):
        if (re.search(rf"\b({RESULT_VERBS})\b", sentence, re.I)
                and any(re.search(p, sentence) for p in dates)
                and any(name.lower() in sentence.lower() for name in opponents)):
            flags.append("result_sentence:" + sentence.strip()[:120])
            break
    if TONIGHT_RESULT.search(text):
        flags.append("tonight_result:" + TONIGHT_RESULT.search(text).group(0))
    return flags
