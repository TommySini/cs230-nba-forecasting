"""Mechanical evidence guards; these do not certify semantic entailment.

Kept separate from the frozen v4 experiment until its source audit finishes.
Dates must be present in cited source text or explicitly resolved by a cited
season heading. A missing year stays unresolved; the prediction year is never
silently supplied. Entity/date association still needs source review.
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
import re


_NUMERIC_CELL = re.compile(r"[+−-]?(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)(?:[.:/−-]\d+)*%?")
_AGGREGATE_LABEL = re.compile(r"^(?:(?:TEAM|OPPONENT)\s+)?(?:TOTALS?|AVERAGES?|OPPONENTS?|PER\s+GAME)\b", re.I)
_MONTH_NAMES = "January February March April May June July August September October November December".split()
_MONTH_NUMBERS = {name.lower(): i for i, name in enumerate(_MONTH_NAMES, 1)}
_MONTH_NUMBERS.update({name[:3].lower(): i for i, name in enumerate(_MONTH_NAMES, 1)})
_MONTH_NUMBERS["sept"] = 9
_MONTH = "(?:" + "|".join(sorted(_MONTH_NUMBERS, key=len, reverse=True)) + r")\.?"
_NAMED_DATE = re.compile(r"\b(" + _MONTH + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+((?:19|20)\d{2}))?\b", re.I)
_SEASON = re.compile(r"\b((?:19|20)\d{2})\s*[-–/]\s*((?:19|20)\d{2}|\d{2})\b")


def dense_statistical_row(text):
    """Recognize numeric columns, including short labeled/thousands-total rows.

    This identifies rows needing a decoder, not their column meanings. It does
    not derive rates or relabel a total as an average.
    """
    tokens = text.strip().split()
    numeric = sum(bool(_NUMERIC_CELL.fullmatch(token)) for token in tokens)
    if numeric >= 6 and numeric / max(len(tokens), 1) >= .45:
        return True
    # Named aggregate rows can have fewer than six columns. Require almost all
    # remaining tokens to be numeric, so ordinary sentences are not quarantined.
    match = _AGGREGATE_LABEL.match(text.strip())
    if match:
        tail = text.strip()[match.end():].strip().split()
        numeric_tail = sum(bool(_NUMERIC_CELL.fullmatch(token)) for token in tail)
        return numeric_tail >= 3 and numeric_tail / max(len(tail), 1) >= .8
    return False


def _valid_day(year, month, day):
    try:
        return date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return None


def _source_dates(text):
    explicit, undated = set(), []
    for year, month, day in re.findall(r"\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b", text):
        if value := _valid_day(year, month, day):
            explicit.add(value)
    for month, day, year in re.findall(r"(?<!\d)(\d{1,2})/(\d{1,2})/((?:19|20)\d{2}|\d{2})(?!\d)", text):
        year = int(year) if len(year) == 4 else datetime.strptime(year, "%y").year
        if value := _valid_day(year, month, day):
            explicit.add(value)
    for match in _NAMED_DATE.finditer(text):
        month = _MONTH_NUMBERS[match[1].rstrip(".").lower()]
        if match[3]:
            if value := _valid_day(match[3], month, match[2]):
                explicit.add(value)
        else:
            undated.append((month, int(match[2])))
    return explicit, undated


def date_grounding(fact_date, citations):
    """Return a source-date check, never an entity or time-scope certification.

    Each citation must already be canonicalized against the immutable source.
    Season-based resolution is deliberately narrow: one unambiguous consecutive
    season heading within the same citation as the month/day. Ambiguous/uncited
    years require a null fact_date and a qualified textual statement.
    """
    if fact_date is None:
        return {"grounded": True, "method": "unspecified_date", "source_dates": []}
    if not isinstance(fact_date, str):
        raise ValueError("fact_date must be ISO date or null")
    try:
        if date.fromisoformat(fact_date).isoformat() != fact_date:
            raise ValueError
    except ValueError as exc:
        raise ValueError("fact_date must be ISO date or null") from exc
    explicit, seasonal = set(), set()
    for citation in citations:
        quotes = citation.get("quotations")
        if not isinstance(quotes, list) or not all(isinstance(q.get("text"), str) for q in quotes):
            raise ValueError("Canonical source quotations required for date grounding")
        text = "\n".join(q["text"] for q in quotes)
        dates, undated = _source_dates(text)
        explicit.update(dates)
        seasons = set()
        for start, end in _SEASON.findall(text):
            first = int(start)
            last = int(end) if len(end) == 4 else first // 100 * 100 + int(end)
            if last == first + 1:
                seasons.add((first, last))
        if len(seasons) == 1:
            first, last = next(iter(seasons))
            for month, day in undated:
                if value := _valid_day(first if month >= 7 else last, month, day):
                    seasonal.add(value)
    result = {"grounded": fact_date in explicit | seasonal, "source_dates": sorted(explicit | seasonal),
              "limitation": "Date presence only; entity, event and season association require semantic review."}
    result["method"] = "explicit_cited_date" if fact_date in explicit else "cited_season_and_month_day" if fact_date in seasonal else "unresolved_year_or_date"
    if not result["grounded"]:
        result["reason"] = "fact_date is not grounded in cited dates; cite the dated context or use null, preserving any qualified month/day in prose."
    return result


def prose_date_grounding(text, citations):
    """Check explicit full dates in prose, without inventing years for month/day.

    No entity/event association is certified. A named month/day without a year
    stays as written; a supplied year must have cited date/season support.
    """
    explicit, _ = _source_dates(text)
    checks = [date_grounding(value, citations) | {'date': value} for value in sorted(explicit)]
    return {'grounded': all(check['grounded'] for check in checks), 'checks': checks}


def cited_season_years(citations):
    """Expand only explicit consecutive cited season labels (2024-25 ->2025)."""
    years = set()
    for citation in citations:
        text = '\n'.join(q['text'] for q in citation['quotations'])
        for start, end in _SEASON.findall(text):
            first = int(start)
            last = int(end) if len(end) == 4 else first // 100 * 100 + int(end)
            if last == first + 1:
                years.update((first, last))
    return years


# Deliberately a small literal grammar. It never binds numbers from a metric
# name (3-point), a record (20-5), a rank, or a multi-value parenthesis.
_SCALAR = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)(?:\.\d+)?"
_METRIC = r"(?:(?:opponent\s+)?points\s+per\s+game|PPG|rebounds\s+per\s+game|RPG|assists\s+per\s+game|APG|turnovers\s+per\s+game|(?:3-point|three-point|field[- ]goal|free[- ]throw)\s+percentage|3FG%|FG%|FT%)"
_OPERATOR = r"higher|lower|greater|less"
_PAREN_COMPARISON = re.compile(
    rf"\b(?P<op>{_OPERATOR})\s+(?P<metric>{_METRIC})\s*"
    rf"\(\s*(?P<a>{_SCALAR})\s*(?P<au>%?)\s*\)\s*"
    rf"(?:compared\s+to|than)\s+(?P<rhs>(?:(?:the\s+)?opponent['’]s\s*)?(?:{_METRIC})?|(?:the\s+)?opponent['’]s)"
    rf"\s*\(\s*(?P<b>{_SCALAR})\s*(?P<bu>%?)\s*\)", re.I)
_INLINE_COMPARISON = re.compile(
    rf"(?<![\w.\d-])(?P<a>{_SCALAR})\s*(?P<au>%?)\s+(?P<metric>{_METRIC})\s+"
    rf"(?:is|was|are|were)\s+(?P<op>{_OPERATOR})\s+than\s+"
    rf"(?P<b>{_SCALAR})\s*(?P<bu>%?)\s+(?P<rhs>{_METRIC})(?![\w-])", re.I)


def _comparison_metric(text):
    text = re.sub(r"^(?:the\s+)?opponent['’]s\s*", "", text.lower()).strip()
    text = re.sub(r"^opponent\s+", "", text)
    aliases = {'ppg': 'points per game', 'rpg': 'rebounds per game', 'apg': 'assists per game',
               '3fg%': '3-point percentage', 'three-point percentage': '3-point percentage',
               'fg%': 'field goal percentage', 'ft%': 'free throw percentage'}
    return aliases.get(text, text.replace('-', ' ') if text.startswith(('field-', 'free-')) else text)


def literal_comparison_consistency(text):
    """Reject only contradictory literal comparisons with identifiable units.

    A skipped/consistent statement has NOT been verified against a source.
    Threshold semantics, implied operands and arbitrary prose stay with review.
    """
    checks = []
    for pattern in (_PAREN_COMPARISON, _INLINE_COMPARISON):
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start()-30):match.start()]
            if re.search(r"(?:\bnot\b|\bno\b|\bnever\b|n't|n’t)[^.;!?]*$", prefix, re.I):
                continue
            left = _comparison_metric(match['metric'])
            right = _comparison_metric(match['rhs'])
            # Only a possessive ellipsis may inherit the named metric.
            if not right and not re.fullmatch(r"(?:the\s+)?opponent['’]s\s*", match['rhs'], re.I):
                continue
            if (right and right != left) or match['au'] != match['bu']:
                continue
            a, b = (Decimal(match[key].replace(',', '')) for key in ('a', 'b'))
            ascending = match['op'].lower() in ('higher', 'greater')
            checks.append({'text': match[0], 'metric': left, 'left': str(a), 'right': str(b),
                           'operator': match['op'].lower(), 'consistent': a > b if ascending else a < b})
    return {'consistent': all(check['consistent'] for check in checks), 'checks': checks,
            'limitation': 'Literal arithmetic only; source support and ambiguous comparisons require semantic review.'}


_SMALL_NUMBERS = {word: str(i) for i, word in enumerate('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split())}
_AVERAGE_NUMBER = rf"(?:{_SCALAR}|{'|'.join(_SMALL_NUMBERS)})"
_AVERAGE_ITEM = rf"(?P<value>{_AVERAGE_NUMBER})\s+(?P<unit>points?|rebounds?|assists?|steals?|blocks?|minutes?)\b"
_AVERAGE_ITEM_RE = re.compile(_AVERAGE_ITEM, re.I)
_AVERAGE_ITEM_BODY = rf"{_AVERAGE_NUMBER}\s+(?:points?|rebounds?|assists?|steals?|blocks?|minutes?)\b"
_AVERAGE_LIST = re.compile(
    rf"\baverag(?:e[sd]?|ing)\s+(?P<bound>at\s+least\s+)?"
    rf"(?P<stats>{_AVERAGE_ITEM_BODY}(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+){_AVERAGE_ITEM_BODY}){{1,5}})", re.I)


def threshold_average_precision(text, citations):
    """Flag loss of a literal lower-bound qualifier, not threshold truth.

    Only identical ordered lists of at least two explicitly named statistics
    after 'averages/averaging' are compared. An independent exact source list
    avoids this narrow flag. Different values, implicit metrics, single values
    and other grammatical forms remain semantic-review work.
    """
    def lists(value):
        result = []
        for match in _AVERAGE_LIST.finditer(value):
            prefix = value[max(0, match.start()-30):match.start()]
            if re.search(r"(?:\bnot\b|\bnever\b|n't|n’t)[^.;!?]*$", prefix, re.I):
                continue
            items = []
            for item in _AVERAGE_ITEM_RE.finditer(match['stats']):
                raw = item['value'].lower()
                number = Decimal(_SMALL_NUMBERS.get(raw, raw).replace(',', ''))
                items.append((number, item['unit'].lower().rstrip('s')))
            result.append((tuple(items), bool(match['bound']), match[0]))
        return result
    source_lists = [value for cite in citations for value in lists('\n'.join(q['text'] for q in cite['quotations']))]
    exact_source = {items for items, bound, _ in source_lists if not bound}
    lower_bounds = {items for items, bound, _ in source_lists if bound}
    flags = [{'candidate': phrase, 'reason': 'Cited matching statistics are lower bounds, not exact averages; retain at least or cite independent exact averages.'}
             for items, bound, phrase in lists(text) if not bound and items in lower_bounds and items not in exact_source]
    return {'supported_precision': not flags, 'flags': flags,
            'limitation': 'Literal qualifier consistency only; entities, statistical scope and source entailment still require semantic review.'}
