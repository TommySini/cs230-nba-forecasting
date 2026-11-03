"""Decode two explicitly headed season-average layouts without language-model arithmetic.

The original page spacing is needed to separate player cards from schedules and
notes printed beside them. Unrecognized layouts remain undecoded source material.
Facts describe what the table lists, including season, preseason and sample size;
they never turn a probable-starter card into current availability confirmation.
"""
from datetime import date
import hashlib
import json
import re

NUMBER = r'(?:\d+(?:\.\d+)?|\.\d+)'
PLAYER_SEASON = re.compile(r'PLAYER\s*/\s*(20\d{2})[-–](\d{2})\s+(PRESEASON|REGULAR\s+SEASON)\s+AVERAGES', re.I)
PLAYER_HEADER = re.compile(r'GP/GS\s+PPG\s+RPG\s+APG\s+SPG\s+BPG\s+MPG\b')
PLAYER_ROW = re.compile(r'(?<!\S)(\d{1,3})/(\d{1,3})\s+' + r'\s+'.join(['('+NUMBER+')'] * 6) + r'(?![\d.])')
TEAM_HEADER = re.compile(r'^\s*(20\d{2})[-–](\d{2})\s+REGULAR\s+SEASON\s+PPG\s+OPP\.\s*PPG\s+FG%\s+3FG%\s+FT%\s+RPG\s+APG\s+SPG\s+TO\s+BPG\s*$', re.I)
TEAM_ROW = re.compile(r'^\s*(76ers|[A-Za-z][A-Za-z .\'-]{1,40}?)\s+' + r'\s+'.join(['('+NUMBER+')'] * 10) + r'\s*$')
CARD_START = re.compile(r'#\d{1,3}(?=\s|$)')
NEW_SCOPE = re.compile(r'(?:LAST\s+\d+\s+GAMES|RECENT\s+GAMES|CAREER(?:\s+(?:HIGHS|AVERAGES|STATISTICS|STATS))?|(?:20\d{2}[-–]\d{2}\s+)?(?:PRESEASON|REGULAR\s+SEASON|PLAYOFFS?|POSTSEASON)(?:\s+(?:AVERAGES|STATISTICS|STATS|RECORDS))?)', re.I)


def _season(year, ending, game):
    year = int(year)
    if int(ending) != (year + 1) % 100:
        raise ValueError('Season heading has inconsistent years')
    day = date.fromisoformat(game['game_date_et'])
    active_year = day.year if day.month >= 7 else day.year - 1
    if year > active_year:
        raise ValueError('Average table refers to a future season')
    return f'{year}-{ending}', year == active_year


def _fact(source, page, lines, kind, claim, scope, caveat, values):
    citations = [{'source_id': source['source_id'], 'page': page['page'],
                  'lines': sorted(set(lines))}]
    identity = {'source_id': source['source_id'], 'page': page['page'],
                'lines': citations[0]['lines'], 'decoder': kind, 'values': values}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {'id': 'season-table:' + digest, 'fact': claim, 'time_scope': scope,
            'uncertainty': caveat, 'citations': citations,
            'decoder': kind, 'parsed_values': values}


def _name_segment(text):
    candidate = re.split(r'\s{2,}', text.strip(), maxsplit=1)[0]
    if (not 2 <= len(candidate.split()) <= 6 or len(candidate) > 80
            or any(not (ch.isalpha() or ch in " .'-’") for ch in candidate)):
        return None
    return candidate


def _player_name(text):
    match = re.search(r'#\d{1,3}\s+(.+)', text)
    return _name_segment(match[1]) if match else None


def decode_snapshot_tables(game):
    """Return BAM-compatible exact facts and explicit recognized-layout failures."""
    facts, failures = [], []
    for source in game['sources']:
        for page in source['pages']:
            lines = page['lines']
            season = player = pending_card = None
            for index, line in enumerate(lines):
                text = line['text'].replace('\x00', '').rstrip('\n')
                heading = PLAYER_SEASON.search(text)
                if heading:
                    season = (heading, line['line'])
                    player = pending_card = None
                elif season:
                    # An explicit new scope ends this averages block. Match a
                    # short heading in the card region, not words buried in notes.
                    segment = re.split(r'\s{2,}', text[season[0].start():].strip(), maxsplit=1)[0]
                    if NEW_SCOPE.fullmatch(segment):
                        season = player = pending_card = None
                card = CARD_START.search(text)
                if card:
                    # A card with an unparsed name MUST end the previous player's
                    # ownership; otherwise its stats can inherit the wrong name.
                    player = None
                    pending_card = (index, line['line'], card.start())
                    name = _player_name(text)
                    if name:
                        player = (name, [line['line']], index)
                        pending_card = None
                elif pending_card and index-pending_card[0] <= 2:
                    # Some PDFs wrap the name below the jersey/position line.
                    # Crop the neighboring left schedule using the card's column.
                    name = _name_segment(text[pending_card[2]:])
                    if name:
                        player = (name, [pending_card[1],line['line']], index)
                        pending_card = None
                header = PLAYER_HEADER.search(text)
                if not header or season is None or player is None or index-player[2] > 6:
                    continue
                # Values form a complete contiguous seven-column block. Adjacent
                # prose/schedule text is neither included nor used to infer columns.
                candidates = []
                for following in lines[index+1:index+4]:
                    following_text = following['text'].replace('\x00','').strip('\n')
                    following_scope = re.split(r'\s{2,}', following_text[season[0].start():].strip(), maxsplit=1)[0]
                    if (CARD_START.search(following_text) or PLAYER_HEADER.search(following_text)
                            or PLAYER_SEASON.search(following_text) or NEW_SCOPE.fullmatch(following_scope)):
                        break
                    candidates.extend((following, match) for match in
                        PLAYER_ROW.finditer(following['text'].replace('\x00', '')))
                try:
                    if len(candidates) != 1:
                        raise ValueError('Player averages need exactly one GP/GS plus six-value row')
                    candidate, row = candidates[0]
                    if not -8 <= row.start()-header.start() <= 8:
                        raise ValueError('Player values do not align with the explicit table header')
                    tail = candidate['text'].replace('\x00','')[row.end():].lstrip()
                    adjacent_schedule = re.match(r'^\d{1,2}/\d{1,2}\s+(?:vs\.?|@|at)\s+\S+', tail)
                    if re.match(r'^[\d.]', tail) and not adjacent_schedule:
                        raise ValueError('Unexpected extra numeric player-average column')
                    season_name, current = _season(season[0][1], season[0][2], game)
                    kind = 'preseason' if season[0][3].upper() == 'PRESEASON' else 'regular-season'
                    gp, gs = int(row[1]), int(row[2])
                    ppg, rpg, apg, spg, bpg, mpg = map(float, row.groups()[2:])
                    if not (0 < gp <= 100 and 0 <= gs <= gp and 0 <= mpg <= 80
                            and all(0 <= v <= 150 for v in (ppg,rpg,apg,spg,bpg))):
                        raise ValueError('Player averages fail sample-size or nonnegative/range checks')
                    values = dict(player=player[0], season=season_name, season_type=kind,
                                  games_played=gp, games_started=gs, points_per_game=ppg,
                                  rebounds_per_game=rpg, assists_per_game=apg,
                                  steals_per_game=spg, blocks_per_game=bpg, minutes_per_game=mpg)
                    claim = (f"{player[0]}'s {season_name} {kind} averages table lists {gp} games played "
                             f"and {gs} starts: {row[3]} points, {row[4]} rebounds, {row[5]} assists, "
                             f"{row[6]} steals, {row[7]} blocks and {row[8]} minutes per game.")
                    facts.append(_fact(source,page,[season[1],*player[1],line['line'],candidate['line']],
                        'player_season_averages_v1',claim,'season_to_date' if current and kind=='regular-season' else 'historical',
                        'Published rounded averages; the explicit game count is the sample size. Preseason is separate from regular season. A player card does not confirm availability or the upcoming starting lineup.',values))
                except ValueError as exc:
                    failures.append({'source_id':source['source_id'],'page':page['page'],
                                     'line':line['line'],'decoder':'player_season_averages_v1','reason':str(exc)})
            for index,line in enumerate(lines):
                heading = TEAM_HEADER.fullmatch(line['text'].replace('\x00','').strip())
                if not heading:
                    continue
                try:
                    season_name,current = _season(heading[1],heading[2],game)
                    # Only the immediately following contiguous named rows belong
                    # to this exact heading, never a later records/series table.
                    for candidate in lines[index+1:index+5]:
                        if not candidate['text'].strip():
                            continue
                        row = TEAM_ROW.fullmatch(candidate['text'].replace('\x00','').strip())
                        if not row:
                            break
                        values_raw = row.groups()[1:]
                        vals = list(map(float,values_raw))
                        if not (all(0 <= v <= 250 for v in vals) and all(0 <= v <= 1 for v in vals[2:5])):
                            raise ValueError('Team averages fail percentage-unit or nonnegative/range checks')
                        team = row[1].strip()
                        fields = ('points_per_game','opponent_points_per_game','field_goal_fraction',
                                  'three_point_fraction','free_throw_fraction','rebounds_per_game',
                                  'assists_per_game','steals_per_game','turnovers_per_game','blocks_per_game')
                        values = dict(team=team,season=season_name,season_type='regular-season',**dict(zip(fields,vals)))
                        v=values_raw
                        claim=(f"The {team} {season_name} regular-season table lists {v[0]} points scored and {v[1]} points allowed per game; "
                               f"shooting fractions FG {v[2]}, 3FG {v[3]} and FT {v[4]}; and per-game averages of "
                               f"{v[5]} rebounds, {v[6]} assists, {v[7]} steals, {v[8]} turnovers and {v[9]} blocks.")
                        facts.append(_fact(source,page,[line['line'],candidate['line']],
                            'team_season_averages_v1',claim,'season_to_date' if current else 'historical',
                            'Published rounded team averages, not opponent-adjusted rates or causal effects. Shooting values are fractions; this table does not specify a game-count denominator.',values))
                except ValueError as exc:
                    failures.append({'source_id':source['source_id'],'page':page['page'],
                                     'line':line['line'],'decoder':'team_season_averages_v1','reason':str(exc)})
    return facts, failures
