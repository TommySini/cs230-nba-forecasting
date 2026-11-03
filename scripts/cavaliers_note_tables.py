"""Decode the dated Cleveland recent-game tables, preserving source citations."""
from collections import Counter
from datetime import date
import re

from run_game_note_research import page_lines

HEADER = 'OPPONENT W/L MIN FG-A 3P-A FT-A OR DR TR A PF ST TO BS PTS'
SIDEBAR = r'\s+(?:PTS|FGM|FGA|3FGM|3FGA|FTM|FTA|OREB|DREB|TREB|ASTS|STLS|BLKS|MINS)\b'


def source_date(raw, season_start):
    parts = list(map(int, raw.split('/')))
    month, day = parts[:2]
    year = parts[2] if len(parts) == 3 else season_start + int(month < 7)
    if year < 100:
        year += 2000
    return date(year, month, day).isoformat()


def parse_row(text, season_start, game_date):
    match = re.match(r'^(\d{1,2}/\d{1,2}(?:/\d{2,4})?) (at|vs\.?|@) ([A-Z]{2,4}) ([WL])(?: (.*))?$', text)
    if not match:
        raise ValueError('Unrecognized dated recent-game row')
    day = source_date(match[1], season_start)
    if day >= game_date:
        raise ValueError('Target/future date in recent-player table')
    row = {'date': day, 'site': match[2], 'opponent': match[3], 'team_result': match[4]}
    remainder = match[5] or ''
    if not remainder or re.match(r'^(PTS|FGM|FGA|3FGM|3FGA|FTM|FTA|OREB|DREB|TREB|ASTS|STLS|BLKS|MINS)\b', remainder):
        return {**row, 'status': 'participation detail not specified in extracted text', 'missing_detail': True}
    fields = remainder.split()
    if re.fullmatch(r'\d{1,2}:\d{2}', fields[0]):
        if len(fields) < 13:
            raise ValueError('Incomplete numeric table row')
        pairs = [re.fullmatch(r'(\d+)[−–-](\d+)', token) for token in fields[1:4]]
        if not all(pairs) or any(not re.fullmatch(r'\d+', token) for token in fields[4:13]):
            raise ValueError('Unexpected shooting/count column format')
        (fg, fga), (three, three_a), (ft, fta) = [(int(p[1]), int(p[2])) for p in pairs]
        off, defensive, rebounds, assists, fouls, steals, turnovers, blocks, points = map(int, fields[4:13])
        minutes, seconds = map(int, fields[0].split(':'))
        if seconds >= 60 or points != 2 * fg + three + ft or rebounds != off + defensive:
            raise ValueError('Scoring/rebound/minute identity failed')
        if not (0 <= three <= fg <= fga and three <= three_a <= fga and 0 <= ft <= fta):
            raise ValueError('Shooting counts inconsistent')
        tail = ' '.join(fields[13:])
        if tail and not re.match(r'^(PTS|FGM|FGA|3FGM|3FGA|FTM|FTA|OREB|DREB|TREB|ASTS|STLS|BLKS|MINS)\b', tail):
            raise ValueError('Unexpected trailing numeric columns')
        row.update(status='stat_line', minutes=fields[0], seconds=minutes * 60 + seconds,
                   points=points, rebounds=rebounds, assists=assists)
    else:
        status = re.split(SIDEBAR, remainder, maxsplit=1)[0].strip()
        if not re.match(r'^(Inactive|DNP|DND|Did Not|NWT|G League|Not With|Suspended)\b', status, flags=re.I):
            raise ValueError('Unrecognized nonparticipation status')
        row['status'] = status
    return row


def decode_packet(packet):
    if packet['publisher'] != 'CLE':
        return [], []
    tables, issues = [], []
    for page in packet['pages']:
        lines = page_lines(page)
        if not any('PLAYER PAGES' in line['text'] for line in lines[:6]):
            continue
        season = next((re.search(r'(20\d{2})-(\d{2}) PLAYER PAGES', line['text']) for line in lines[:6]
                       if re.search(r'(20\d{2})-(\d{2}) PLAYER PAGES', line['text'])), None)
        if not season or int(season[2]) != (int(season[1]) + 1) % 100:
            issues.append({'page': page['page'], 'reason': 'Missing/unrecognized player-table season heading'})
            continue
        season_start = int(season[1])
        player, active = None, None
        for line in lines:
            name = re.fullmatch(r'#\s*\d+\s+(.+)', line['text'])
            if name:
                player, active = {'name': name[1].strip(), 'heading': line}, None
                continue
            if line['text'].startswith(HEADER) and player:
                active = {**player, 'header': line, 'rows': [], 'errors': [], 'page': page['page']}
                tables.append(active)
                continue
            if line['text'].startswith('NOTES:'):
                active = None
            if active and re.match(r'^\d{1,2}/\d{1,2}(?:/\d{2,4})?\s', line['text']):
                try:
                    row = parse_row(line['text'], season_start, packet['game_date_et'])
                    active['rows'].append({**row, 'line': line})
                except ValueError as error:
                    active['errors'].append({'line_id': line['id'], 'reason': str(error), 'text': line['text']})
    usable = []
    for table in tables:
        if table['errors']:
            issues.append({'page': table['page'], 'player': table['name'], 'reason': 'Entire table excluded after a row failed validation', 'rows': table['errors']})
        elif not table['rows']:
            issues.append({'page': table['page'], 'player': table['name'], 'reason': 'No dated rows decoded'})
        elif len({r['date'] for r in table['rows']}) != len(table['rows']):
            issues.append({'page': table['page'], 'player': table['name'], 'reason': 'Duplicate game dates within a player table'})
        else:
            usable.append(table)
            missing = [r['line']['id'] for r in table['rows'] if r.get('missing_detail')]
            if missing:
                issues.append({'page': table['page'], 'player': table['name'], 'reason': 'Blank participation detail preserved as unknown; no absence inferred', 'line_ids': missing})
    return usable, issues


def packet_facts(packet):
    tables, issues = decode_packet(packet)
    facts = []
    for table in tables:
        rows = sorted(table['rows'], key=lambda r: r['date'])
        latest = rows[-1]
        citations = [table['heading'], table['header']]
        prefix = f"{table['name']}'s latest listed game ({latest['date']}, {latest['site']} {latest['opponent']})"
        if latest['status'] == 'stat_line':
            claim = f"{prefix} lists {latest['minutes']} minutes, {latest['points']} points, {latest['rebounds']} rebounds and {latest['assists']} assists."
        else:
            claim = f"{prefix} lists status {latest['status']}. This is a dated prior-game entry."
        facts.append({'claim': claim, 'category': 'workload' if latest['status'] == 'stat_line' else 'availability',
                      'time_scope': 'prior_game', 'caveat': 'A prior-game entry does not confirm availability for the upcoming game.',
                      'citations': citations + [latest['line']]})
        played = [r for r in rows if r['status'] == 'stat_line']
        if len(played) > 1:
            n = len(played)
            facts.append({'claim': f"Across the {n} listed stat lines from {played[0]['date']} through {played[-1]['date']}, {table['name']} averaged {sum(r['seconds'] for r in played)/(60*n):.2f} minutes and {sum(r['points'] for r in played)/n:.2f} points.",
                          'category': 'workload', 'time_scope': 'recent_games',
                          'caveat': 'Computed from listed numeric rows only; other status or missing-detail entries excluded. Not a full-season average.',
                          'citations': citations + [r['line'] for r in played]})
        statuses = Counter(r['status'] for r in rows if r['status'] != 'stat_line')
        if statuses:
            text = '; '.join(f'{status}: {count}' for status, count in sorted(statuses.items()))
            facts.append({'claim': f"Among {table['name']}'s {len(rows)} listed team games from {rows[0]['date']} through {rows[-1]['date']}, {len(played)} have numeric stat lines. Other recorded entries: {text}.",
                          'category': 'rotation', 'time_scope': 'recent_games',
                          'caveat': 'Historical participation entries; does not establish current availability or the cause of any unstated absence.',
                          'citations': citations + [r['line'] for r in rows]})
    for i, fact in enumerate(facts):
        fact.update(id=packet['game_id'].replace(':', '-') + f'-cle-table-{i:04}',
                    origin='deterministic_table',
                    support_check={'verdict': 'supported', 'reason': 'Cleveland table prefix decoded with exact header, season/date, field-count, scoring and rebound checks; means and status counts calculated by code.'})
    return facts, issues
