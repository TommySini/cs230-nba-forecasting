"""Conservative, source-cited decoding of explicit recent-player game tables."""
from datetime import datetime
import re

HEADER = 'DATE OPP. MIN FG FGA 3FG 3FGA FT FTA OR DR TOT A PF ST TO BS +/- PTS'
FIELDS = ('fg', 'fga', 'three_fg', 'three_fga', 'ft', 'fta', 'off_rebounds', 'def_rebounds',
          'rebounds', 'assists', 'fouls', 'steals', 'turnovers', 'blocks', 'plus_minus', 'points')


def numeric_row(text):
    """Identify dense rows whose column meaning cannot safely be guessed by an LM."""
    tokens = text.split()
    numeric = sum(bool(re.fullmatch(r'[+−-]?\d+(?:[.:/−-]\d+)*%?', t)) for t in tokens)
    return numeric >= 6 and numeric / max(len(tokens), 1) >= .45


def decode_recent_tables(unit):
    tables, current = [], None
    for line in unit['lines']:
        title = re.fullmatch(r'(.+?) RECENT GAMES', line['text'])
        if title:
            current = {'player': title[1].title(), 'heading': line, 'header': None, 'rows': []}
            tables.append(current)
            continue
        if line['text'] == HEADER and current:
            current['header'] = line
            continue
        if not current or current['header'] is None:
            continue
        match = re.fullmatch(r'(\d{2}/\d{2}/\d{2}) (vs|at) ([A-Z]{2,3}) (.+)', line['text'])
        if not match:
            # A new player section ends the preceding table.
            if line['text'].startswith('# '):
                current = None
            continue
        date = datetime.strptime(match[1], '%m/%d/%y').date().isoformat()
        if date >= unit['game']['game_date_et']:
            raise ValueError('Recent-player table contains target/future dated data')
        row = {'date': date, 'site': match[2], 'opponent': match[3], 'line': line}
        values = match[4].split()
        if re.fullmatch(r'\d{1,2}:\d{2}', values[0]):
            if len(values) != 17 or any(not re.fullmatch(r'[+-]?\d+', v) for v in values[1:]):
                raise ValueError('Recognized recent-game table has an unexpected numeric column count')
            row.update(zip(FIELDS, map(int, values[1:])))
            minutes, seconds = map(int, values[0].split(':'))
            if seconds >= 60 or row['points'] != 2 * row['fg'] + row['three_fg'] + row['ft'] or row['rebounds'] != row['off_rebounds'] + row['def_rebounds']:
                raise ValueError('Recent-game table fails scoring/rebound/minute identities')
            if not (0 <= row['three_fg'] <= row['fg'] <= row['fga'] and row['three_fg'] <= row['three_fga'] <= row['fga'] and 0 <= row['ft'] <= row['fta']):
                raise ValueError('Recent-game shooting counts are inconsistent')
            row.update(minutes=values[0], seconds=minutes * 60 + seconds, status='stat_line')
        elif re.fullmatch(r'(DNP-CD|NWT|INACTIVE|G LEAGUE|DND)(?: .*)?', match[4]):
            row['status'] = match[4]
        else:
            # Unrecognized status stays in the raw source, with no invented interpretation.
            continue
        current['rows'].append(row)
    return [t for t in tables if t['header'] and t['rows']]


def table_facts(unit):
    facts = []
    for table in decode_recent_tables(unit):
        rows = sorted(table['rows'], key=lambda r: r['date'])
        latest = rows[-1]
        prefix = f"{table['player']}'s latest listed game ({latest['date']}, {latest['site']} {latest['opponent']})"
        if latest['status'] == 'stat_line':
            claim = f"{prefix} lists {latest['minutes']} minutes, {latest['points']} points, {latest['rebounds']} rebounds and {latest['assists']} assists."
        else:
            claim = f"{prefix} lists status {latest['status']}. This is a dated prior-game entry."
        facts.append({'claim': claim, 'category': 'workload' if latest['status'] == 'stat_line' else 'availability',
                      'time_scope': 'prior_game', 'caveat': 'A prior-game entry does not confirm availability for the upcoming game.',
                      'citations': [table['heading'], table['header'], latest['line']]})
        played = [r for r in rows if r['status'] == 'stat_line']
        if len(played) > 1:
            n = len(played)
            avg_points = sum(r['points'] for r in played) / n
            avg_minutes = sum(r['seconds'] for r in played) / (60 * n)
            claim = (f"Across the {n} listed stat lines from {played[0]['date']} through {played[-1]['date']}, "
                     f"{table['player']} averaged {avg_minutes:.2f} minutes and {avg_points:.2f} points.")
            facts.append({'claim': claim, 'category': 'workload', 'time_scope': 'recent_games',
                          'caveat': 'Computed from listed numeric rows only; nonparticipation rows are excluded. This is not a full-season average.',
                          'citations': [table['heading'], table['header']] + [r['line'] for r in played]})
    for i, fact in enumerate(facts):
        fact.update(id=unit['id'] + f'-t{i:03}', unit_id=unit['id'], origin='deterministic_table',
                    mechanical_verdict={'verdict': 'supported', 'reason': 'Decoded by code from named columns; date, field count, scoring identity and rebound identity checked. Means computed by code.'})
    return facts


def table_context(unit):
    facts = table_facts(unit)
    if not facts:
        return ''
    lines = ['<checked_table_context>', 'The following facts were decoded and calculated by code from the explicit source headers and dated rows. They are already retained separately; do not repeat or recalculate them.']
    for fact in facts:
        lines.append(f"{fact['claim']} {fact['caveat']} Source lines: {[c['line'] for c in fact['citations']]}")
    lines.append('</checked_table_context>')
    return '\n'.join(lines)


def mechanical_claim_check(fact, unit):
    # Language-model interpretations of dense raw tables are not trusted. The
    # supported adapter above supplies canonical workload/status facts instead.
    if any(numeric_row(c['text']) for c in fact['citations']):
        return {'verdict': 'unclear', 'reason': 'LM interpretation of a dense statistical row excluded; use the separately decoded table facts when available.'}
    table_row_ids = {r['line']['id'] for table in decode_recent_tables(unit) for r in table['rows']}
    if any(c['id'] in table_row_ids for c in fact['citations']):
        return {'verdict': 'unclear', 'reason': 'LM interpretation of dated player table entries excluded; canonical dated table facts are retained separately.'}
    cited = ' '.join(c['text'] for c in fact['citations'])
    pattern = r'(?<![A-Za-z0-9])(?:\d[\d,]*(?:\.\d+)?|\.\d+)'
    values = lambda text: {float(v.replace(',', '')) for v in re.findall(pattern, text)}
    source_values = values(cited)
    year = int(unit['game']['game_date_et'][:4])
    source_values.update((year, year - 1))
    source_values.update(2000 + int(y) for y in re.findall(r'\d{1,2}/\d{1,2}/(\d{2})\b', cited))
    source_values.update(v * 100 for v in list(source_values) if 0 <= v < 1)
    if any(not any(abs(v - s) < 1e-8 for s in source_values) for v in values(fact['claim'])):
        return {'verdict': 'unclear', 'reason': 'A numerical value in the LM claim is absent from its cited prose; new arithmetic is reserved for validated code.'}
    if fact['category'] == 'availability' and fact['time_scope'] == 'current':
        if not re.search(r'\b(out|doubtful|questionable|probable|available|tonight|today|tbd|unknown)\b', cited.lower()):
            return {'verdict': 'unclear', 'reason': 'Citations lack an explicit current-game availability statement; past participation or injury history is insufficient.'}
    return None
