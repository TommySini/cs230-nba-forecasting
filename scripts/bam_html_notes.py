"""Pure, fail-closed adapter for archived NBA HTML game-note articles.

Only the article pageObject's own structured blocks become numbered sections.
Archive proof establishes byte availability, not the truth of every claim;
independent content review remains necessary before snapshot admission.
"""
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import gzip
import hashlib
from html.parser import HTMLParser
import json
import re
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo


TEAM_NAMES = dict(zip(
    'ATL BOS BKN CHA CHI CLE DAL DEN DET GSW HOU IND LAC LAL MEM MIA MIL MIN NOP NYK OKC ORL PHI PHX POR SAC SAS TOR UTA WAS'.split(),
    ['Hawks', 'Celtics', 'Nets', 'Hornets', 'Bulls', 'Cavaliers', 'Mavericks',
     'Nuggets', 'Pistons', 'Warriors', 'Rockets', 'Pacers', 'Clippers', 'Lakers',
     'Grizzlies', 'Heat', 'Bucks', 'Timberwolves', 'Pelicans', 'Knicks', 'Thunder',
     'Magic', '76ers', 'Suns', 'Trail Blazers', 'Kings', 'Spurs', 'Raptors', 'Jazz', 'Wizards']))
# Cities are accepted only in the explicitly scoped last-meeting heading.
# Los Angeles is deliberately absent: it cannot distinguish LAC from LAL.
HEADING_CITIES = dict(zip(
    'ATL BOS BKN CHA CHI CLE DAL DEN DET GSW HOU IND MEM MIA MIL MIN NOP NYK OKC ORL PHI PHX POR SAC SAS TOR UTA WAS'.split(),
    ['Atlanta', 'Boston', 'Brooklyn', 'Charlotte', 'Chicago', 'Cleveland',
     'Dallas', 'Denver', 'Detroit', 'Golden State', 'Houston', 'Indiana',
     'Memphis', 'Miami', 'Milwaukee', 'Minnesota', 'New Orleans', 'New York',
     'Oklahoma City', 'Orlando', 'Philadelphia', 'Phoenix', 'Portland',
     'Sacramento', 'San Antonio', 'Toronto', 'Utah', 'Washington']))
PROVENANCE_KEYS = ('url', 'final_url', 'status', 'memento_datetime', 'sha256',
                   'raw_path', 'retrieved_at_utc', 'publisher_url', 'publisher',
                   'content_type', 'content_encoding', 'last_modified')
TEXT_BLOCKS = {'paragraph', 'heading', 'list', 'table', 'quote'}
EMBED_BLOCKS = {'image', 'video', 'embed', 'nba/tickets', 'separator', 'spacer'}
MONTHS = {name.lower(): month for month in range(1, 13)
          for name in (datetime(2000, month, 1).strftime('%B'), datetime(2000, month, 1).strftime('%b'))}
EXPLICIT_DATE = re.compile(r'\b(' + '|'.join(MONTHS) + r')\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20\d{2})\b', re.I)


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _stamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Timestamp needs an explicit timezone')
    return result.astimezone(timezone.utc)


class _Scripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts, self.current = [], None

    def handle_starttag(self, tag, attrs):
        if tag == 'script':
            self.current = {'attrs': dict(attrs), 'parts': []}

    def handle_data(self, text):
        if self.current is not None:
            self.current['parts'].append(text)

    def handle_endtag(self, tag):
        if tag == 'script' and self.current is not None:
            self.scripts.append(self.current)
            self.current = None


class _ArticleText(HTMLParser):
    """Visible block text, with structural boundaries and no attribute values."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style', 'iframe', 'nav', 'noscript'}:
            self.hidden += 1
        if not self.hidden and tag in {'p', 'br', 'li', 'tr', 'blockquote', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}:
            self.parts.append('\n')
        if not self.hidden and tag in {'td', 'th'}:
            self.parts.append(' | ')

    def handle_endtag(self, tag):
        if tag in {'script', 'style', 'iframe', 'nav', 'noscript'}:
            self.hidden = max(0, self.hidden - 1)
        elif not self.hidden and tag in {'p', 'li', 'tr', 'blockquote'}:
            self.parts.append('\n')

    def handle_data(self, text):
        if not self.hidden:
            self.parts.append(text)


def _text(fragment):
    parser = _ArticleText()
    parser.feed(fragment)
    parser.close()
    return '\n'.join(value for line in ''.join(parser.parts).splitlines()
                     if (value := ' '.join(line.split()).strip(' |')))


def _extract(raw):
    parser = _Scripts()
    decoded = gzip.decompress(raw) if raw.startswith(b'\x1f\x8b') else raw
    parser.feed(decoded.decode('utf-8-sig'))
    parser.close()
    next_data = [s for s in parser.scripts if s['attrs'].get('id') == '__NEXT_DATA__']
    if len(next_data) != 1:
        raise ValueError('Expected exactly one __NEXT_DATA__ script')
    try:
        page = json.loads(''.join(next_data[0]['parts']))['props']['pageProps']['pageObject']
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Missing or malformed article pageObject') from exc
    if not isinstance(page, dict) or page.get('type') != 'article':
        raise ValueError('pageObject is not an article')
    schemas = []
    for script in parser.scripts:
        if script['attrs'].get('type') == 'application/ld+json':
            data = json.loads(''.join(script['parts']))
            nodes = data if isinstance(data, list) else data.get('@graph', [data])
            schemas.extend(n for n in nodes if isinstance(n, dict)
                           and n.get('@type') in ('NewsArticle', 'BlogPosting', 'Article'))
    if len(schemas) > 1:
        raise ValueError('Ambiguous article structured metadata')
    blocks = page.get('contentStructured')
    if not isinstance(blocks, list) or not blocks:
        raise ValueError('Article has no structured body')
    return page, schemas[0] if schemas else None, blocks


def _aliases(game, side):
    abbreviation = game.get(side + '_abbr')
    name = TEAM_NAMES.get(abbreviation)
    if not name:
        full = game.get(side + '_team', '')
        name = next((n for n in TEAM_NAMES.values() if full.lower().endswith(n.lower())), None)
    if not name:
        raise ValueError('Unknown ' + side + ' team identity')
    result = [name]
    if name == '76ers':
        result += ['Sixers']
    if name == 'Trail Blazers':
        result += ['Blazers']
    return result


def _mentions(text, aliases):
    return any(re.search(r'(?<!\w)' + re.escape(a) + r'(?!\w)', text, re.I) for a in aliases)


def _capture(provenance, publisher_url, cutoff, reasons):
    match = re.fullmatch(r'https://web\.archive\.org/web/(\d{14})(?:id_)?/(.+)', provenance.get('final_url', ''))
    if not match or match[2] != publisher_url or provenance.get('status') != 200:
        reasons.append('no_exact_independent_archive_capture')
        return None
    try:
        when = datetime.strptime(match[1], '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
        memento = parsedate_to_datetime(provenance['memento_datetime'])
        if memento.tzinfo is None or memento != when:
            raise ValueError('Memento does not match actual replay')
    except (ValueError, TypeError, KeyError):
        reasons.append('archive_memento_mismatch')
        return None
    if when > cutoff:
        reasons.append('archive_capture_after_cutoff')
    return when


def parse_html_note(raw, provenance, game):
    """Extract full article; return quarantined eligibility on timing/content issues.

    Structural parse failures and checksum mismatches raise ValueError. Inputs
    are never changed. Only allowlisted provenance/game metadata are returned.
    Eligibility is a mechanical source gate, not semantic approval.
    """
    raw = raw.encode('utf-8') if isinstance(raw, str) else raw
    sha = _hash(raw)
    if provenance.get('sha256') != sha:
        raise ValueError('Raw HTML checksum does not match provenance')
    page, schema, blocks = _extract(raw)
    title = _text(page.get('title') or '')
    if not title:
        raise ValueError('Missing article title')
    cutoff = _stamp(game['cutoff_utc']) if game.get('cutoff_utc') else _stamp(game['scheduled_start_utc']) - timedelta(minutes=60)
    if game.get('scheduled_start_utc') and cutoff != _stamp(game['scheduled_start_utc']) - timedelta(minutes=60):
        raise ValueError('Prediction cutoff must be scheduled tipoff minus 60 minutes')
    day = datetime.strptime(game['game_date_et'], '%Y-%m-%d')
    scheduled_et = (cutoff + timedelta(minutes=60)).astimezone(ZoneInfo('America/New_York'))
    if scheduled_et.date() != day.date():
        raise ValueError('Game date disagrees with scheduled tipoff in Eastern time')
    publisher_url = provenance.get('publisher_url', '')
    reasons, excluded, sections, body = [], [], [], []
    if urlsplit(publisher_url).hostname not in ('nba.com', 'www.nba.com') or not urlsplit(publisher_url).path:
        reasons.append('publisher_url_not_official_nba')
    if page.get('permalink') != publisher_url:
        reasons.append('article_url_mismatch')

    def append_section(label, text, block_index):
        sections.append({'section': len(sections) + 1, 'label': label, 'block_index': block_index,
                         'text': text, 'text_sha256': _hash(text.encode()),
                         'lines': [{'line': i + 1, 'text': line} for i, line in enumerate(text.splitlines(keepends=True))]})

    append_section('title', title, None)
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise ValueError('Invalid structured article block')
        kind = block.get('type')
        if kind in TEXT_BLOCKS:
            fragment = block.get('html')
            # Text-only structured blocks are allowed, but HTML is authoritative
            # when both fields exist. This avoids duplicating the same paragraph.
            text = _text(fragment) if isinstance(fragment, str) else _text(block.get('text') or '')
            if text:
                append_section(kind, text, index)
                body.append((kind, text))
        else:
            excluded.append({'block_index': index, 'type': kind, 'reason': 'non_article_embed' if kind in EMBED_BLOCKS else 'unsupported_block'})
            if kind not in EMBED_BLOCKS:
                reasons.append('unsupported_article_block')
    if not body:
        reasons.append('empty_article_body')
    opening = next((text for kind, text in body if kind == 'paragraph'), '')
    if not re.search(r'\bgame\s+notes\b', title, re.I):
        reasons.append('title_not_game_notes')
    for side in ('home', 'away'):
        aliases = _aliases(game, side)
        if not _mentions(title, aliases) or not _mentions(opening, aliases):
            reasons.append(side + '_team_missing_from_title_or_opening')
    date_pattern = rf'\b(?:{day:%B}|{day:%b}\.?)\s+0?{day.day}(?:st|nd|rd|th)?,?\s+{day.year}\b'
    if not re.search(date_pattern, title, re.I):
        reasons.append('game_date_missing_from_title')
    # Historical sections may contain other dates; only the title and opening
    # identify the upcoming game and must not contradict the scheduled date.
    for match in EXPLICIT_DATE.finditer(title + '\n' + opening):
        try:
            stated_day = datetime(int(match[3]), MONTHS[match[1].lower()], int(match[2])).date()
        except ValueError:
            reasons.append('invalid_game_date_in_title_or_opening')
            continue
        if stated_day != day.date():
            reasons.append('game_date_conflict_in_title_or_opening')
    for match in re.finditer(r'\b(\d{1,2})(?::(\d{2}))?\s*(AM|PM)\s*(?:ET|EST|EDT)\b', opening, re.I):
        hour, minute = int(match[1]), int(match[2] or 0)
        if not 1 <= hour <= 12 or not 0 <= minute <= 59:
            reasons.append('invalid_tipoff_time_in_opening')
        elif ((hour % 12) + (12 if match[3].upper() == 'PM' else 0), minute) != (scheduled_et.hour, scheduled_et.minute):
            reasons.append('tipoff_time_conflict_in_opening')
    publisher = provenance.get('publisher')
    if publisher in (game.get('home_abbr'), game.get('away_abbr')):
        other_side = 'away' if publisher == game.get('home_abbr') else 'home'
        other_abbr = game.get(other_side + '_abbr')
        opponent = _aliases(game, other_side) + ([other_abbr] if other_abbr else [])
        if other_abbr in HEADING_CITIES:
            opponent.append(HEADING_CITIES[other_abbr])
        for kind, text in body:
            match = re.match(r'Last\s+game\s+(?:played\s+)?(?:vs\.?|against)\s+(.+)', text, re.I)
            if match and not _mentions(match[1], opponent):
                reasons.append('last_meeting_heading_wrong_opponent')

    captured = _capture(provenance, publisher_url, cutoff, reasons)
    times = {}
    for field, value in [('published', page.get('date')), ('modified', page.get('modified'))]:
        try:
            times[field] = _stamp(value)
            if times[field] > cutoff:
                reasons.append(field + '_after_cutoff')
            if captured is not None and times[field] > captured:
                reasons.append(field + '_after_archive_capture')
        except (ValueError, TypeError, AttributeError):
            reasons.append('missing_or_invalid_' + field)
    if len(times) == 2 and times['modified'] < times['published']:
        reasons.append('modified_before_published')
    if schema is not None:
        if _text(schema.get('headline') or '') != title:
            reasons.append('title_metadata_disagrees')
        identity = schema.get('mainEntityOfPage')
        identity = identity.get('@id') if isinstance(identity, dict) else identity
        if identity != publisher_url:
            reasons.append('schema_url_mismatch')
        for field, key in [('published', 'datePublished'), ('modified', 'dateModified')]:
            try:
                if _stamp(schema.get(key)) != times.get(field):
                    reasons.append(field + '_metadata_disagrees')
            except (ValueError, TypeError, AttributeError):
                reasons.append('missing_or_invalid_schema_' + field)

    return {'source_kind': 'html', 'pagination': 'article_sections', 'source_id': 'sha256:' + sha,
            'sha256': sha, 'title': title, 'article_url': publisher_url, 'publisher': publisher,
            'article_date': page.get('date'), 'article_modified': page.get('modified'),
            'archive_time': captured.isoformat().replace('+00:00', 'Z') if captured else None,
            'cutoff_utc': cutoff.isoformat().replace('+00:00', 'Z'),
            'game': {k: game[k] for k in ('game_id', 'game_date_et', 'home_team', 'away_team', 'home_abbr', 'away_abbr', 'scheduled_start_utc', 'season') if k in game},
            'provenance': {k: provenance[k] for k in PROVENANCE_KEYS if k in provenance},
            'eligibility': {'status': 'quarantined' if reasons else 'verified_pregame',
                            'reasons': sorted(set(reasons)), 'requires_content_review': True},
            'sections': sections, 'excluded_blocks': excluded,
            'text_sha256': _hash('\n\n'.join(s['text'] for s in sections).encode()),
            'parser_version': 'nba_article_sections_v1'}
