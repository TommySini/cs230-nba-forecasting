import copy
from email.utils import format_datetime
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from bam_html_notes import parse_html_note


URL = 'https://www.nba.com/knicks/news/knicks-vs-clippers-game-notes-march-7-2025'
GAME = {'game_id': 'fixture:nyk-lac', 'game_date_et': '2025-03-07',
        'home_abbr': 'LAC', 'away_abbr': 'NYK', 'home_team': 'LA Clippers',
        'away_team': 'New York Knicks', 'scheduled_start_utc': '2025-03-08T03:30:00Z',
        'cutoff_utc': '2025-03-08T02:30:00Z', 'season': 2024,
        'outcome': 'DO NOT COPY LABEL'}
PAGE = {'type': 'article', 'title': 'Knicks vs Clippers Game Notes: March 7, 2025',
        'permalink': URL, 'date': '2025-03-07T17:24:07Z', 'modified': '2025-03-07T17:24:07Z',
        'contentStructured': [
            {'type': 'paragraph', 'html': '<p>The Knicks will tip off against the LA Clippers at 10:30PM ET.</p>'},
            {'type': 'heading', 'html': '<h5>Last game played vs LA Clippers:</h5>'},
            {'type': 'paragraph', 'html': '<p>A historical meeting recap, <a href="https://nba.com/future-final-score">with link text</a>.</p>'},
            {'type': 'list', 'html': '<ul><li>First point</li><li>Second point</li></ul>'}]}


def fixture(page=None, schema=True, capture='20250307200000', compressed=False):
    page = copy.deepcopy(PAGE if page is None else page)
    hydration = {'props': {'pageProps': {'pageObject': page,
                    'relatedNews': {'text': 'UNRELATED CURRENT NEWS'}}}}
    text = '<nav>FUTURE SCOREBOARD</nav><script id="__NEXT_DATA__" type="application/json">' + json.dumps(hydration) + '</script>'
    if schema:
        text += '<script type="application/ld+json">' + json.dumps({
            '@type': 'BlogPosting', 'headline': page['title'], 'mainEntityOfPage': {'@id': page['permalink']},
            'datePublished': page['date'], 'dateModified': page['modified']}) + '</script>'
    raw = text.encode()
    if compressed:
        raw = gzip.compress(raw, mtime=0)
    when = datetime.strptime(capture, '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
    replay = 'https://web.archive.org/web/' + capture + 'id_/' + URL
    provenance = {'url': replay, 'final_url': replay, 'status': 200,
                  'memento_datetime': format_datetime(when, usegmt=True),
                  'sha256': hashlib.sha256(raw).hexdigest(), 'raw_path': 'fixture/article.bin',
                  'publisher': 'NYK', 'publisher_url': URL, 'retrieved_at_utc': '2026-11-01T10:00:00Z',
                  'outcome': 'DO NOT COPY LABEL'}
    return raw, provenance


class HtmlNotes(unittest.TestCase):
    def parse(self, page=None, **kwargs):
        raw, source = fixture(page, **kwargs)
        return parse_html_note(raw, source, GAME)

    def test_full_article_sections_exclude_page_chrome_urls_and_outcomes(self):
        result = self.parse()
        self.assertEqual(result['eligibility']['status'], 'verified_pregame')
        self.assertTrue(result['eligibility']['requires_content_review'])
        self.assertEqual(result['source_kind'], 'html')
        self.assertEqual(result['pagination'], 'article_sections')
        self.assertNotIn('pages', result)
        text = json.dumps(result)
        for absent in ('CURRENT NEWS', 'FUTURE SCOREBOARD', 'future-final-score', 'DO NOT COPY LABEL'):
            self.assertNotIn(absent, text)
        self.assertIn('with link text', text)
        self.assertEqual([s['section'] for s in result['sections']], [1, 2, 3, 4, 5])
        for section in result['sections']:
            self.assertEqual(''.join(line['text'] for line in section['lines']), section['text'])
            self.assertEqual(hashlib.sha256(section['text'].encode()).hexdigest(), section['text_sha256'])
        self.assertEqual(result['sections'][-1]['text'], 'First point\nSecond point')

    def test_long_article_and_table_are_not_truncated(self):
        page = copy.deepcopy(PAGE)
        huge = ('Important source text. ' * 10000) + 'END OF ARTICLE'
        page['contentStructured'] += [{'type': 'paragraph', 'html': '<p>' + huge + '</p>'},
                                     {'type': 'table', 'html': '<table><tr><th>Team</th><th>PTS</th></tr><tr><td>Knicks</td><td>100</td></tr></table>'}]
        result = self.parse(page)
        self.assertEqual(result['sections'][-2]['text'], huge)
        self.assertEqual(result['sections'][-1]['text'], 'Team | PTS\nKnicks | 100')

    def test_raw_gzip_hash_and_original_provenance_preserved(self):
        raw, source = fixture(compressed=True)
        saved = copy.deepcopy(source)
        result = parse_html_note(raw, source, GAME)
        self.assertEqual(source, saved)
        self.assertEqual(result['sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(result['provenance']['raw_path'], source['raw_path'])
        self.assertEqual(result['eligibility']['status'], 'verified_pregame')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            parse_html_note(raw + b'x', source, GAME)

    def test_cms_timestamps_without_archive_proof_are_quarantined(self):
        raw, source = fixture()
        source.update(url=URL, final_url=URL, memento_datetime=None)
        result = parse_html_note(raw, source, GAME)
        self.assertEqual(result['eligibility']['status'], 'quarantined')
        self.assertIn('no_exact_independent_archive_capture', result['eligibility']['reasons'])
        self.assertEqual(len(result['sections']), 5)

    def test_final_replay_memento_and_cutoff_must_all_match(self):
        for mutation, expected in [
            ({'final_url': 'https://web.archive.org/web/20250307200000id_/https://nba.com/other'}, 'no_exact_independent_archive_capture'),
            ({'memento_datetime': 'Fri, 07 Mar 2025 21:00:00 GMT'}, 'archive_memento_mismatch'),
            ({'status': 404}, 'no_exact_independent_archive_capture')]:
            raw, source = fixture(); source.update(mutation)
            self.assertIn(expected, parse_html_note(raw, source, GAME)['eligibility']['reasons'])
        result = self.parse(capture='20250308030000')
        self.assertIn('archive_capture_after_cutoff', result['eligibility']['reasons'])

    def test_modified_after_cutoff_never_accepted_even_with_earlier_capture(self):
        page = copy.deepcopy(PAGE); page['modified'] = '2025-03-31T14:45:10Z'
        reasons = self.parse(page)['eligibility']['reasons']
        self.assertIn('modified_after_cutoff', reasons)
        self.assertIn('modified_after_archive_capture', reasons)

    def test_timestamp_timezone_and_internal_order_required(self):
        for modified, expected in [('2025-03-07T17:24:07', 'missing_or_invalid_modified'),
                                   ('2025-03-06T17:24:07Z', 'modified_before_published')]:
            page = copy.deepcopy(PAGE); page['modified'] = modified
            self.assertIn(expected, self.parse(page)['eligibility']['reasons'])

    def test_title_alone_does_not_prove_body_opponent(self):
        page = copy.deepcopy(PAGE)
        page['contentStructured'][0]['html'] = '<p>The Knicks face the Lakers tonight.</p>'
        self.assertIn('home_team_missing_from_title_or_opening', self.parse(page)['eligibility']['reasons'])
        page = copy.deepcopy(PAGE)
        page['contentStructured'][1]['html'] = '<h5>Last game played vs Detroit Pistons:</h5>'
        self.assertIn('last_meeting_heading_wrong_opponent', self.parse(page)['eligibility']['reasons'])
        page['contentStructured'][1]['html'] = '<h5>Last game vs DET:</h5>'
        self.assertIn('last_meeting_heading_wrong_opponent', self.parse(page)['eligibility']['reasons'])
        page['contentStructured'][1]['html'] = '<h5>Last game vs LAC:</h5>'
        self.assertEqual(self.parse(page)['eligibility']['status'], 'verified_pregame')

    def test_exact_game_date_and_url_required(self):
        page = copy.deepcopy(PAGE); page['title'] = page['title'].replace('March 7', 'March 8')
        self.assertIn('game_date_missing_from_title', self.parse(page)['eligibility']['reasons'])
        page = copy.deepcopy(PAGE); page['permalink'] = 'https://www.nba.com/knicks/news/other'
        self.assertIn('article_url_mismatch', self.parse(page)['eligibility']['reasons'])

    def test_last_meeting_heading_accepts_explicit_team_code_but_not_ambiguous_city(self):
        page = copy.deepcopy(PAGE)
        page['contentStructured'][1]['html'] = '<h5>Last game played vs LAC:</h5>'
        self.assertEqual(self.parse(page)['eligibility']['status'], 'verified_pregame')
        page['contentStructured'][1]['html'] = '<h5>Last game played vs Los Angeles:</h5>'
        self.assertIn('last_meeting_heading_wrong_opponent', self.parse(page)['eligibility']['reasons'])
        page = copy.deepcopy(PAGE)
        for key in ('title',):
            page[key] = page[key].replace('Clippers', 'Heat')
        for block in page['contentStructured']:
            block['html'] = block['html'].replace('LA Clippers', 'Miami Heat')
        page['contentStructured'][1]['html'] = '<h5>Last game played vs Miami:</h5>'
        raw, source = fixture(page)
        game = {**GAME, 'home_abbr': 'MIA', 'home_team': 'Miami Heat'}
        self.assertEqual(parse_html_note(raw, source, game)['eligibility']['status'], 'verified_pregame')

    def test_schema_disagreement_and_duplicate_hydration_rejected(self):
        raw, source = fixture()
        # Alter only the schema headline, preserving the pageObject title.
        raw = raw.replace(b'"headline": "Knicks', b'"headline": "Lakers')
        source['sha256'] = hashlib.sha256(raw).hexdigest()
        self.assertIn('title_metadata_disagrees', parse_html_note(raw, source, GAME)['eligibility']['reasons'])
        raw, source = fixture(); raw += raw; source['sha256'] = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            parse_html_note(raw, source, GAME)

    def test_unknown_blocks_quarantine_and_embeds_explicitly_recorded(self):
        page = copy.deepcopy(PAGE)
        page['contentStructured'] += [{'type': 'nba/tickets', 'attributes': {'future': 'FUTURE'}},
                                     {'type': 'unrecognized-text', 'text': 'Do not silently omit.'}]
        result = self.parse(page)
        self.assertIn('unsupported_article_block', result['eligibility']['reasons'])
        self.assertEqual(len(result['excluded_blocks']), 2)
        self.assertNotIn('FUTURE', json.dumps(result))

    def test_schema_optional_but_never_general_html_fallback(self):
        self.assertEqual(self.parse(schema=False)['eligibility']['status'], 'verified_pregame')
        raw = b'<article><h1>Game Notes</h1><p>Knicks Clippers</p></article>'
        with self.assertRaisesRegex(ValueError, '__NEXT_DATA__'):
            parse_html_note(raw, {'sha256': hashlib.sha256(raw).hexdigest()}, GAME)

    def test_cutoff_not_allowed_to_drift_after_tipoff(self):
        raw, source = fixture(); game = {**GAME, 'cutoff_utc': GAME['scheduled_start_utc']}
        with self.assertRaisesRegex(ValueError, '60 minutes'):
            parse_html_note(raw, source, game)

    def test_opening_date_or_tipoff_cannot_contradict_matching_title(self):
        page = copy.deepcopy(PAGE)
        page['contentStructured'][0]['html'] += '<p>March 8, 2025.</p>'
        self.assertIn('game_date_conflict_in_title_or_opening', self.parse(page)['eligibility']['reasons'])
        page = copy.deepcopy(PAGE)
        page['contentStructured'][0]['html'] = page['contentStructured'][0]['html'].replace('10:30PM', '7:30PM')
        self.assertIn('tipoff_time_conflict_in_opening', self.parse(page)['eligibility']['reasons'])
        raw, source = fixture()
        with self.assertRaisesRegex(ValueError, 'Eastern time'):
            parse_html_note(raw, source, {**GAME, 'game_date_et': '2025-03-08'})


if __name__ == '__main__':
    unittest.main()
