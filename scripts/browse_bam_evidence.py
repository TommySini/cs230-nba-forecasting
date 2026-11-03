"""Read-only localhost BAM evidence viewer; no model calls, collection or writes."""
import argparse
import hashlib
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
STAGES = ('prepare-research', 'selection', 'modules', 'gold', 'export', 'research-train', 'review-student', 'forecast', 'evaluate')
TABS = {'sources': 'Sources', 'selection': 'Selection & tables', 'prompted': 'Prompted research',
        'postmortem': 'Training postmortem', 'gold': 'Gold targets', 'student': 'Actual student research',
        'forecast': 'Forecast input & predictions'}
GAME_FILES = {'selection': 'selection.json', 'prompted': 'prompted_modules.json',
              'postmortem': 'postmortem.json', 'gold': 'gold.json', 'student': 'student_modules.json'}
TRAIN_ONLY = {'postmortem', 'gold'}
STYLE = """
*{box-sizing:border-box}body{margin:0;background:#f3f6f6;color:#1d2935;font:16px/1.55 system-ui,sans-serif}
main{max-width:1200px;margin:auto;padding:26px 24px 70px}h1{font-size:28px;line-height:1.25}h2{font-size:21px}h3{font-size:17px}
a{color:#176653;text-underline-offset:3px}nav{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}
nav a,.badge{display:inline-block;padding:3px 10px;border:1px solid #ccd8d9;border-radius:6px;background:white}
nav a[aria-current=page]{background:#176653;color:white}.muted{color:#536575}.pending{background:#fff7dc;padding:12px;border-radius:6px}
.card{background:white;border:1px solid #d7e0e1;border-radius:8px;padding:18px;margin:15px 0;overflow-wrap:anywhere}
.badge.approved,.badge.complete{background:#e3f2eb}.badge.rejected,.badge.failed{background:#ffeded}.badge.mechanically_valid{background:#fff7dc}
.target{border-left:5px solid #a86b12;background:#fff7dc;padding:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.55 ui-monospace,monospace}
table{width:100%;border-collapse:collapse}th,td{text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid #dce3e8}
.source{overflow:auto}.source td.line{white-space:pre;font:13px/1.5 ui-monospace,monospace}.source td.number{width:65px;white-space:nowrap}
tr:target{background:#fff2b3}details{margin:10px 0}dt{font-weight:600}dd{margin:0 0 9px}blockquote{border-left:3px solid #c8d6d0;padding:5px 12px;margin:10px 0}
.stage-list{display:flex;flex-wrap:wrap;gap:8px}.links a{display:inline-block;margin:2px 8px 2px 0}
@media(max-width:650px){main{padding:18px 12px}.card{padding:12px}th,td{padding:6px}h1{font-size:23px}}
"""


def esc(value):
    return html.escape(str(value), quote=True)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def page(title, body):
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(title)}</title><style>{STYLE}</style></head><body><main>{body}</main></body></html>').encode()


def badge(status):
    allowed = {'approved': 'Approved', 'mechanically_valid': 'Mechanically valid — not semantic approval',
               'rejected': 'Rejected', 'unreviewed': 'Unreviewed', 'complete': 'Complete', 'failed': 'Failed',
               'running': 'Running', 'pending': 'Pending'}
    key = status if status in allowed else 'unreviewed'
    return f'<span class="badge {key}">{esc(allowed[key])}</span>'


class EvidenceViewer:
    def __init__(self, run_id, *, root=ROOT, snapshot=None):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,119}', run_id):
            raise ValueError('Invalid BAM run ID')
        self.repo = Path(root).resolve(); self.run_id = run_id
        self.run = self._contained(self.repo / 'data/processed/bam_runs', run_id)
        raw = self._read(self.run, 'protocol.json')
        self.protocol_sha = digest(raw); self.protocol = json.loads(raw)
        if self.protocol.get('run_id') != run_id: raise ValueError('Run protocol identity mismatch')
        snapshot_root = self.repo / 'data/processed/bam_pipeline'
        if snapshot is None:
            matches = [p.parent for p in snapshot_root.glob('*/manifest.json') if not p.is_symlink()
                       and p.resolve().is_relative_to(snapshot_root.resolve())
                       and digest(p.read_bytes()) == self.protocol['snapshot_manifest_sha256']]
            if len(matches) != 1: raise ValueError('Need exactly one matching snapshot; pass --snapshot explicitly')
            snapshot = matches[0]
        self.snapshot = Path(snapshot).resolve()
        if not self.snapshot.is_relative_to(snapshot_root.resolve()): raise ValueError('Snapshot must be within BAM snapshot namespace')
        manifest_raw = self._read(self.snapshot, 'manifest.json')
        if digest(manifest_raw) != self.protocol['snapshot_manifest_sha256']: raise ValueError('Snapshot manifest does not match run')
        self.manifest = json.loads(manifest_raw)
        rows = self._jsonl(self.snapshot, 'inputs.jsonl', self.manifest['artifact_sha256']['inputs.jsonl'])
        selected = self.protocol['game_ids']
        if len(set(selected)) != len(selected): raise ValueError('Duplicate protocol game ID')
        self.games = {}
        for row in rows:
            if row['game_id'] not in selected: continue
            if not re.fullmatch(r'[A-Za-z0-9:._-]+', row['game_id']) or row['split'] not in ('train', 'dev', 'test'):
                raise ValueError('Invalid game identity')
            key = digest(row['game_id'].encode())[:20]
            if key in self.games: raise ValueError('Duplicate game route')
            self.games[key] = row
        if {g['game_id'] for g in self.games.values()} != set(selected): raise ValueError('Protocol game absent from snapshot')

    @staticmethod
    def _contained(base, relative):
        base = Path(base).resolve(); path = Path(relative)
        if path.is_absolute() or '..' in path.parts or '\\' in str(path): raise ValueError('Unsafe artifact path')
        result = base / path
        cursor = base
        for part in path.parts:
            cursor = cursor / part
            if cursor.is_symlink(): raise ValueError('Artifact symlinks are not allowed')
        if not result.resolve().is_relative_to(base): raise ValueError('Artifact escapes its allowed namespace')
        return result

    def _read(self, base, relative, expected=None):
        raw = self._contained(base, relative).read_bytes()
        if expected is not None and digest(raw) != expected: raise ValueError('Artifact checksum mismatch: ' + str(relative))
        return raw

    def _json(self, base, relative, expected=None):
        return json.loads(self._read(base, relative, expected))

    def _jsonl(self, base, relative, expected=None):
        return [json.loads(line) for line in self._read(base, relative, expected).splitlines() if line.strip()]

    def _optional(self, relative):
        path = self._contained(self.run, relative)
        return self._json(self.run, relative) if path.is_file() else None

    def _game_file(self, game, name):
        return self._optional('games/' + game['game_id'].replace(':', '_') + '/' + name)

    def _game_rows(self, game, name):
        relative = 'training/' + name
        if not self._contained(self.run, relative).is_file(): return None
        return [row for row in self._jsonl(self.run, relative) if row.get('game_id') == game['game_id']] or None

    def stages(self):
        active = self._optional('status.json') or {}
        result = {}
        for stage in STAGES:
            marker = self._optional(f'stages/{stage}/complete.json')
            if marker and marker.get('protocol_sha256') == self.protocol_sha and marker.get('outputs'):
                result[stage] = 'complete'
            elif active.get('stage') == stage and active.get('status') in ('running', 'failed'):
                result[stage] = active['status']
            else: result[stage] = 'pending'
        return result, active

    def sources(self, game):
        result = []
        for source in game['sources']:
            if (not re.fullmatch(r'[a-f0-9]{64}', source.get('sha256', ''))
                    or source.get('pages_path') != 'sources/' + source['sha256'] + '.json'
                    or source.get('raw_path') != 'raw/' + source['sha256'] + '.bin'):
                raise ValueError('Source paths must be content-addressed snapshot assets')
            expected = source['pages_sha256']
            if self.manifest['artifact_sha256'].get(source['pages_path']) != expected:
                raise ValueError('Source pages are not referenced by the snapshot manifest')
            packet = self._json(self.snapshot, source['pages_path'], expected)
            if packet['source_sha256'] != source['sha256']: raise ValueError('Source/page association changed')
            for section in packet['pages']:
                if type(section.get('page')) is not int or section['page'] < 1:
                    raise ValueError('Invalid source section number')
                for line in section['lines']:
                    if type(line.get('line')) is not int or line['line'] < 1 or not isinstance(line.get('text'), str):
                        raise ValueError('Invalid numbered source line')
            result.append({**source, 'pages': packet['pages']})
        return result

    def evaluation_unlocked(self):
        """Never read a label file while checking the gate."""
        stages, _ = self.stages()
        if stages['evaluate'] != 'complete': return False
        required = ['evaluation_authorization.json', 'training/evaluation_started.json',
                    'training/evaluation_manifest.json', 'training/forecast/selection.json',
                    'training/test_predictions.jsonl', 'training/evaluation.json']
        if not all(self._contained(self.run, name).is_file() for name in required): return False
        authorization = self._json(self.run, required[0]); started = self._json(self.run, required[1])
        manifest = self._json(self.run, required[2])
        selection_sha = digest(self._read(self.run, required[3]))
        if (authorization.get('protocol_sha256') != self.protocol_sha
                or authorization.get('selection_sha256') != selection_sha or started.get('selection_sha256') != selection_sha):
            return False
        files = manifest.get('files', {})
        if not all(name in files for name in ('test_predictions.jsonl', 'evaluation.json', 'labels_test.jsonl')): return False
        prediction_sha = digest(self._read(self.run, required[4], files['test_predictions.jsonl']))
        evaluation = self._json(self.run, required[5], files['evaluation.json'])
        return (evaluation.get('selection_sha256') == selection_sha and evaluation.get('test_predictions_sha256') == prediction_sha
                and evaluation.get('test_labels_sha256') == files['labels_test.jsonl'])

    def tab_data(self, key, tab):
        game = self.games[key]
        if tab in TRAIN_ONLY and game['split'] != 'train': raise KeyError('Training targets are unavailable for heldout games')
        if tab == 'sources': return {'sources': self.sources(game)}
        if tab == 'selection':
            requests = self._game_file(game, 'selection_requests.json') or []
            tables = {}
            for request in requests:
                for message in request.get('messages', []):
                    if message.get('role') != 'user': continue
                    try: payload = json.loads(message['content'])
                    except (ValueError, KeyError): continue
                    for fact in payload.get('code_decoded_tables', []):
                        tables[fact.get('id', digest(json.dumps(fact, sort_keys=True).encode()))] = fact
            return {'selection': self._game_file(game, 'selection.json'), 'code_decoded_tables': list(tables.values())}
        if tab in ('prompted', 'postmortem'): return self._game_file(game, GAME_FILES[tab])
        if tab == 'gold':
            return {'gold_report': self._game_file(game, 'gold.json'),
                    'student_training_examples': self._game_rows(game, 'research_examples.jsonl'),
                    'forecast_explanation_targets': self._game_rows(game, 'forecast_explanations_train.jsonl')}
        if tab == 'student':
            return {'reviewed_modules': self._game_file(game, 'student_modules.json'),
                    'selected_generations': self._game_file(game, 'selected_student_generations.json'),
                    'initial_generations': self._game_rows(game, 'generated_modules.jsonl')}
        if tab == 'forecast':
            result = {'original_booster_probability': game.get('p_boost'),
                      'forecast_eligible': game.get('forecast_eligible'),
                      'forecast_ineligibility_reason': game.get('forecast_ineligibility_reason'),
                      'exact_prediction_inputs': self._game_rows(game, 'prediction_inputs.jsonl'),
                      'predictions': self._game_rows(game, 'test_predictions.jsonl'), 'evaluation_complete': self.evaluation_unlocked()}
            if result['evaluation_complete'] and game['split'] == 'test':
                manifest = self._json(self.run, 'training/evaluation_manifest.json')
                labels = self._jsonl(self.run, 'training/labels_test.jsonl', manifest['files']['labels_test.jsonl'])
                result['evaluated_outcome'] = next((r for r in labels if r['game_id'] == game['game_id']), None)
            return result
        raise KeyError(tab)

    def _citation(self, key, cite):
        game = self.games[key]
        for index, source in enumerate(game['sources']):
            if source['source_id'] != cite.get('source_id'): continue
            kind = 'Article section' if source.get('source_kind') == 'html' or source.get('pagination') == 'article_sections' else 'PDF page'
            page_id = cite.get('page'); lines = cite.get('lines', [])
            if type(page_id) is not int or any(type(n) is not int for n in lines): break
            target = f'/source/{key}/{index}#p{page_id}'
            links = ', '.join(f'<a href="{target}-l{n}">{n}</a>' for n in lines)
            return f'<p class="muted">{esc(source.get("publisher", "Source"))} · {kind} {page_id} · lines {links}</p>'
        return '<p class="muted">Source citation: ' + esc(json.dumps(cite, ensure_ascii=False)) + '</p>'

    def _render(self, value, key):
        if value is None: return '<p class="pending">Pending — no artifact has been saved for this stage.</p>'
        if isinstance(value, list):
            if not value: return '<p class="muted">This saved field contains no items.</p>'
            return ''.join(self._render(item, key) for item in value)
        if not isinstance(value, dict): return '<p>' + esc(value) + '</p>'
        if 'candidate' in value and isinstance(value['candidate'], dict):
            return '<section class="card">' + badge('rejected') + self._render(value['candidate'], key) + '<p>' + esc(value.get('errors', value.get('reason', ''))) + '</p></section>'
        if 'fact' in value or 'observation' in value:
            body = '<h3>' + esc(value.get('fact', value.get('observation'))) + '</h3><dl>'
            for field in ('interpretation', 'counterevidence', 'uncertainty', 'time_scope', 'fact_date', 'foreseeable_signal', 'unforeseeable', 'booster_overlap'):
                if value.get(field) is not None: body += '<dt>' + esc(field.replace('_', ' ').capitalize()) + '</dt><dd>' + esc(value[field]) + '</dd>'
            body += '</dl>' + ''.join(self._citation(key, cite) for cite in value.get('citations', []))
            if value.get('parsed_values'):
                body += '<table><tbody>' + ''.join('<tr><th>' + esc(k.replace('_', ' ')) + '</th><td>' + esc(v) + '</td></tr>' for k, v in value['parsed_values'].items()) + '</tbody></table>'
            return '<article class="card">' + body + '</article>'
        if 'output' in value:
            body = badge(value.get('status')) + '<p class="muted">' + esc(value.get('stage', 'Artifact')) + ' · ' + esc(value.get('module') or '') + '</p>'
            if value.get('model'): body += '<p class="muted">' + esc(value['model']) + '</p>'
            body += self._render(value['output'], key)
            if value.get('errors'): body += '<p>' + esc('; '.join(map(str, value['errors']))) + '</p>'
            if value.get('raw_response'): body += '<details><summary>Original model response</summary><pre>' + esc(value['raw_response']) + '</pre></details>'
            return '<section class="card">' + body + '</section>'
        if 'report' in value:
            provenance = value.get('research_provenance', {})
            body = badge('unreviewed') + '<p>' + esc(value.get('module', 'Student report')) + ' · ' + esc(provenance.get('mode', 'Unknown lineage')) + '</p>'
            body += '<p class="muted">Trained through: ' + esc(provenance.get('training_max_date') or 'No training: cold prompted fold') + '</p>'
            try: body += self._render(json.loads(value['report']), key)
            except (ValueError, TypeError): body += '<pre>' + esc(value['report']) + '</pre>'
            return '<section class="card">' + body + '</section>'
        body = ''
        for field, item in value.items():
            if field == 'citations': body += ''.join(self._citation(key, cite) for cite in item); continue
            if field in ('findings', 'modules', 'general', 'unknowns', 'timeline', 'rejected_findings'):
                body += '<h3>' + esc(field.replace('_', ' ').capitalize()) + '</h3>' + self._render(item, key)
            elif isinstance(item, (dict, list)):
                body += '<details><summary>' + esc(field.replace('_', ' ').capitalize()) + '</summary><pre>' + esc(json.dumps(item, indent=2, ensure_ascii=False)) + '</pre></details>'
            else: body += '<p><strong>' + esc(field.replace('_', ' ').capitalize()) + ':</strong> ' + esc(item) + '</p>'
        return body

    def index(self):
        stages, active = self.stages()
        body = f'<p class="muted">BAM evidence · {esc(self.run_id)}</p><h1>Research, sources and forecasts</h1>'
        body += '<p>Last recorded update: ' + esc(active.get('stage', 'No stage started')) + ' · ' + esc(active.get('status', 'pending')) + '</p>'
        if active.get('error'): body += '<p class="pending">' + esc(active['error']) + '</p>'
        body += '<div class="stage-list">' + ''.join('<span>' + esc(stage) + ' ' + badge(status) + '</span>' for stage, status in stages.items()) + '</div>'
        body += '<p class="muted">Training postmortems and gold targets are separate from forecast inputs. Missing stages remain pending.</p><table><thead><tr><th>Date</th><th>Game</th><th>Split</th><th>Evidence</th></tr></thead><tbody>'
        for key, game in self.games.items():
            body += '<tr><td>' + esc(game['game_date_et']) + '</td><td>' + esc(game['away_abbr'] + ' at ' + game['home_abbr']) + '</td><td>' + esc(game['split']) + '</td><td class="links">'
            body += ''.join(f'<a href="/game/{key}/{tab}">{esc(TABS[tab])}</a>' for tab in ('sources', 'selection', 'student', 'forecast')) + '</td></tr>'
        return page('BAM evidence · ' + self.run_id, body + '</tbody></table>')

    def document(self, key, tab):
        game = self.games[key]; data = self.tab_data(key, tab)
        title = f"{game['away_abbr']} at {game['home_abbr']} · {game['game_date_et']}"
        body = '<a href="/">← Run status and games</a><h1>' + esc(title) + '</h1><p>' + esc(game['split']) + ' · ' + esc(game['game_id']) + '</p><nav>'
        for name, label in TABS.items():
            if name in TRAIN_ONLY and game['split'] != 'train': continue
            body += f'<a href="/game/{key}/{name}"' + (' aria-current="page"' if name == tab else '') + '>' + esc(label) + '</a>'
        body += '</nav><h2>' + esc(TABS[tab]) + f'</h2><p><a href="/raw/{key}/{tab}">Download this game’s JSON</a></p>'
        if tab in TRAIN_ONLY:
            body += '<p class="target"><strong>TRAINING TARGET — never a forecast input.</strong> This material may use this training game’s result.</p>'
        if tab == 'sources':
            for index, source in enumerate(data['sources']):
                kind = 'HTML article sections' if source.get('source_kind') == 'html' or source.get('pagination') == 'article_sections' else 'PDF pages'
                body += f'<section class="card"><h3>{esc(source.get("publisher", "Source"))} · {kind}</h3><p>{len(source["pages"])} numbered sections/pages · archive {esc(source.get("archive_time"))}</p>'
                body += '<p>' + esc('Active for research' if source.get('active_for_research', True) else 'Retained historical version; not active research input') + '</p>'
                body += f'<a href="/source/{key}/{index}">Read full numbered source</a> · <a href="/source-raw/{key}/{index}">Download original archived bytes</a>'
                body += '<details><summary>Approval scope and provenance</summary><pre>' + esc(json.dumps({k: v for k, v in source.items() if k != 'pages'}, indent=2)) + '</pre></details></section>'
        elif tab == 'selection':
            body += '<h3>Saved source selection</h3>' + self._render(data['selection'], key)
            body += '<h3>Code-decoded table facts from the saved requests</h3>' + self._render(data['code_decoded_tables'], key)
        elif tab == 'student':
            body += '<h3>Independently reviewed actual student modules</h3>' + self._render(data['reviewed_modules'], key)
            body += '<h3>Original out-of-fold / final-adapter generations</h3><p class="muted">Raw generations are unreviewed here; approval is recorded in the modules above. Prompted modules and gold are never substituted.</p>' + self._render(data['initial_generations'], key)
            if data['selected_generations'] is not None:
                body += '<details><summary>Selected generation and repair lineage</summary><pre>' + esc(json.dumps(data['selected_generations'], indent=2)) + '</pre></details>'
        elif tab == 'gold':
            body += self._render(data['gold_report'], key)
            body += '<h3>Exported student completion targets</h3>'
            if not data['student_training_examples']: body += self._render(None, key)
            for example in data['student_training_examples'] or []:
                body += '<section class="card"><h3>' + esc(example.get('module', 'Module')) + '</h3><p>' + esc(example.get('review_status', 'unreviewed')) + '</p>'
                for message in example.get('messages', []):
                    if message.get('role') == 'assistant':
                        try: body += self._render(json.loads(message['content']), key)
                        except (ValueError, TypeError): body += '<pre>' + esc(message['content']) + '</pre>'
                body += '</section>'
            body += '<h3>Training-only forecast explanation targets</h3>' + self._render(data['forecast_explanation_targets'], key)
        elif tab == 'forecast':
            body += '<p>Original boosting probability: ' + esc(data['original_booster_probability']) + '</p>'
            if data['forecast_eligible'] is False: body += '<p class="pending">Research-only game: ' + esc(data['forecast_ineligibility_reason']) + '</p>'
            body += '<h3>Exact forecasting messages</h3>'
            if not data['exact_prediction_inputs']: body += self._render(None, key)
            for row in data['exact_prediction_inputs'] or []:
                for message in row.get('messages', []): body += '<section class="card"><h3>' + esc(message['role']) + '</h3><pre>' + esc(message['content']) + '</pre></section>'
            body += '<h3>Saved predictions</h3>' + self._render(data['predictions'] or None, key)
            body += '<p>' + ('Evaluation complete and prediction/selection locks verified.' if data['evaluation_complete'] else 'Evaluation pending or not unlocked; heldout outcome files remain unopened.') + '</p>'
            if 'evaluated_outcome' in data: body += '<h3>Evaluated outcome</h3>' + self._render(data['evaluated_outcome'], key)
        else: body += self._render(data, key)
        return page(TABS[tab] + ' · ' + title, body)

    def source_document(self, key, index):
        source = self.sources(self.games[key])[index]
        article = source.get('source_kind') == 'html' or source.get('pagination') == 'article_sections'
        label = 'Article section' if article else 'PDF page'
        body = f'<a href="/game/{key}/sources">← Game sources</a><h1>{esc(source.get("publisher", "Source"))}: full numbered source</h1>'
        body += '<p>These are article sections, not PDF pages.</p>' if article else '<p>Original extracted PDF page text; numbering matches citations.</p>'
        for section in source['pages']:
            number = section['page']; body += f'<section class="card source" id="p{number}"><h2>{label} {number} {esc(section.get("label", ""))}</h2><table><tbody>'
            for line in section['lines']:
                lineno = line['line']; body += f'<tr id="p{number}-l{lineno}"><td class="number"><a href="#p{number}-l{lineno}">L{lineno}</a></td><td class="line">{esc(line["text"])}</td></tr>'
            body += '</tbody></table></section>'
        return page('Numbered source', body)

    def dispatch(self, path):
        """Opaque fixed routes only; URL components never become filesystem paths."""
        if path == '/': return self.index(), 'text/html; charset=utf-8', None
        parts = path.strip('/').split('/')
        if len(parts) != 3 or parts[1] not in self.games: raise KeyError(path)
        route, key, name = parts
        if route in ('game', 'raw') and name in TABS:
            if route == 'game': return self.document(key, name), 'text/html; charset=utf-8', None
            data = self.tab_data(key, name)
            return (json.dumps(data, indent=2, ensure_ascii=False) + '\n').encode(), 'application/json; charset=utf-8', key + '-' + name + '.json'
        if route in ('source', 'source-raw') and name.isascii() and name.isdigit():
            index = int(name); sources = self.games[key]['sources']
            if index >= len(sources): raise KeyError(path)
            if route == 'source': return self.source_document(key, index), 'text/html; charset=utf-8', None
            source = sources[index]
            if (not re.fullmatch(r'[a-f0-9]{64}', source.get('sha256', ''))
                    or source.get('raw_path') != 'raw/' + source['sha256'] + '.bin'):
                raise ValueError('Raw source is not a content-addressed asset')
            if self.manifest['artifact_sha256'].get(source['raw_path']) != source['sha256']:
                raise ValueError('Raw source is not whitelisted by snapshot')
            raw = self._read(self.snapshot, source['raw_path'], source['sha256'])
            pdf = raw.startswith(b'%PDF')
            return raw, 'application/pdf' if pdf else 'application/octet-stream', key + '-source-' + name + ('.pdf' if pdf else '.bin')
        raise KeyError(path)


def handler_for(viewer):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try: content, mime, filename = viewer.dispatch(urlsplit(self.path).path)
            except KeyError: self.send_error(404); return
            except (ValueError, OSError, IndexError):
                self.send_error(409, 'Evidence unavailable or integrity check failed'); return
            self.send_response(200)
            self.send_header('Content-Type', mime); self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store'); self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
            if filename: self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
            self.end_headers(); self.wfile.write(content)
    return Handler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535: parser.error('Use a nonprivileged local port')
    viewer = EvidenceViewer(args.run_id, snapshot=args.snapshot)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(viewer))
    print(f'BAM evidence viewer: http://127.0.0.1:{args.port} ({len(viewer.games)} games)', flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
