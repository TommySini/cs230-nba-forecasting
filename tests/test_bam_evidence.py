import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from browse_bam_evidence import EvidenceViewer, TABS


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = data if isinstance(data, bytes) else (json.dumps(data, sort_keys=True) + '\n').encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def jsonl(path, rows):
    return write(path, ''.join(json.dumps(row) + '\n' for row in rows).encode())


class BamEvidence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / 'data/processed/bam_runs/fixture'
        self.snapshot = self.root / 'data/processed/bam_pipeline/snapshot'
        self.manifest = {'artifact_sha256': {}}
        self.rows = []
        for index, (split, kind) in enumerate((('train', 'pdf'), ('test', 'html'))):
            raw = b'%PDF source bytes' if kind == 'pdf' else b'<html>Archived article</html>'
            sha = hashlib.sha256(raw).hexdigest(); raw_path = f'raw/{sha}.bin'; pages_path = f'sources/{sha}.json'
            self.manifest['artifact_sha256'][raw_path] = write(self.snapshot / raw_path, raw)
            packet = {'source_sha256': sha, 'pages': [{'page': 1, 'label': 'Article title' if kind == 'html' else '',
                       'lines': [{'line': 1, 'text': '<script>unsafe()</script> Source heading\n'}, {'line': 2, 'text': 'Player doubtful.'}]}]}
            pages_sha = write(self.snapshot / pages_path, packet)
            self.manifest['artifact_sha256'][pages_path] = pages_sha
            self.rows.append({'game_id': f'espn:{index}', 'game_date_et': '2025-01-01', 'split': split,
                              'home_abbr': 'BOS', 'away_abbr': 'NYK', 'p_boost': .6, 'forecast_eligible': True,
                              'sources': [{'source_id': 'sha256:' + sha, 'sha256': sha, 'publisher': 'NYK', 'source_kind': kind,
                                           'pagination': 'article_sections' if kind == 'html' else 'pdf_pages',
                                           'raw_path': raw_path, 'pages_path': pages_path, 'pages_sha256': pages_sha,
                                           'archive_time': '2025-01-01T12:00:00Z', 'active_for_research': True}]})
        self.refresh()
        write(self.run / 'status.json', {'stage': 'prepare-research', 'status': 'complete'})
        self.viewer = EvidenceViewer('fixture', root=self.root)
        self.train, self.test = list(self.viewer.games)

    def refresh(self):
        self.manifest['artifact_sha256']['inputs.jsonl'] = jsonl(self.snapshot / 'inputs.jsonl', self.rows)
        self.manifest['artifact_sha256']['labels_test.jsonl'] = jsonl(self.snapshot / 'labels_test.jsonl', [{'game_id': 'espn:1', 'home_win': 1}])
        manifest_sha = write(self.snapshot / 'manifest.json', self.manifest)
        self.protocol = {'run_id': 'fixture', 'game_ids': [r['game_id'] for r in self.rows], 'snapshot_manifest_sha256': manifest_sha}
        self.protocol_sha = write(self.run / 'protocol.json', self.protocol)

    def game_file(self, gid, name, data):
        return write(self.run / 'games' / gid.replace(':', '_') / name, data)

    def forbid_labels(self, original):
        def guarded(viewer, base, relative, expected=None):
            if Path(relative).name.startswith('labels'):
                self.fail('Viewer attempted an early label read: ' + str(relative))
            return original(viewer, base, relative, expected)
        return guarded

    def test_pending_stages_never_imply_completed_pipeline_and_no_early_labels(self):
        jsonl(self.run / 'training/labels_test.jsonl', [{'game_id': 'espn:1', 'home_win': 1}])
        with patch.object(EvidenceViewer, '_read', self.forbid_labels(EvidenceViewer._read)):
            viewer = EvidenceViewer('fixture', root=self.root)
            self.assertIn(b'Pending', viewer.index())
            for key, game in viewer.games.items():
                for tab in TABS:
                    if game['split'] != 'train' and tab in ('gold', 'postmortem'): continue
                    content, _, _ = viewer.dispatch(f'/game/{key}/{tab}')
                    self.assertNotIn(b'home_win', content)
            self.assertIn(b'Pending', viewer.document(self.train, 'prompted'))
            self.assertIn(b'heldout outcome files remain unopened', viewer.document(self.test, 'forecast'))

    def test_train_targets_not_routed_or_read_for_heldout_games(self):
        for name in ('gold.json', 'postmortem.json'):
            self.game_file('espn:1', name, {'secret': 'HELDOUT RESULT'})
        for route in ('raw', 'game'):
            for tab in ('gold', 'postmortem'):
                with self.assertRaises(KeyError): self.viewer.dispatch(f'/{route}/{self.test}/{tab}')
        self.game_file('espn:0', 'gold.json', {'status': 'approved', 'stage': 'gold', 'output': {'modules': []}})
        document = self.viewer.document(self.train, 'gold')
        self.assertIn(b'TRAINING TARGET', document); self.assertIn(b'never a forecast input', document)

    def test_full_source_text_and_html_sections_are_escaped_and_downloads_whitelisted(self):
        html_doc = self.viewer.dispatch(f'/source/{self.test}/0')[0]
        pdf_doc = self.viewer.dispatch(f'/source/{self.train}/0')[0]
        self.assertIn(b'Article section 1', html_doc); self.assertIn(b'not PDF pages', html_doc)
        self.assertIn(b'PDF page 1', pdf_doc)
        self.assertIn(b'&lt;script&gt;unsafe()', html_doc); self.assertNotIn(b'<script>', html_doc)
        self.assertIn(b'id="p1-l2"', html_doc)
        raw, mime, name = self.viewer.dispatch(f'/source-raw/{self.train}/0')
        self.assertEqual(raw, b'%PDF source bytes'); self.assertEqual(mime, 'application/pdf')
        self.assertTrue(name.endswith('.pdf'))

    def test_human_facts_status_and_numbered_citation_links(self):
        fact = {'fact': 'Player doubtful.', 'interpretation': 'Rotation uncertain.', 'counterevidence': 'No confirmed absence.',
                'uncertainty': 'Status may change.', 'citations': [{'source_id': self.rows[0]['sources'][0]['source_id'], 'page': 1, 'lines': [1, 2]}]}
        self.game_file('espn:0', 'selection.json', [{'stage': 'selection', 'status': 'mechanically_valid', 'output': {'findings': [fact]}}])
        doc = self.viewer.document(self.train, 'selection')
        for value in (b'Mechanically valid', b'not semantic approval', b'Rotation uncertain.', b'Status may change.', b'#p1-l2'):
            self.assertIn(value, doc)
        self.assertNotIn(b'class="badge approved"', doc)

    def test_actual_student_and_exact_forecast_input_are_distinct_from_gold(self):
        self.game_file('espn:0', 'gold.json', {'output': {'fact': 'SECRET GOLD TARGET'}})
        jsonl(self.run / 'training/generated_modules.jsonl', [{'game_id': 'espn:0', 'module': 'matchup',
               'report': '{"findings":[{"fact":"Actual OOF report"}]}',
               'research_provenance': {'mode': 'expanding_fold', 'training_max_date': '2024-12-01'}}])
        jsonl(self.run / 'training/prediction_inputs.jsonl', [{'game_id': 'espn:0', 'messages': [
            {'role': 'system', 'content': 'Exact system'}, {'role': 'user', 'content': 'Actual student input <unsafe>'}]}])
        student = self.viewer.document(self.train, 'student'); forecast = self.viewer.document(self.train, 'forecast')
        self.assertIn(b'Actual OOF report', student); self.assertIn(b'Unreviewed', student)
        self.assertIn(b'Actual student input &lt;unsafe&gt;', forecast)
        self.assertNotIn(b'SECRET GOLD TARGET', forecast)

    def test_unknown_routes_encoded_traversal_and_symlinks_fail(self):
        for path in ('/raw/../../secret', f'/raw/{self.train}/../protocol.json', f'/raw/{self.train}/%2e%2e',
                     f'/source/{self.train}/-1', f'/source-raw/{self.train}/999', '/game/unknown/gold'):
            with self.assertRaises(KeyError): self.viewer.dispatch(path)
        outside = self.root / 'secret'; outside.write_text('secret')
        source = self.rows[0]['sources'][0]; path = self.snapshot / source['raw_path']
        path.unlink(); path.symlink_to(outside)
        with self.assertRaises(ValueError): self.viewer.dispatch(f'/source-raw/{self.train}/0')

    def test_source_cannot_alias_a_label_file_even_if_manifest_references_it(self):
        self.rows[0]['sources'][0]['pages_path'] = 'labels_test.jsonl'
        self.rows[0]['sources'][0]['pages_sha256'] = self.manifest['artifact_sha256']['labels_test.jsonl']
        self.refresh(); viewer = EvidenceViewer('fixture', root=self.root)
        with patch.object(EvidenceViewer, '_read', self.forbid_labels(EvidenceViewer._read)):
            with self.assertRaises(ValueError): viewer.sources(viewer.games[self.train])

    def unlock(self):
        selection_sha = write(self.run / 'training/forecast/selection.json', {'selected': 'fixture'})
        pred_sha = jsonl(self.run / 'training/test_predictions.jsonl', [{'game_id': 'espn:1', 'boosting': .6, 'forecast_lora': .7}])
        label_sha = jsonl(self.run / 'training/labels_test.jsonl', [{'game_id': 'espn:1', 'home_win': 1}])
        evaluation_sha = write(self.run / 'training/evaluation.json', {'selection_sha256': selection_sha,
                'test_predictions_sha256': pred_sha, 'test_labels_sha256': label_sha})
        write(self.run / 'training/evaluation_manifest.json', {'files': {'evaluation.json': evaluation_sha,
              'test_predictions.jsonl': pred_sha, 'labels_test.jsonl': label_sha}})
        write(self.run / 'evaluation_authorization.json', {'protocol_sha256': self.protocol_sha, 'selection_sha256': selection_sha})
        write(self.run / 'training/evaluation_started.json', {'selection_sha256': selection_sha})
        write(self.run / 'stages/evaluate/complete.json', {'protocol_sha256': self.protocol_sha, 'outputs': {'evaluation': evaluation_sha}})

    def test_outcomes_require_completion_and_matching_prediction_selection_locks(self):
        self.unlock(); self.assertTrue(self.viewer.evaluation_unlocked())
        self.assertEqual(self.viewer.tab_data(self.test, 'forecast')['evaluated_outcome']['home_win'], 1)
        write(self.run / 'evaluation_authorization.json', {'protocol_sha256': self.protocol_sha, 'selection_sha256': 'wrong'})
        with patch.object(EvidenceViewer, '_read', self.forbid_labels(EvidenceViewer._read)):
            data = self.viewer.tab_data(self.test, 'forecast')
            self.assertFalse(data['evaluation_complete']); self.assertNotIn('evaluated_outcome', data)

    def test_tampered_prediction_or_source_hash_fails_closed(self):
        self.unlock(); (self.run / 'training/test_predictions.jsonl').write_text('{}\n')
        with patch.object(EvidenceViewer, '_read', self.forbid_labels(EvidenceViewer._read)):
            with self.assertRaisesRegex(ValueError, 'checksum'): self.viewer.evaluation_unlocked()
        source = self.rows[0]['sources'][0]
        (self.snapshot / source['pages_path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'checksum'): self.viewer.sources(self.rows[0])

    def test_only_protocol_games_are_routable(self):
        self.protocol['game_ids'] = ['espn:0']; write(self.run / 'protocol.json', self.protocol)
        viewer = EvidenceViewer('fixture', root=self.root)
        self.assertEqual(len(viewer.games), 1)
        with self.assertRaises(KeyError): viewer.dispatch(f'/source/{self.test}/0')
        with self.assertRaises(ValueError): EvidenceViewer('../fixture', root=self.root)


if __name__ == '__main__':
    unittest.main()
