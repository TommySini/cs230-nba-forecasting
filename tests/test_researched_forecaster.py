"""Checks of probability fitting, evidence coverage, and leakage boundaries."""
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.optimize import check_grad
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_researched_forecaster import (objective, fit_head, choose_candidate, pool_vectors,
                                       prompt_messages, chunk_report, evidence_with_context)
from run_game_note_research import page_lines
from run_boosting_notes import log_odds


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(len(' '.join(m['content'] for m in messages).split())))


class ResearchedForecasterTests(unittest.TestCase):
    def test_analytic_gradient_and_fitting_with_fixed_offset(self):
        rng = np.random.default_rng(230)
        x = rng.normal(size=(30, 5))
        y = (x[:, 0] > 0).astype(float)
        offset = rng.normal(size=30)
        theta = rng.normal(size=6)
        error = check_grad(lambda t: objective(t, x, y, offset, .1)[0],
                           lambda t: objective(t, x, y, offset, .1)[1], theta)
        self.assertLess(error, 1e-6)
        fitted, stats = fit_head(x, y, offset, .1)
        self.assertLess(objective(fitted, x, y, offset, .1)[0], objective(np.zeros(6), x, y, offset, .1)[0])
        self.assertLess(stats['gradient_max'], 1e-6)

    def test_zero_head_preserves_booster_and_dev_ties_prefer_zero(self):
        p = np.array([.01, .3, .5, .9, .99])
        np.testing.assert_allclose(expit(log_odds(p)), p, rtol=0, atol=1e-14)
        candidates = [{'name': 'l2_1', 'penalty': 1, 'dev': {'brier': .2}},
                      {'name': 'zero_correction', 'penalty': 0, 'dev': {'brier': .2}}]
        self.assertEqual(choose_candidate(candidates)['name'], 'zero_correction')

    def test_prompt_whitelist_and_complete_chunk_coverage(self):
        row = {'home_abbr': 'BOS', 'away_abbr': 'MIA', 'game_date_et': '2024-12-02', 'p_boost': .65, 'home_win': 1,
               'game_id': 'secret-id', 'raw_espn': 'POSTGAME RECAP'}
        facts = [{'id': str(i), 'origin': 'language_model', 'category': 'workload', 'time_scope': 'UNVERIFIED_SCOPE',
                  'claim': 'HALLUCINATED_SUMMARY', 'caveat': 'UNVERIFIED_CAVEAT',
                  'citations': [{'page': i + 1, 'text': f'Player {i} played thirty minutes yesterday. ' * 10}]}
                 for i in range(9)]
        chunks = chunk_report(row, facts, TinyTokenizer(), 220)
        self.assertGreater(len(chunks), 1)
        self.assertEqual([f['id'] for c in chunks for f in c], [f['id'] for f in facts])
        text = str(prompt_messages(row, facts))
        for forbidden in ('home_win', 'secret-id', 'POSTGAME RECAP', 'HALLUCINATED_SUMMARY', 'UNVERIFIED_SCOPE', 'UNVERIFIED_CAVEAT'):
            self.assertNotIn(forbidden, text)
        with self.assertRaises(ValueError):
            chunk_report(row, facts, TinyTokenizer(), 10)

    def test_context_is_copied_from_source_and_citation_changes_rejected(self):
        packet = {'pages': [{'page': 2, 'text': 'Player heading\nEarlier sentence\nCited sentence\nLater qualifier\nDated context'}]}
        cited = page_lines(packet['pages'][0])[2]
        fact = {'origin': 'language_model', 'citations': [cited]}
        result = evidence_with_context([fact], packet)[0]['forecast_citations']
        self.assertEqual(result, page_lines(packet['pages'][0]))
        bad = {'origin': 'language_model', 'citations': [{**cited, 'text': 'Invented source quote'}]}
        with self.assertRaises(ValueError):
            evidence_with_context([bad], packet)

    def test_pooling_uses_later_chunks(self):
        vector = pool_vectors([[1., 0.], [0., 1.]])
        np.testing.assert_allclose(vector, np.array([1., 1.]) / np.sqrt(2))
        with self.assertRaises(ValueError):
            pool_vectors([[0., 0.]])


if __name__ == '__main__':
    unittest.main()
