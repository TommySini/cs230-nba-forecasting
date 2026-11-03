import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bam_claim_guards import date_grounding, dense_statistical_row, prose_date_grounding, cited_season_years, literal_comparison_consistency, threshold_average_precision


def cited(*lines):
    return [{"quotations": [{"line": i, "text": text} for i, text in enumerate(lines, 1)]}]


class ClaimGuardTests(unittest.TestCase):
    def test_short_thousands_totals_are_dense(self):
        for row in ("TOTALS 3,044 870 1,106 2,280 1,118", "OPPONENTS 2,694 890 908 1,872 1,123", "AVERAGES 51.6 14.7 18.7 38.6 18.9"):
            self.assertTrue(dense_statistical_row(row))

    def test_ordinary_columns_and_numeric_formats(self):
        self.assertTrue(dense_statistical_row("Player 8 7 .450 42.5% +12 −5 1,230"))
        self.assertTrue(dense_statistical_row("01/02/24 at BOS 32 8 15 2 5 4 4 22"))

    def test_narrative_and_labeled_rates_are_not_dense(self):
        for text in ("TOTALS were 3,044 points across 59 games, including 870 at home.", "He averaged 20 points, 6 rebounds and 8 assists.", "LAST FIVE PLAYED: 17.8 pts | 3.0 reb | 1.6 ast | 1.2 stl | 0.2 blk"):
            self.assertFalse(dense_statistical_row(text))

    def test_null_date_is_explicitly_unknown(self):
        self.assertEqual(date_grounding(None, [])['method'], 'unspecified_date')

    def test_yearless_milestone_does_not_invent_year(self):
        for year in (2023, 2024):
            self.assertFalse(date_grounding(f'{year}-10-30', cited('Mitchell made his 1,400th three-pointer on Oct. 30.'))['grounded'])

    def test_explicit_formats_and_wrong_year(self):
        for text in ('2024-10-30', '10/30/2024', '10/30/24', 'Oct. 30, 2024', 'October 30th 2024'):
            self.assertTrue(date_grounding('2024-10-30', cited(text))['grounded'], text)
            self.assertFalse(date_grounding('2023-10-30', cited(text))['grounded'], text)

    def test_cited_season_context_resolves_month_day(self):
        quotes = cited('2024-25 SEASON MILESTONES', 'Milestones on Oct. 30 and January 15.')
        self.assertTrue(date_grounding('2024-10-30', quotes)['grounded'])
        self.assertTrue(date_grounding('2025-01-15', quotes)['grounded'])
        self.assertFalse(date_grounding('2023-10-30', quotes)['grounded'])

    def test_ambiguous_seasons_stay_unresolved(self):
        quotes = cited('2023-24 and 2024-25 milestones', 'On Oct. 30 he reached a milestone.')
        self.assertFalse(date_grounding('2024-10-30', quotes)['grounded'])

    def test_season_from_another_citation_does_not_resolve_year(self):
        quotes = cited('2024-25 season') + cited('Career milestone on Oct. 30.')
        self.assertFalse(date_grounding('2024-10-30', quotes)['grounded'])

    def test_invalid_source_dates_are_not_grounding(self):
        self.assertFalse(date_grounding('2024-10-30', cited('2024-02-30 and 13/30/24'))['grounded'])

    def test_date_field_and_citations_must_be_valid(self):
        for value in ('2024-2-01', '2024-02-30', 20241030):
            with self.assertRaises(ValueError):
                date_grounding(value, cited('2024-10-30'))
        with self.assertRaises(ValueError):
            date_grounding('2024-10-30', [{'lines': [1]}])

    def test_dates_in_prose_need_evidence_independent_of_metadata(self):
        quotes = cited('Mitchell milestone on Oct. 30.')
        for text in ('Milestone on2023-10-30.', 'Milestone on October 30, 2023.', 'Milestone on10/30/23.'):
            # Real prose separates its date token from preceding words.
            text = text.replace('on20', 'on 20').replace('on10', 'on 10')
            self.assertFalse(prose_date_grounding(text, quotes)['grounded'])
        self.assertTrue(prose_date_grounding('Milestone on Oct. 30.', quotes)['grounded'])
        self.assertTrue(prose_date_grounding('Milestone on October 30, 2024.', cited('2024-25 SEASON', 'Milestone on Oct. 30.'))['grounded'])

    def test_only_explicit_consecutive_seasons_supply_year_expansion(self):
        self.assertEqual(cited_season_years(cited('2024-25 season')), {2024,2025})
        self.assertEqual(cited_season_years(cited('2023-25 seasons')), set())

    def test_actual_miami_comparisons_are_checked_independently(self):
        text = "The Miami Heat's 2024-25 regular season stats comparison shows a lower points per game (111.3) compared to the opponent's (120.7), and a higher opponent points per game (109.9) compared to the opponent's (110.8)."
        result = literal_comparison_consistency(text)
        self.assertFalse(result['consistent'])
        self.assertEqual([c['consistent'] for c in result['checks']], [True, False])

    def test_literal_same_unit_arithmetic_and_strict_equality(self):
        for text in ('109.9 points per game is lower than 110.8 PPG',
                     'higher points per game (120.7) than PPG (111.3)',
                     'lower 3-point percentage (35.2%) than 3FG% (36.1%)'):
            result = literal_comparison_consistency(text)
            self.assertTrue(result['consistent'], text)
            self.assertEqual(len(result['checks']), 1, text)
        for text in ('109.9 PPG is higher than 110.8 points per game',
                     'higher 3-point percentage (35.2%) than 3FG% (36.1%)',
                     'higher points per game (111.3) than PPG (111.3)'):
            self.assertFalse(literal_comparison_consistency(text)['consistent'], text)

    def test_ambiguous_numbers_metrics_units_and_negation_are_not_bound(self):
        for text in ('higher 3-point percentage (rank 3, 35.2%) than 3FG% (36.1%)',
                     'higher points per game (111.3 (rank 3)) than PPG (120.7)',
                     'higher points per game (20-5) than PPG (30-7)',
                     'higher points per game (10) than assists per game (20)',
                     'higher 3-point percentage (35.2%) than FG% (45.1%)',
                     'higher 3-point percentage (.352) than 3FG% (36.1%)',
                     'not higher points per game (109.9) than PPG (110.8)',
                     'no higher points per game (109.9) than PPG (110.8)',
                     "isn't higher points per game (109.9) than PPG (110.8)",
                     '3-point shooting is higher than last season',
                     'at least 20 points, 6 rebounds and 8 assists'):
            result = literal_comparison_consistency(text)
            self.assertTrue(result['consistent'], text)
            self.assertEqual(result['checks'], [], text)

    def test_actual_cunningham_threshold_is_not_an_exact_average(self):
        quotes = cited('Cunningham joins Jokic and LeBron James as the only players in the NBA averaging at least',
                       '20 points, six rebounds and eight assists while nearing league leaderboards.',
                       'With 25.7 PPG and 24.2 points created off assists, Cunningham accounts for 50.0 points.')
        self.assertFalse(threshold_average_precision('He averages 20 points, six rebounds, and eight assists per game.', quotes)['supported_precision'])
        self.assertTrue(threshold_average_precision('He averages at least 20 points, six rebounds and eight assists per game.', quotes)['supported_precision'])

    def test_independent_exact_average_evidence_and_different_values_not_rejected(self):
        bound = cited('Players averaging at least 20 points, 6 rebounds and 8 assists.')
        exact = cited('Cunningham averages 20 points, six rebounds and eight assists per game.')
        claim = 'Cunningham averages 20 points, six rebounds and eight assists per game.'
        self.assertTrue(threshold_average_precision(claim, bound + exact)['supported_precision'])
        self.assertTrue(threshold_average_precision('He averages 25.7 points, 6.3 rebounds and 9.5 assists.', bound)['supported_precision'])
        self.assertTrue(threshold_average_precision('He averages 20 assists and 6 points.', bound)['supported_precision'])
        self.assertTrue(threshold_average_precision('He is not averaging 20 points, 6 rebounds and 8 assists.', bound)['supported_precision'])
        self.assertTrue(threshold_average_precision('His 3-point average is 20.', bound)['supported_precision'])


if __name__ == '__main__':
    unittest.main()
