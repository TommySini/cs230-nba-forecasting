"""BAM provenance, coverage, review and hindsight boundary regression tests."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import bam_research as bam


def game(split='train'):
    return {'game_id': 'g1', 'game_date_et': '2025-01-02', 'cutoff_utc': '2025-01-02T23:00:00Z',
            'tipoff_utc': '2025-01-03T00:00:00Z', 'home_abbr': 'BOS', 'away_abbr': 'MIA',
            'split': split, 'fold_id': 'fold_0', 'p_boost': .6, 'forecast_eligible': True,
            'home_win': 1, 'recap': 'NEVER PASS THIS',
            'sources': [{'source_id': 's1', 'publisher': 'BOS', 'url': 'https://example.com/notes.pdf',
                         'sha256': 'a' * 64, 'archive_time': '2025-01-02T19:00:00Z',
                         'pages': [{'page': 1, 'lines': [{'line': 1, 'text': 'Brown is questionable tonight with an ankle injury.'},
                                                       {'line': 2, 'text': 'The team played on January 1, 2025.'}]},
                                   {'page': 2, 'lines': []}]}]}


def finding(module=bam.MODULES[0], ident='f1'):
    return {'id': ident, 'module': module, 'fact': 'Brown is questionable tonight.',
            'fact_date': None, 'time_scope': 'current',
            'interpretation': 'Rotation depth may matter if Brown is unavailable.',
            'counterevidence': 'An absence has not been confirmed.', 'uncertainty': 'Final status remains unknown.',
            'booster_overlap': 'unknown', 'citations': [{'source_id': 's1', 'page': 1, 'lines': [1]}]}


def module_output(module=bam.MODULES[0]):
    return {'module': module, 'findings': [finding(module, module + '-f1')],
            'unknowns': ['Opponent coverage unavailable.'], 'coverage_note': 'BOS notes only.'}


def postgame():
    return {'game_id': 'g1', 'split': 'train', 'normalized': True,
            'sources': [{'source_id': 'p1', 'publisher': 'ESPN normalized box score',
                         'url': 'https://example.com/box', 'sha256': 'b' * 64,
                         'archive_time': '2025-01-03T04:00:00Z',
                         'pages': [{'page': 1, 'lines': [{'line': 1, 'text': 'Boston won; Brown did not play.'}]}]}]}


def review_decision(finding):
    return {"id": finding["id"], "source_reading": "Brown is questionable for tonight, not confirmed absent.", "required_citations": [{"source_id": "s1", "page": 1, "lines": [1]}], "reason": "The source supports the scoped fact and conditional interpretation; the caveat preserves uncertainty.", "verdict": "supported"}


def legacy_review_decision(finding):
    return {"id": finding["id"], "source_evidence": [{"source_id": "s1", "page": 1, "lines": [1], "quotation": "Brown is questionable tonight with an ankle injury.", "source_reading": "Brown is questionable for tonight."}], "attribute_checks": [{"attribute": attribute, "claim_value": "Questionable tonight or no extra assertion", "source_value": "Questionable tonight or not applicable", "assessment": "match"} for attribute in bam.REVIEW_ATTRIBUTES], "contradictions": [], "reason": "The dated source supports this scoped claim.", "verdict": "supported"}


class BamResearchTests(unittest.TestCase):
    def setUp(self):
        self.game = game()
        self.requests = bam.build_selection_requests(self.game)

    def selections(self):
        return [bam.ingest_response(r, {'reviewed_chunk_id': r['chunk_id'], 'findings': [finding()] if r['coverage']['lines'] else [], 'disposition': 'All lines reviewed.'}, self.game, 'student') for r in self.requests]

    def approve(self, request, output, model='student'):
        artifact = bam.ingest_response(request, output, self.game, model)
        self.assertEqual(artifact['status'], 'mechanically_valid', artifact['errors'])
        request = bam.build_review_request(self.game, artifact, 'independent')
        response = {'decisions': [review_decision(f) for f in bam.all_findings(artifact['output'])], 'report_verdict': 'supported', 'report_reason': 'Unknowns and one-sided coverage explicit.'}
        review = bam.ingest_response(request, response, self.game, 'independent')
        return bam.apply_semantic_review(artifact, review)

    def module_requests(self):
        return bam.build_module_requests(self.game, self.selections())

    def modules(self):
        return [self.approve(r, module_output(r['module'])) for r in self.module_requests()]

    def postmortem(self):
        request = bam.build_postmortem_request(self.game, postgame())
        response = {'general': [{'observation': 'Brown did not play.', 'foreseeable_signal': 'Pregame uncertainty about rotation.', 'unforeseeable': 'The realized outcome was uncertain.', 'citations': [{'source_id': 'p1', 'page': 1, 'lines': [1]}]}], 'timeline': ['Pregame status was uncertain before a confirmed absence.'], 'market_view': 'Unavailable.'}
        return bam.ingest_response(request, response, self.game, 'teacher')

    def test_all_lines_and_empty_pages_are_covered(self):
        reqs = bam.build_selection_requests(self.game, 10)
        self.assertEqual([(r['coverage']['page'], r['coverage']['lines']) for r in reqs], [(1, [1]), (1, [2]), (2, [])])
        self.assertEqual(len(set(r['request_id'] for r in reqs)), 3)

    def test_pregame_allowlist_excludes_outcomes(self):
        for request in self.requests + self.module_requests():
            text = json.dumps(request['messages'])
            self.assertNotIn('home_win', text)
            self.assertNotIn('NEVER PASS THIS', text)
            self.assertNotIn('p_boost', text)

    def test_cutoff_and_archive_timestamps_fail_closed(self):
        for field, value in [('archive_time', '2025-01-02T23:00:01Z'), ('archive_time', '2025-01-02T19:00:00')]:
            bad = copy.deepcopy(self.game)
            bad['sources'][0][field] = value
            with self.assertRaises(ValueError):
                bam.build_selection_requests(bad)
        bad = copy.deepcopy(self.game)
        bad['tipoff_utc'] = '2025-01-03T00:01:00Z'
        with self.assertRaises(ValueError):
            bam.build_selection_requests(bad)

    def test_missing_chunk_cannot_silently_disappear(self):
        with self.assertRaisesRegex(ValueError, 'coverage'):
            bam.build_module_requests(self.game, self.selections()[:1])
        with self.assertRaisesRegex(ValueError, 'coverage'):
            bam.build_module_requests(self.game, self.selections() * 2)

    def test_selection_requires_explicit_disposition(self):
        artifact = bam.ingest_response(self.requests[0], {'findings': []}, self.game, 'student')
        self.assertEqual(artifact['status'], 'rejected')

    def test_quotes_are_copied_from_canonical_source(self):
        output = module_output()
        output['findings'][0]['citations'][0]['quotations'] = ['forged quote']
        artifact = bam.ingest_response(self.module_requests()[0], output, self.game, 'student')
        self.assertEqual(artifact['output']['findings'][0]['citations'][0]['quotations'][0]['text'], self.game['sources'][0]['pages'][0]['lines'][0]['text'])
        self.assertEqual(artifact['status'], 'mechanically_valid')  # copying != approval

    def test_bad_citations_are_retained_and_repairable(self):
        request = self.module_requests()[0]
        output = module_output()
        output['findings'][0]['citations'][0]['lines'] = [999]
        artifact = bam.ingest_response(request, output, self.game, 'student')
        self.assertEqual(artifact['status'], 'rejected')
        self.assertEqual(artifact['raw_response'], output)
        repair = bam.build_repair_request(self.game, request, artifact)
        self.assertIn(artifact['artifact_hash'], repair['parent_hashes'])
        self.assertNotEqual(repair['request_id'], request['request_id'])
        with self.assertRaisesRegex(ValueError, 'budget'):
            bam.build_repair_request(self.game, request, artifact, attempt=3)

    def test_postmortems_and_gold_are_train_only(self):
        for split in ('dev', 'test'):
            with self.assertRaisesRegex(ValueError, 'TRAIN ONLY'):
                bam.build_postmortem_request(game(split), postgame())
            with self.assertRaisesRegex(ValueError, 'TRAIN ONLY'):
                bam.build_gold_request(game(split), [], {})

    def test_postmortems_require_normalized_sourced_records(self):
        bad = postgame()
        bad['normalized'] = False
        with self.assertRaisesRegex(ValueError, 'normalized'):
            bam.build_postmortem_request(self.game, bad)
        bad = postgame()
        bad['sources'] = []
        with self.assertRaises(ValueError):
            bam.build_postmortem_request(self.game, bad)

    def test_postmortem_citations_use_only_postgame_store(self):
        artifact = self.postmortem()
        self.assertEqual(artifact['status'], 'mechanically_valid', artifact['errors'])
        self.assertEqual(artifact['output']['general'][0]['citations'][0]['source_id'], 'p1')
        output = module_output()
        output['findings'][0]['citations'][0]['source_id'] = 'p1'
        artifact = bam.ingest_response(self.module_requests()[0], output, self.game, 'student')
        self.assertEqual(artifact['status'], 'rejected')

    def test_gold_combines_reviewed_research_outputs_and_separate_postmortem(self):
        modules = self.modules()
        post = self.postmortem()
        request = bam.build_gold_request(self.game, modules, post)
        payload = json.loads(request['messages'][1]['content'])
        self.assertEqual(len(payload['forecasting_context']['researcher_outputs']), 3)
        self.assertIn('training_only_postmortem', payload)
        self.assertEqual(request['parent_hashes'], [a['artifact_hash'] for a in modules] + [post['artifact_hash']])
        modules[0]['status'] = 'mechanically_valid'
        with self.assertRaises(ValueError):
            bam.build_gold_request(self.game, modules, post)

    def test_future_facts_and_arbitrary_effects_rejected(self):
        request = self.module_requests()[0]
        for change in ({'fact_date': '2025-01-03'}, {'fact': 'Brown returned on 2025-01-03.'}, {'interpretation': 'This raises win probability by 7%.'}, {'fact': 'Boston won this game.'}):
            output = module_output()
            output['findings'][0].update(change)
            artifact = bam.ingest_response(request, output, self.game, 'student')
            self.assertEqual(artifact['status'], 'rejected', change)

    def test_unsupported_numeric_claim_rejected(self):
        output = module_output()
        output['findings'][0]['fact'] = 'Brown scored 57 points.'
        artifact = bam.ingest_response(self.module_requests()[0], output, self.game, 'student')
        self.assertEqual(artifact['status'], 'rejected')

    def test_missing_semantic_verdict_is_rejected(self):
        artifact = bam.ingest_response(self.module_requests()[0], module_output(), self.game, 'student')
        request = bam.build_review_request(self.game, artifact, 'independent')
        review = bam.ingest_response(request, {'decisions': [], 'report_verdict': 'supported', 'report_reason': 'okay'}, self.game, 'independent')
        self.assertEqual(review['status'], 'rejected')
        self.assertEqual(bam.apply_semantic_review(artifact, review)['status'], 'rejected')

    def test_same_model_cannot_self_certify(self):
        artifact = bam.ingest_response(self.module_requests()[0], module_output(), self.game, 'student')
        with self.assertRaisesRegex(ValueError, 'differ'):
            bam.build_review_request(self.game, artifact, 'student')
        request = bam.build_review_request(self.game, artifact, 'independent')
        review = bam.ingest_response(request, {}, self.game, 'student')
        self.assertEqual(review['status'], 'rejected')

    def test_reviews_never_receive_postmortem(self):
        request = bam.build_gold_request(self.game, self.modules(), self.postmortem())
        candidate = bam.ingest_response(request, {'modules': [module_output(m) for m in bam.MODULES]}, self.game, 'teacher')
        review = bam.build_review_request(self.game, candidate, 'independent')
        text = json.dumps(review['messages'])
        self.assertNotIn('Boston won', text)
        self.assertNotIn('training_only_postmortem', text)

    def test_truncated_or_non_json_response_is_retained(self):
        for response, reason in [('{"findings":', 'length'), ('not json', 'stop')]:
            artifact = bam.ingest_response(self.requests[0], response, self.game, 'student', finish_reason=reason)
            self.assertEqual(artifact['status'], 'rejected')
            self.assertEqual(artifact['raw_response'], response)

    def test_gold_export_has_three_module_targets_and_no_postmortem_input(self):
        module_requests = self.module_requests()
        request = bam.build_gold_request(self.game, self.modules(), self.postmortem())
        gold = self.approve(request, {'modules': [module_output(m) for m in bam.MODULES]}, model='teacher')
        examples = bam.export_student_examples(self.game, module_requests, gold)
        self.assertEqual(len(examples), 3)
        for example in examples:
            self.assertEqual([m['role'] for m in example['messages']], ['system', 'user', 'assistant'])
            self.assertEqual(example['fold_id'], 'fold_0')
            self.assertNotIn('training_only_postmortem', json.dumps(example['messages'][:-1]))
        with self.assertRaises(ValueError):
            bam.assemble_student_brief(self.game, [gold] * 3)

    def test_student_brief_rejects_same_game_training(self):
        with self.assertRaisesRegex(ValueError, 'trained'):
            bam.assemble_student_brief(self.game, self.modules(), researcher_training_game_ids=['g1'])
        brief = bam.assemble_student_brief(self.game, self.modules(), researcher_training_game_ids=['earlier'])
        self.assertEqual(brief['origin'], 'student_generated')
        self.assertNotIn('home_win', brief)

    def test_source_changes_between_request_and_ingest_fail_closed(self):
        changed = copy.deepcopy(self.game)
        changed["sources"][0]["pages"][0]["lines"][0]["text"] = "Different evidence"
        artifact = bam.ingest_response(self.requests[0], {"reviewed_chunk_id": self.requests[0]["chunk_id"], "findings": [], "disposition": "read"}, changed, "student")
        self.assertEqual(artifact["status"], "rejected")
        self.assertIn("source changed", artifact["errors"][0])

    def test_code_decoded_tables_require_exact_fact_and_complete_context(self):
        from game_note_tables import HEADER
        g = copy.deepcopy(self.game)
        g["sources"][0]["pages"].append({"page": 3, "lines": [{"line": i, "text": text} for i, text in enumerate(["  BAM ADEBAYO RECENT GAMES  ", HEADER, "12/01/24 at TOR 38:36 5 13 1 2 2 2 4 16 20 7 3 1 5 0 +15 13"], 1)]})
        table = bam.code_tables(g)[0][0]
        f = finding(bam.MODULES[2])
        f.update(fact=table["fact"], fact_date="2024-12-01", time_scope="prior_game", decoded_table_id=table["id"], citations=table["citations"])
        self.assertEqual(bam._validate_finding(g, f)["fact"], table["fact"])
        f["citations"][0]["lines"] = [3]
        with self.assertRaisesRegex(ValueError, "heading/header"):
            bam._validate_finding(g, f)

    def test_bounded_hierarchy_covers_every_group_and_checks_siblings(self):
        findings = []
        for i in range(15):
            f = finding(ident=f"f{i}")
            f["uncertainty"] = "A source-specific uncertainty. " * 20
            findings.append(f)
        requests = bam._bounded_module_requests(self.game, bam.MODULES[0], findings, [], {"reviewed_chunks": ["all"]}, 9000)
        self.assertGreater(len(requests), 1)
        self.assertTrue(all(r["stage"] == "module_partial" for r in requests))
        seen = [f["id"] for r in requests for f in json.loads(r["messages"][1]["content"])["selected_findings"]]
        self.assertEqual(seen, [f["id"] for f in findings])
        self.assertTrue(all(sum(len(m["content"]) for m in r["messages"]) <= 9000 for r in requests))
        artifacts = [bam.ingest_response(r, module_output(), self.game, "student") for r in requests]
        with self.assertRaisesRegex(ValueError, "siblings"):
            bam.build_module_synthesis_requests(self.game, artifacts[:-1])
        final = bam.build_module_synthesis_requests(self.game, artifacts)
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]["stage"], "module")

    def test_compact_training_export_and_forecast_target_keep_citations(self):
        request = bam.build_gold_request(self.game, self.modules(), self.postmortem())
        gold = self.approve(request, {"modules": [module_output(m) for m in bam.MODULES]}, model="teacher")
        example = bam.export_student_examples(self.game, self.module_requests(), gold)[0]
        self.assertEqual(example["review_status"], "accepted")
        self.assertEqual(example["teacher_hash"], gold["artifact_hash"])
        self.assertNotIn("quotations", example["messages"][-1]["content"])
        explanation = bam.export_forecast_explanation(self.game, gold)
        self.assertIn('"source_id": "s1"', explanation["explanation"])
        self.assertNotIn("quotations", explanation["explanation"])
        inputs = bam.export_student_inputs(self.game, self.module_requests())
        self.assertTrue(all(len(row["messages"]) == 2 and "teacher_hash" not in row for row in inputs))

    def test_selection_retains_individual_rejects_without_losing_page(self):
        bad = finding(ident="bad")
        bad["citations"][0]["lines"] = [999]
        output = {"reviewed_chunk_id": self.requests[0]["chunk_id"], "findings": [finding(), bad], "disposition": "All page lines reviewed; incomplete citation retained for repair."}
        artifact = bam.ingest_response(self.requests[0], output, self.game, "student")
        self.assertEqual(artifact["status"], "mechanically_valid")
        self.assertEqual(len(artifact["output"]["findings"]), 1)
        self.assertEqual(artifact["output"]["rejected_findings"][0]["candidate"], bad)
        repair = bam.build_selection_repair_request(self.game, self.requests[0], artifact)
        self.assertIn(artifact["artifact_hash"], repair["parent_hashes"])

    def test_cited_context_keeps_distant_season_and_player_headers(self):
        g = copy.deepcopy(self.game)
        lines = [{"line": i, "text": "Unrelated paragraph."} for i in range(1, 41)]
        lines[9]["text"] = "JARRETT ALLEN"
        lines[11]["text"] = "2023-24 SEASON"
        lines[13]["text"] = "GP GS MPG PTS REB AST"
        lines[29]["text"] = "Played 77 games."
        g["sources"][0]["pages"][0]["lines"] = lines
        f = finding()
        f["citations"][0]["lines"] = [30]
        evidence = bam._evidence_for(g, [f])
        included = {line["line"] for line in evidence[0]["pages"][0]["lines"]}
        self.assertTrue({10, 12, 14, 30} <= included)

    def test_token_budget_is_checked_in_addition_to_characters(self):
        counter = lambda request: len(json.loads(request["messages"][1]["content"])["selected_findings"]) * 100
        findings = [finding(ident=str(i)) for i in range(5)]
        requests = bam._bounded_module_requests(self.game, bam.MODULES[0], findings, [], {}, 48000, token_counter=counter, max_input_tokens=200)
        self.assertEqual(len(requests), 3)
        self.assertTrue(all(counter(r) <= 200 for r in requests))

    def test_malformed_nested_gold_and_postmortem_are_retained(self):
        request = bam.build_gold_request(self.game, self.modules(), self.postmortem())
        artifact = bam.ingest_response(request, {"modules": [1, 2, 3]}, self.game, "teacher")
        self.assertEqual(artifact["status"], "rejected")
        request = bam.build_postmortem_request(self.game, postgame())
        artifact = bam.ingest_response(request, {"general": [None], "timeline": [], "market_view": "Unavailable."}, self.game, "teacher")
        self.assertEqual(artifact["status"], "rejected")

    def test_repaired_semantic_review_keeps_candidate_binding(self):
        candidate = bam.ingest_response(self.module_requests()[0], module_output(), self.game, "student")
        request = bam.build_review_request(self.game, candidate, "independent")
        rejected = bam.ingest_response(request, {"decisions": []}, self.game, "independent")
        repair = bam.build_repair_request(self.game, request, rejected)
        response = {"decisions": [review_decision(f) for f in bam.all_findings(candidate["output"])], "report_verdict": "supported", "report_reason": "Complete scoped report."}
        reviewed = bam.ingest_response(repair, response, self.game, "independent")
        self.assertEqual(bam.apply_semantic_review(candidate, reviewed)["status"], "approved")
        reviewed["target_artifact_hash"] = "wrong"
        with self.assertRaisesRegex(ValueError, "parent mismatch"):
            bam.apply_semantic_review(candidate, reviewed)

    def test_review_sees_omitted_opposing_evidence_from_original_module_input(self):
        request = self.module_requests()[0]
        payload = json.loads(request["messages"][1]["content"])
        opposing = finding(ident="opposing")
        opposing["fact"] = "Omitted contradictory availability evidence."
        payload["selected_findings"].append(opposing)
        request["messages"][1]["content"] = json.dumps(payload)
        request["input_hash"] = bam.value_hash(payload)
        candidate = bam.ingest_response(request, module_output(), self.game, "student")
        review = bam.build_review_request(self.game, candidate, "independent", original_request=request)
        supplied = json.loads(review["messages"][1]["content"])
        self.assertIn("Omitted contradictory availability evidence.", json.dumps(supplied["original_pregame_research_context"]))
        with self.assertRaisesRegex(ValueError, "token budget"):
            bam.build_review_request(self.game, candidate, "independent", original_request=request, token_counter=lambda r:30000)

    def test_gold_review_original_context_excludes_hindsight(self):
        request = bam.build_gold_request(self.game, self.modules(), self.postmortem())
        candidate = bam.ingest_response(request, {"modules": [module_output(m) for m in bam.MODULES]}, self.game, "teacher")
        review = bam.build_review_request(self.game, candidate, "independent", original_request=request)
        supplied = json.loads(review["messages"][1]["content"])
        self.assertIn("forecasting_context", supplied["original_pregame_research_context"])
        self.assertNotIn("training_only_postmortem", json.dumps(supplied))
        self.assertNotIn("Boston won", json.dumps(supplied))
        self.assertNotIn("Brown did not play", json.dumps(supplied))

    def test_mixed_layout_prose_can_supply_numbers_without_decoding_table(self):
        g = copy.deepcopy(self.game)
        g["sources"][0]["pages"][0]["lines"] = [
            {"line": 1, "text": "NIKOLA JOKIC — 2024-25 PRESEASON"},
            {"line": 2, "text": "Jokic averaged 16.5 points, 8.0 rebounds and 4.8 assists over four preseason games."},
            {"line": 3, "text": "GP GS PPG RPG APG SPG BPG MPG"},
            {"line": 4, "text": "4 4 16.5 8.0 4.8 1.5 0.5 21.9"}]
        f = finding(bam.MODULES[1])
        f.update(fact="Jokic averaged 16.5 points, 8.0 rebounds and 4.8 assists over 4 preseason games.", time_scope="historical")
        f["citations"][0]["lines"] = [1, 2, 3, 4]
        self.assertEqual(bam._validate_finding(g, f)["fact"], f["fact"])
        f["citations"][0]["lines"] = [1, 3, 4]
        with self.assertRaisesRegex(ValueError, "Dense tables"):
            bam._validate_finding(g, f)

    def test_explicit_current_season_ending_and_remainder_status_supported(self):
        for announcement in ("NOTE: Will undergo season-ending surgery on his right PCL.", "JANUARY 1, 2025: The club announced Brown would miss the remainder of the season with a shoulder injury."):
            g = copy.deepcopy(self.game)
            g["sources"][0]["pages"][0]["lines"] = [{"line": 1, "text": "BROWN — 2024-25 SEASON"}, {"line": 2, "text": announcement}]
            f = finding();f.update(fact="Brown is described as having a season-ending injury.", fact_date=None)
            f["citations"][0]["lines"] = [1, 2]
            bam._validate_finding(g, f)

    def test_old_season_ending_recap_cannot_establish_current_absence(self):
        g = copy.deepcopy(self.game)
        g["sources"][0]["pages"][0]["lines"] = [{"line": 1, "text": "BROWN — 2023-24 SEASON"}, {"line": 2, "text": "NOVEMBER 22, 2023: Will undergo season-ending surgery on his right PCL."}]
        f = finding();f.update(fact="Brown is unavailable this season after season-ending surgery.", fact_date=None)
        f["citations"][0]["lines"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "Current availability"):
            bam._validate_finding(g, f)

    def test_supported_review_cannot_contradict_its_own_evidence_matrix(self):
        candidate = bam.ingest_response(self.module_requests()[0], module_output(), self.game, "student")
        request = bam.build_review_request(self.game, candidate, "independent")
        request["review_schema_version"] = 3  # Historical artifact validation remains available.
        decision = legacy_review_decision(candidate["output"]["findings"][0])
        decision["attribute_checks"][2].update(claim_value="5 points", source_value="13 points", assessment="contradiction")
        response = {"decisions": [decision], "report_reason": "Complete report.", "report_verdict": "supported"}
        artifact = bam.ingest_response(request, response, self.game, "independent")
        self.assertEqual(artifact["status"], "rejected")
        self.assertIn("contradicts", artifact["errors"][0])

    def test_review_context_union_preserves_omitted_source_line_without_duplication(self):
        request = self.module_requests()[0]
        payload = json.loads(request["messages"][1]["content"])
        original_lines = payload["source_context"][0]["pages"][0]["lines"]
        candidate = bam.ingest_response(request, module_output(), self.game, "student")
        review = bam.build_review_request(self.game, candidate, "independent", original_request=request)
        reviewed = json.loads(review["messages"][1]["content"])
        lines = reviewed["pregame_source_context"][0]["pages"][0]["lines"]
        self.assertTrue({line["line"] for line in original_lines} <= {line["line"] for line in lines})
        self.assertEqual(len(lines), len({line["line"] for line in lines}))
        self.assertEqual(reviewed["original_pregame_research_context"]["source_context"]["location"], "pregame_source_context")

    def test_old_season_ending_slash_date_cannot_be_redated_by_model(self):
        g = copy.deepcopy(self.game)
        g["sources"][0]["pages"][0]["lines"] = [{"line": 1, "text": "DRU SMITH"}, {"line": 2, "text": "INJURY: Suffered a season-ending right ACL injury last season at CLE on 11/22/23."}]
        f = finding();f.update(fact="Dru Smith has a season-ending ACL injury.", fact_date=g["game_date_et"])
        f["citations"][0]["lines"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "fact_date is not grounded"):
            bam._validate_finding(g, f)
        f["fact_date"] = None
        with self.assertRaisesRegex(ValueError, "Current availability"):
            bam._validate_finding(g, f)

    def test_season_table_decoder_integrates_into_admission_and_selection(self):
        g = copy.deepcopy(self.game)
        g["sources"][0]["pages"][0]["lines"] = [
            {"line": 1, "text": "2024-25 REGULAR SEASON PPG OPP. PPG FG% 3FG% FT% RPG APG SPG TO BPG"},
            {"line": 2, "text": "Cavaliers 121.8 110.2 .504 .406 .775 43.5 28.5 9.13 13.2 4.43"}]
        table = next(t for t in bam.code_tables(g)[0] if t.get("decoder") == "team_season_averages_v1")
        f = finding(bam.MODULES[1]);f.update(fact=table["fact"], time_scope=table["time_scope"], citations=table["citations"], decoded_table_id=table["id"])
        self.assertEqual(bam._validate_finding(g, f)["fact"], table["fact"])
        request = bam.build_selection_requests(g)[0]
        supplied = json.loads(request["messages"][1]["content"])["code_decoded_tables"]
        self.assertIn(table["id"], {t["id"] for t in supplied})
        f["fact"] = f["fact"].replace("121.8", "128.1")
        with self.assertRaisesRegex(ValueError, "remain exact"):
            bam._validate_finding(g, f)

    def test_unreviewed_extra_outcome_or_summary_fields_are_rejected(self):
        request = self.module_requests()[0]
        for place in ("finding", "module"):
            output = module_output()
            if place == "finding": output["findings"][0]["home_win"] = 1
            else: output["uncited_summary"] = "An outcome-aware statement outside findings."
            artifact = bam.ingest_response(request, output, self.game, "student")
            self.assertEqual(artifact["status"], "rejected")
            self.assertIn("Unexpected", artifact["errors"][0])

    def test_v3_review_materializes_source_quotes_instead_of_model_copy(self):
        candidate = bam.ingest_response(self.module_requests()[0], module_output(), self.game, "student")
        request = bam.build_review_request(self.game, candidate, "independent")
        request["review_schema_version"] = 3
        decision = legacy_review_decision(candidate["output"]["findings"][0])
        decision["source_evidence"][0]["quotation"] = "Untrusted model-provided quotation."
        response = {"decisions": [decision], "report_reason": "Complete scoped report.", "report_verdict": "supported"}
        reviewed = bam.ingest_response(request, response, self.game, "independent")
        self.assertEqual(reviewed["status"], "mechanically_valid")
        evidence = reviewed["output"]["decisions"][0]["source_evidence"][0]
        self.assertNotIn("quotation", evidence)
        self.assertEqual(evidence["quotations"][0]["text"], self.game["sources"][0]["pages"][0]["lines"][0]["text"])
        self.assertEqual(reviewed["raw_response"]["decisions"][0]["source_evidence"][0]["quotation"], "Untrusted model-provided quotation.")

    def test_compact_review_accepts_citation_superset_and_materializes_quotes(self):
        output = module_output()
        output["findings"][0]["citations"][0]["lines"] = [1, 2]
        candidate = bam.ingest_response(self.module_requests()[0], output, self.game, "student")
        request = bam.build_review_request(self.game, candidate, "independent")
        self.assertEqual(request["review_schema_version"], 4)
        self.assertNotIn("attribute_checks", json.dumps(request["output_schema"]))
        response = {"decisions": [review_decision(output["findings"][0])], "report_verdict": "supported", "report_reason": "The report preserves scope and uncertainty."}
        review = bam.ingest_response(request, response, self.game, "independent")
        self.assertEqual(review["status"], "mechanically_valid", review["errors"])
        decision = review["output"]["decisions"][0]
        self.assertTrue(decision["citation_check"]["complete"])
        self.assertEqual(decision["required_citations"][0]["quotations"][0]["text"], self.game["sources"][0]["pages"][0]["lines"][0]["text"])
        self.assertEqual(bam.apply_semantic_review(candidate, review)["status"], "approved")

    def test_truncated_selection_expands_bounded_budget_and_preserves_partial_audit(self):
        request = self.requests[0]
        partial = '{"findings": [{"fact": "Partial source reading'
        first = bam.ingest_response(request, partial, self.game, "student", finish_reason="length")
        repair = bam.build_repair_request(self.game, request, first)
        self.assertEqual(repair["max_tokens"], 6000)
        self.assertEqual(repair["messages"][:2], request["messages"])
        self.assertNotIn(partial, json.dumps(repair["messages"]))
        self.assertEqual(first["raw_response"], partial)
        self.assertEqual(repair["repair_audit"]["raw_response_hash"], bam.value_hash(partial))
        self.assertEqual(repair["parent_hashes"][-1], first["artifact_hash"])
        second = bam.ingest_response(repair, partial, self.game, "student", finish_reason="length")
        self.assertEqual(second["generation_repair"]["previous_max_tokens"], 3000)
        final = bam.build_repair_request(self.game, repair, second, attempt=2)
        self.assertEqual(final["max_tokens"], 8192)
        self.assertEqual(final["generation_repair"]["rejected_artifact_hash"], second["artifact_hash"])
        third = bam.ingest_response(final, partial, self.game, "student", finish_reason="length")
        with self.assertRaisesRegex(ValueError, "8192-token output cap"):
            bam.build_repair_request(self.game, final, third, attempt=3, max_attempts=3)
        malformed = bam.ingest_response(request, "malformed", self.game, "student", finish_reason="stop")
        self.assertEqual(bam.build_repair_request(self.game, request, malformed)["max_tokens"], 3000)

    def test_placeholder_selection_ids_do_not_discard_supported_material(self):
        proposal = [finding(ident="unique string"), finding(bam.MODULES[1], ident="unique string")]
        raw = {"reviewed_chunk_id": self.requests[0]["chunk_id"], "findings": proposal, "disposition": "All supplied evidence reviewed."}
        first = bam.ingest_response(self.requests[0], raw, self.game, "student")
        second = bam.ingest_response(self.requests[0], raw, self.game, "student")
        self.assertEqual(len(first["output"]["findings"]), 2)
        self.assertFalse(first["output"]["rejected_findings"])
        ids = [f["id"] for f in first["output"]["findings"]]
        self.assertEqual(ids, [self.requests[0]["chunk_id"] + "-f1", self.requests[0]["chunk_id"] + "-f2"])
        self.assertEqual(ids, [f["id"] for f in second["output"]["findings"]])
        self.assertEqual([f["id"] for f in first["raw_response"]["findings"]], ["unique string", "unique string"])

    def test_module_and_gold_ids_are_readable_code_owned_and_reviews_stay_strict(self):
        raw = module_output()
        raw["findings"] = [finding(ident="unique string"), finding(ident="unique string")]
        candidate = bam.ingest_response(self.module_requests()[0], raw, self.game, "student")
        self.assertEqual([f["id"] for f in candidate["output"]["findings"]], ["availability_rotation-f1", "availability_rotation-f2"])
        self.assertEqual(candidate["raw_response"]["findings"][0]["id"], "unique string")
        request = bam.build_review_request(self.game, candidate, "independent")
        response = {"decisions": [review_decision(f) for f in candidate["output"]["findings"]], "report_verdict": "supported", "report_reason": "Scope preserved."}
        for foreign_id in ("unique string", "matchup-f1", "availability_rotation-f3"):
            bad = copy.deepcopy(response); bad["decisions"][0]["id"] = foreign_id
            self.assertEqual(bam.ingest_response(request, bad, self.game, "independent")["status"], "rejected")
        missing = copy.deepcopy(response); missing["decisions"].pop()
        self.assertEqual(bam.ingest_response(request, missing, self.game, "independent")["status"], "rejected")
        self.assertEqual(bam.ingest_response(request, response, self.game, "independent")["status"], "mechanically_valid")
        gold_request = bam.build_gold_request(self.game, self.modules(), self.postmortem())
        gold_raw = {"modules": [module_output(m) for m in bam.MODULES]}
        for module in gold_raw["modules"]: module["findings"][0]["id"] = "unique string"
        gold = bam.ingest_response(gold_request, gold_raw, self.game, "teacher")
        self.assertEqual([f["id"] for f in bam.all_findings(gold["output"])], [m + "-f1" for m in bam.MODULES])

    def test_canonical_tables_survive_paraphrase_rejections_once_across_chunks(self):
        g = copy.deepcopy(self.game)
        g["sources"][0]["pages"][0]["lines"] = [
            {"line": 1, "text": "PLAYER / 2024-25 REGULAR SEASON AVERAGES"},
            {"line": 2, "text": "#5 Sam Merrill"},
            {"line": 3, "text": "GP/GS PPG RPG APG SPG BPG MPG"},
            {"line": 4, "text": "26/2 7.3 1.8 1.8 0.7 0.2 19.9"}]
        table = next(t for t in bam.code_tables(g)[0] if t.get("decoder") == "player_season_averages_v1")
        requests = bam.build_selection_requests(g, max_chunk_chars=80)
        artifacts = []
        for request in requests:
            paraphrase = finding(); paraphrase.update(fact="Sam Merrill has played in 26 games and started 2 games this season.", time_scope="season_to_date", citations=table["citations"], decoded_table_id=table["id"])
            raw = {"reviewed_chunk_id":request["chunk_id"], "findings":[paraphrase], "disposition":"All supplied lines reviewed."}
            artifact = bam.ingest_response(request, raw, g, "student")
            self.assertEqual(artifact["status"], "mechanically_valid", artifact["errors"])
            self.assertIn("remain exact", artifact["output"]["rejected_findings"][0]["reason"])
            artifacts.append(artifact)
        tables = [f for a in artifacts for f in a["output"]["findings"] if f.get("decoded_table_id")]
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["fact"], table["fact"])
        self.assertEqual(tables[0]["time_scope"], table["time_scope"])
        self.assertEqual(tables[0]["uncertainty"], table["uncertainty"])
        self.assertEqual(bam._citation_keys(tables[0]["citations"]), bam._citation_keys(table["citations"]))
        module_requests = bam.build_module_requests(g, artifacts, max_chunk_chars=80)
        selected = [f for r in module_requests for f in json.loads(r["messages"][1]["content"])["selected_findings"]]
        self.assertEqual(sum(f.get("decoded_table_id") == table["id"] for f in selected), 1)

    def test_valid_table_proposal_does_not_duplicate_or_certify_model_interpretation(self):
        g = copy.deepcopy(self.game)
        g["sources"][0]["pages"][0]["lines"] = [
            {"line":1, "text":"2024-25 REGULAR SEASON PPG OPP. PPG FG% 3FG% FT% RPG APG SPG TO BPG"},
            {"line":2, "text":"Cavaliers 121.8 110.2 .504 .406 .775 43.5 28.5 9.13 13.2 4.43"}]
        table = bam.code_tables(g)[0][0]
        f = finding(bam.MODULES[1]); f.update(fact=table["fact"], time_scope=table["time_scope"], citations=table["citations"], decoded_table_id=table["id"], interpretation="A model claim that has not been semantically reviewed.")
        request = bam.build_selection_requests(g)[0]
        artifact = bam.ingest_response(request, {"reviewed_chunk_id":request["chunk_id"], "findings":[f, f], "disposition":"Read."}, g, "student")
        self.assertEqual(len(artifact["output"]["findings"]), 1)
        self.assertNotIn("A model claim", artifact["output"]["findings"][0]["interpretation"])
        audit = artifact["output"]["table_evidence_audit"]
        self.assertEqual(len(audit["model_proposals"]), 2)
        self.assertEqual(audit["canonical_tables"][0], table)
        self.assertEqual(artifact["raw_response"]["findings"][0]["interpretation"], f["interpretation"])

    def test_compact_review_withholds_supported_when_required_line_missing(self):
        candidate = bam.ingest_response(self.module_requests()[0], module_output(), self.game, "student")
        request = bam.build_review_request(self.game, candidate, "independent")
        decision = review_decision(candidate["output"]["findings"][0])
        decision["required_citations"][0]["lines"] = [1, 2]
        response = {"decisions": [decision], "report_verdict": "supported", "report_reason": "No omissions found."}
        review = bam.ingest_response(request, response, self.game, "independent")
        self.assertEqual(review["status"], "mechanically_valid", review["errors"])
        checked = review["output"]["decisions"][0]
        self.assertEqual(checked["model_verdict"], "supported")
        self.assertEqual(checked["verdict"], "unclear")
        self.assertEqual(checked["citation_check"]["missing_required_citations"], [{"source_id": "s1", "page": 1, "lines": [2]}])
        self.assertEqual(review["raw_response"]["decisions"][0]["verdict"], "supported")
        self.assertEqual(bam.apply_semantic_review(candidate, review)["status"], "rejected")

    def test_compact_review_containment_uses_source_and_page_not_only_line(self):
        g = copy.deepcopy(self.game)
        g["sources"][0]["pages"][1]["lines"] = [{"line": 1, "text": "Different player status in another section."}]
        candidate = bam.ingest_response(bam.build_module_requests(g, [bam.ingest_response(r, {'reviewed_chunk_id': r['chunk_id'], 'findings': [], 'disposition': 'read'}, g, 'student') for r in bam.build_selection_requests(g)])[0], module_output(), g, "student")
        request = bam.build_review_request(g, candidate, "independent")
        payload = json.loads(request["messages"][1]["content"])
        payload["pregame_source_context"][0]["pages"].append(g["sources"][0]["pages"][1])
        request["messages"][1]["content"] = json.dumps(payload)
        decision = review_decision(candidate["output"]["findings"][0])
        decision["required_citations"][0]["page"] = 2
        review = bam.ingest_response(request, {"decisions": [decision], "report_reason": "Scoped.", "report_verdict": "supported"}, g, "independent")
        self.assertEqual(review["output"]["decisions"][0]["verdict"], "unclear")

    def test_compact_review_rejects_unseen_evidence_and_missing_report_verdict(self):
        g = copy.deepcopy(self.game)
        g["sources"][0]["pages"][1]["lines"] = [{"line": 1, "text": "Not provided to this review."}]
        request = self.module_requests()[0]
        request["source_hash"] = bam.value_hash(g["sources"])
        candidate = bam.ingest_response(request, module_output(), g, "student")
        review_request = bam.build_review_request(g, candidate, "independent")
        decision = review_decision(candidate["output"]["findings"][0])
        decision["required_citations"][0]["page"] = 2
        review = bam.ingest_response(review_request, {"decisions": [decision], "report_reason": "Scoped.", "report_verdict": "supported"}, g, "independent")
        self.assertEqual(review["status"], "rejected")
        self.assertIn("not supplied", review["errors"][0])
        review = bam.ingest_response(review_request, {"decisions": [review_decision(candidate["output"]["findings"][0])], "report_reason": "Scoped."}, g, "independent")
        self.assertEqual(review["status"], "rejected")

    def test_compact_review_keeps_rejections_and_whole_report_gate(self):
        candidate = bam.ingest_response(self.module_requests()[0], module_output(), self.game, "student")
        request = bam.build_review_request(self.game, candidate, "independent")
        for claim_verdict, report_verdict in [("unsupported", "supported"), ("supported", "unclear")]:
            decision = review_decision(candidate["output"]["findings"][0]); decision["verdict"] = claim_verdict
            review = bam.ingest_response(request, {"decisions": [decision], "report_reason": "Material uncertainty remains.", "report_verdict": report_verdict}, self.game, "independent")
            self.assertEqual(bam.apply_semantic_review(candidate, review)["status"], "rejected")
            self.assertEqual(review["output"]["decisions"][0]["verdict"], claim_verdict)

    def test_html_sections_keep_labels_through_selection_module_and_review(self):
        g = copy.deepcopy(self.game)
        source = g["sources"][0]
        source.update(source_kind="html", pagination="article_sections", url="https://example.com/article")
        source["pages"][0].update(page_kind="article_section", section=1, label="availability")
        selection = bam.build_selection_requests(g)
        payload = json.loads(selection[0]["messages"][1]["content"])
        self.assertEqual(payload["source"]["source_kind"], "html")
        self.assertEqual(payload["label"], "availability")
        artifacts = [bam.ingest_response(r, {"reviewed_chunk_id":r["chunk_id"], "findings":[finding()] if r["coverage"]["lines"] else [], "disposition":"Article section reviewed."}, g, "student") for r in selection]
        request = bam.build_module_requests(g, artifacts)[0]
        candidate = bam.ingest_response(request, module_output(), g, "student")
        review = bam.build_review_request(g, candidate, "independent", original_request=request)
        supplied = json.loads(review["messages"][1]["content"])
        self.assertEqual(supplied["pregame_source_context"][0]["pagination"], "article_sections")
        self.assertEqual(supplied["pregame_source_context"][0]["pages"][0]["page_kind"], "article_section")
        self.assertEqual(supplied["pregame_source_context"][0]["pages"][0]["label"], "availability")
        self.assertEqual(candidate["output"]["findings"][0]["citations"][0]["section"], 1)

    def test_html_body_preserves_original_title_and_dated_heading_sections(self):
        g = copy.deepcopy(self.game); source = g["sources"][0]
        source.update(source_kind="html", pagination="article_sections")
        source["pages"] = [
            {"page":1, "section":1, "page_kind":"article_section", "label":"title", "lines":[{"line":1, "text":"January 2, 2025 game preview"}]},
            {"page":2, "section":2, "page_kind":"article_section", "label":"heading", "lines":[{"line":1, "text":"LAST GAME — January 1, 2025"}]},
            {"page":3, "section":3, "page_kind":"article_section", "label":"paragraph", "lines":[{"line":1, "text":"Brown did not play."}],
             "context_sections":[{"section":1, "label":"title", "lines":[{"line":1, "text":"January 2, 2025 game preview"}]},
                                 {"section":2, "label":"heading", "lines":[{"line":1, "text":"LAST GAME — January 1, 2025"}]}]}]
        requests = bam.build_selection_requests(g)
        body = next(r for r in requests if r["coverage"]["page"] == 3)
        payload = json.loads(body["messages"][1]["content"])
        self.assertEqual([p["page"] for p in payload["context_sections"]], [1, 2])
        self.assertEqual(payload["lines"], [{"line":1, "text":"Brown did not play."}])
        f = finding(); f.update(fact="Brown did not play in the prior game.", fact_date="2025-01-01", time_scope="prior_game", citations=[{"source_id":"s1", "page":3, "lines":[1]}, {"source_id":"s1", "page":2, "lines":[1]}])
        artifact = bam.ingest_response(body, {"reviewed_chunk_id":body["chunk_id"], "findings":[f], "disposition":"Prior-game body plus original heading reviewed."}, g, "student")
        self.assertEqual(artifact["status"], "mechanically_valid")
        context = bam._evidence_for(g, artifact["output"]["findings"])
        self.assertEqual({p["page"] for p in context[0]["pages"]}, {1, 2, 3})
        review_request = bam.build_review_request(g, artifact, "independent", original_request=body)
        decision = {"id":artifact["output"]["findings"][0]["id"], "source_reading":"A dated prior-game absence, not a current status.", "required_citations":f["citations"], "reason":"The preceding heading supplies the prior-game scope.", "verdict":"supported"}
        review = bam.ingest_response(review_request, {"decisions":[decision], "report_reason":"Scope limited to prior game.", "report_verdict":"supported"}, g, "independent")
        self.assertEqual(review["status"], "mechanically_valid", review["errors"])
        broken = copy.deepcopy(g)
        broken["sources"][0]["pages"][2]["context_sections"][1]["lines"][0]["text"] = "CURRENT GAME — January 2, 2025"
        with self.assertRaisesRegex(ValueError, "differs from canonical"):
            bam.build_selection_requests(broken)

    def test_actual_style_comma_totals_cannot_become_averages(self):
        g = copy.deepcopy(self.game)
        g['sources'][0]['pages'][0]['lines'] = [
            {'line': 1, 'text': 'PITP 2ND PTS FB PTS BENCH PTS PTS TO'},
            {'line': 2, 'text': 'TOTALS 3,044 870 1,106 2,280 1,118'}]
        f = finding(bam.MODULES[1]); f.update(fact='The Pistons average 2,280 bench points.', time_scope='season_to_date', citations=[{'source_id':'s1','page':1,'lines':[1,2]}])
        with self.assertRaisesRegex(ValueError, 'Dense tables'):
            bam._validate_finding(g, f)
        g['sources'][0]['pages'][0]['lines'].append({'line':3, 'text':'The Pistons averaged 38.6 bench points per game this season.'})
        f.update(fact='The Pistons averaged 38.6 bench points per game this season.', citations=[{'source_id':'s1','page':1,'lines':[2,3]}])
        self.assertEqual(bam._validate_finding(g, f)['fact'], f['fact'])

    def test_yearless_milestone_cannot_acquire_metadata_or_prose_year(self):
        g = copy.deepcopy(self.game)
        g['sources'][0]['pages'][0]['lines'] = [{'line':1,'text':'Mitchell made his 1,400th career three-pointer against the Lakers on Oct. 30.'}]
        f = finding(bam.MODULES[2]); f.update(fact='Mitchell made his 1,400th career three-pointer against the Lakers on Oct. 30.', time_scope='historical')
        for year in (2023,2024):
            f['fact_date'] = f'{year}-10-30'
            with self.assertRaisesRegex(ValueError, 'fact_date is not grounded'): bam._validate_finding(g, f)
        f['fact_date'] = None
        self.assertEqual(bam._validate_finding(g, f)['fact'], f['fact'])
        for field in ('fact','interpretation','counterevidence','uncertainty'):
            bad = copy.deepcopy(f); bad[field] += ' The milestone occurred on October 30, 2023.'
            with self.assertRaisesRegex(ValueError, 'Full date in prose'): bam._validate_finding(g, bad)
        bad = copy.deepcopy(f); bad['fact'] += ' This milestone was in 2024.'
        with self.assertRaisesRegex(ValueError, 'Numerical fact absent'): bam._validate_finding(g, bad)

    def test_explicit_cited_season_resolves_correct_year_only(self):
        g = copy.deepcopy(self.game)
        g['sources'][0]['pages'][0]['lines'] = [{'line':1,'text':'2024-25 SEASON MILESTONES'}, {'line':2,'text':'Brown returned on January 1.'}]
        f = finding(bam.MODULES[2]); f.update(fact='Brown returned on January 1, 2025.', fact_date='2025-01-01', time_scope='prior_game', citations=[{'source_id':'s1','page':1,'lines':[1,2]}])
        self.assertEqual(bam._validate_finding(g, f)['fact_date'], '2025-01-01')
        f['fact_date'] = '2024-01-01'
        with self.assertRaisesRegex(ValueError, 'fact_date is not grounded'): bam._validate_finding(g, f)

    def test_repairs_keep_original_evidence_and_only_latest_compact_feedback(self):
        request = self.module_requests()[0]
        rejected = bam.ingest_response(request, module_output(), self.game, 'student')
        rejected.update(status='rejected', errors=['Independent review rejected a claim.'], semantic_review_hash='review-one')
        rejected['semantic_review'] = {'decisions':[{'id':'availability_rotation-f1','verdict':'unsupported','reason':'first reason','source_reading':'First source reading.', 'required_citations':[{'source_id':'s1','page':1,'lines':[1], 'quotations':[{'line':1,'text':'DUPLICATED_QUOTE'*5000}], 'url':'duplicated provenance'}]}], 'report_verdict':'unsupported','report_reason':'First report reason.'}
        rejected['artifact_hash'] = bam.value_hash(rejected)
        repair = bam.build_repair_request(self.game, request, rejected)
        self.assertEqual(repair['messages'][:2], request['messages'])
        self.assertNotIn('DUPLICATED_QUOTE', json.dumps(repair['messages']))
        self.assertNotIn('duplicated provenance', json.dumps(repair['messages']))
        feedback = json.loads(repair['messages'][2]['content'])
        self.assertEqual(feedback['latest_candidate']['findings'][0]['citations'], [{'source_id':'s1','page':1,'lines':[1]}])
        second = copy.deepcopy(rejected); second.update(request_id=repair['request_id'], artifact_hash='second-rejection', semantic_review_hash='review-two')
        second['semantic_review']['decisions'][0]['reason'] = 'latest reason'
        second['semantic_review']['report_reason'] = 'Latest report reason.'
        again = bam.build_repair_request(self.game, repair, second, attempt=2)
        self.assertEqual(len(again['messages']), 3)
        self.assertEqual(again['messages'][:2], request['messages'])
        self.assertNotIn('first reason', json.dumps(again['messages']))
        self.assertIn('latest reason', json.dumps(again['messages']))
        self.assertTrue({rejected['artifact_hash'],'second-rejection','review-one','review-two'} <= set(again['parent_hashes']))
        self.assertEqual(again['repair_audit']['original_evidence_messages_hash'], bam.value_hash(request['messages']))

    def test_selection_repair_feedback_keeps_real_parent_and_fact_association(self):
        bad = finding(); bad['citations'][0]['lines'] = [999]
        r = self.requests[0]
        artifact = bam.ingest_response(r, {'reviewed_chunk_id':r['chunk_id'], 'findings':[finding(),bad], 'disposition':'Read all lines.'}, self.game, 'student')
        repair = bam.build_selection_repair_request(self.game, r, artifact)
        self.assertEqual(repair['parent_hashes'], r['parent_hashes'] + [artifact['artifact_hash']])
        feedback = json.loads(repair['messages'][2]['content'])
        self.assertEqual(feedback['rejected_findings'][0]['candidate']['fact'], bad['fact'])
        self.assertEqual(feedback['rejected_findings'][0]['candidate']['citations'][0]['lines'], [999])
        self.assertIn('absent', feedback['rejected_findings'][0]['reason'])
        self.assertEqual(len(feedback['latest_candidate']['findings']), 1)

    def test_malformed_review_repair_does_not_inject_raw_unvalidated_text(self):
        candidate = bam.ingest_response(self.module_requests()[0], module_output(), self.game, 'student')
        request = bam.build_review_request(self.game, candidate, 'independent')
        raw = 'UNVALIDATED_REVIEW' * 10000
        rejected = bam.ingest_response(request, raw, self.game, 'independent')
        repair = bam.build_repair_request(self.game, request, rejected)
        self.assertEqual(repair['messages'][:2], request['messages'])
        self.assertNotIn('UNVALIDATED_REVIEW', json.dumps(repair['messages']))
        self.assertEqual(repair['parent_hashes'][0], candidate['artifact_hash'])
        self.assertEqual(rejected['raw_response'], raw)

    def test_source_numbers_do_not_excuse_reversed_literal_comparison(self):
        self.game['sources'][0]['pages'][0]['lines'][0]['text'] = 'Miami allowed 109.9 points per game; the comparison team allowed 110.8 points per game.'
        candidate = finding(bam.MODULES[1])
        candidate.update(fact="Miami had a higher opponent points per game (109.9) compared to the opponent's (110.8).",
                         interpretation='The comparison may matter for this matchup.',
                         counterevidence='The teams faced different opponents.', uncertainty='Future scoring remains uncertain.',
                         time_scope='season_to_date')
        with self.assertRaisesRegex(ValueError, 'Literal numerical comparison'):
            bam._validate_finding(self.game, candidate)
        candidate['fact'] = candidate['fact'].replace('higher', 'lower')
        self.assertEqual(bam._validate_finding(self.game, candidate)['fact'], candidate['fact'])

    def test_source_warnings_are_audit_metadata_not_citable_source_lines(self):
        source = self.game['sources'][0]
        source['source_warnings'] = ['Literal PST differs from verified schedule PDT.', 'Unassigned probable clause; do not infer a player.']
        request = bam.build_selection_requests(self.game)[0]
        metadata = json.loads(request['messages'][1]['content'])['source']
        self.assertEqual(metadata['source_warnings'], source['source_warnings'])
        self.assertIn('not canonical source statements', metadata['source_warnings_scope'])
        citation = bam.canonical_citations(self.game, [{'source_id':'s1','page':1,'lines':[1]}])[0]
        self.assertEqual([q['text'] for q in citation['quotations']], [source['pages'][0]['lines'][0]['text']])
        with self.assertRaisesRegex(ValueError, 'absent'):
            bam.canonical_citations(self.game, [{'source_id':'s1','page':0,'lines':[1]}])
        source['source_warnings'] = 'unstructured warning'
        with self.assertRaisesRegex(ValueError, 'audit warnings'):
            bam.build_selection_requests(self.game)

    def test_threshold_precision_repair_retains_useful_supported_statement(self):
        self.game['sources'][0]['pages'][0]['lines'][0]['text'] = 'Cunningham is among players averaging at least 20 points, six rebounds and eight assists.'
        candidate = finding(bam.MODULES[1])
        candidate.update(fact='Cunningham averages 20 points, six rebounds and eight assists per game.',
                         interpretation='His contributions span several categories.', counterevidence='Recent workload is not established here.',
                         uncertainty='Future performance is uncertain.', time_scope='season_to_date')
        with self.assertRaisesRegex(ValueError, 'lower bounds'):
            bam._validate_finding(self.game, candidate)
        candidate['fact'] = candidate['fact'].replace('averages 20', 'averages at least 20')
        self.assertEqual(bam._validate_finding(self.game, candidate)['fact'], candidate['fact'])

    def test_sparse_page_flag_is_readability_uncertainty_not_invented_numbers(self):
        page = self.game['sources'][0]['pages'][1]
        page.update(flags=['sparse_or_empty_text_needs_visual_review'], method='pypdf_layout')
        request = bam.build_selection_requests(self.game)[1]
        payload = json.loads(request['messages'][1]['content'])
        self.assertEqual(payload['flags'], page['flags'])
        self.assertEqual(payload['method'], 'pypdf_layout')
        self.assertEqual(payload['page_coverage_gaps'][0]['classification'], 'sparse_extraction_uncertain')
        self.assertNotIn('numeric_image_untranscribed', json.dumps(payload))
        artifact = bam.ingest_response(request, {'reviewed_chunk_id':request['chunk_id'], 'findings':[], 'disposition':'No relevant information.'}, self.game, 'student')
        self.assertEqual(len(artifact['output']['coverage_unknowns']), 1)
        self.assertIn('does not establish readable content or irrelevance', artifact['output']['coverage_unknowns'][0])

    def test_verified_numeric_image_gap_survives_selection_module_review_and_gold(self):
        source = self.game['sources'][0]; page = source['pages'][1]
        page.update(flags=['sparse_or_empty_text_needs_visual_review'], method='pypdf_layout', lines=[{'line':1,'text':'GAME NOTES PAGE 2'}])
        page['coverage_audit'] = {'game_id':'g1', 'source_id':'s1', 'source_sha256':source['sha256'], 'page':2,
            'classification':'numeric_image_untranscribed', 'mandatory_unknown':True,
            'coverage_note':'Verified image-only numeric table; no transcription admitted.', 'text_sha256':'b'*64, 'review_image_sha256':'c'*64}
        self.requests = bam.build_selection_requests(self.game)
        payload = json.loads(self.requests[1]['messages'][1]['content'])
        self.assertIn('not canonical source text', payload['coverage_audit']['scope'])
        self.assertTrue(any(f['page']==2 and f.get('origin')=='verified_numeric_image_coverage_audit' for f in payload['undecoded_tables']))
        requests = self.module_requests()
        self.assertTrue(all(json.loads(r['messages'][1]['content'])['page_coverage_gaps'] for r in requests))
        raw = module_output(requests[0]['module'])
        candidate = bam.ingest_response(requests[0], raw, self.game, 'student')
        unknown = bam.page_coverage_gaps(self.game)[0]['unknown']
        self.assertIn(unknown, candidate['output']['unknowns'])
        self.assertNotIn(unknown, candidate['raw_response']['unknowns'])
        review = bam.build_review_request(self.game, candidate, 'teacher', original_request=requests[0])
        self.assertEqual(json.loads(review['messages'][1]['content'])['page_coverage_gaps'][0]['unknown'], unknown)
        gold_request = bam.build_gold_request(self.game, self.modules(), self.postmortem())
        self.assertEqual(json.loads(gold_request['messages'][1]['content'])['page_coverage_gaps'][0]['unknown'], unknown)
        gold = bam.ingest_response(gold_request, {'modules':[module_output(m) for m in bam.MODULES]}, self.game, 'teacher')
        self.assertEqual(gold['status'], 'mechanically_valid', gold['errors'])
        self.assertTrue(all(unknown in m['unknowns'] for m in gold['output']['modules']))
        with self.assertRaisesRegex(ValueError, 'absent'):
            bam.canonical_citations(self.game, [{'source_id':'s1','page':2,'lines':[99]}])
        page['coverage_audit']['source_sha256'] = 'wrong source'
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            bam.build_selection_requests(self.game)

    def test_postmortem_input_preserves_every_source_line_without_pointer_duplication(self):
        records = postgame(); source = records['sources'][0]
        source['source_kind'] = 'boxscore'
        source['pages'][0]['label'] = 'Normalized box score'
        source['pages'][0]['lines'][0]['source_pointer'] = '/boxscore/players/0/statistics/0'
        source['pages'].append({'page':2, 'label':'Recap', 'lines':[{'line':5,'text':'Second normalized observation.', 'source_pointer':'/news/0/story'}]})
        original = copy.deepcopy(records)
        request = bam.build_postmortem_request(self.game, records)
        model_source = json.loads(request['messages'][1]['content'])['normalized_postgame_sources'][0]
        self.assertEqual([(p['page'],l['line'],l['text']) for p in model_source['pages'] for l in p['lines']],
                         [(p['page'],l['line'],l['text']) for p in source['pages'] for l in p['lines']])
        self.assertEqual(model_source['pages'][0]['label'], 'Normalized box score')
        self.assertNotIn('source_pointer', json.dumps(request['messages']))
        self.assertEqual(records, original)

    def test_gold_compacts_postmortem_citations_without_dropping_any_content(self):
        post = self.postmortem(); post['output']['general'].append(copy.deepcopy(post['output']['general'][0]))
        post['output']['general'][1]['unforeseeable'] = 'A distinct uncertainty that must survive.'
        post['output']['timeline'].append('A second timeline detail.')
        original = copy.deepcopy(post)
        request = bam.build_gold_request(self.game, self.modules(), post)
        compact = json.loads(request['messages'][1]['content'])['training_only_postmortem']
        self.assertEqual(request['max_tokens'], 8192)
        self.assertEqual(compact['timeline'], post['output']['timeline'])
        self.assertEqual(compact['market_view'], post['output']['market_view'])
        self.assertEqual(len(compact['general']), len(post['output']['general']))
        for before, after in zip(post['output']['general'], compact['general']):
            for key in ('observation','foreseeable_signal','unforeseeable'):
                self.assertEqual(before[key], after[key])
            self.assertEqual(after['citations'], [{k:c[k] for k in ('source_id','page','lines')} for c in before['citations']])
        self.assertNotIn('quotations', json.dumps(compact))
        self.assertIn('quotations', post['output']['general'][0]['citations'][0])
        self.assertEqual(post, original)
        bad = copy.deepcopy(post['output']); bad['extra_raw_payload'] = {'unsafe':'extra'}
        with self.assertRaisesRegex(ValueError, 'Unexpected postmortem fields'):
            bam._compact_postmortem(bad)
        req = bam.build_postmortem_request(self.game, postgame())
        self.assertEqual(bam.ingest_response(req, bad, self.game, 'teacher')['status'], 'rejected')


if __name__ == '__main__':
    unittest.main()
