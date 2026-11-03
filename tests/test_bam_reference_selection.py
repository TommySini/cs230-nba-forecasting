import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import bam_reference_selection as refs
import bam_research as research


def game():
    return {'game_id':'g1','game_date_et':'2025-01-02','cutoff_utc':'2025-01-02T23:00:00Z',
            'tipoff_utc':'2025-01-03T00:00:00Z','home_abbr':'BOS','away_abbr':'MIA','split':'train',
            'home_win':1,'recap':'NEVER INCLUDE OUTCOME',
            'sources':[{'source_id':'s1','publisher':'BOS','url':'https://example.com/notes.pdf','sha256':'a'*64,
                'archive_time':'2025-01-02T19:00:00Z','pages':[{'page':1,'lines':[
                    {'line':1,'text':'2024-25 SEASON'},
                    {'line':2,'text':'Brown is questionable tonight.'},
                    {'line':3,'text':'A prior DNP does not establish current absence.'},
                    {'line':4,'text':'He averages at least 20 points, six rebounds and eight assists.'}]},
                    {'page':2,'lines':[]}]}]}


def answer(request, selections=None):
    return {'reviewed_chunk_id':request['chunk_id'], 'selections':selections or [], 'unresolved':[],
            'disposition':'references_selected' if selections else 'no_selection'}


class ReferenceSelectionTests(unittest.TestCase):
    def test_every_line_and_empty_page_are_accounted_without_outcomes(self):
        g=game(); requests=refs.build_reference_selection_requests(g,max_chunk_chars=80)
        observed=[(r['coverage']['page'],n) for r in requests for n in r['coverage']['lines']]
        self.assertEqual(observed,[(1,1),(1,2),(1,3),(1,4)])
        self.assertEqual(requests[-1]['coverage'],{'source_id':'s1','page':2,'lines':[]})
        self.assertNotIn('home_win',json.dumps([r['messages'] for r in requests]))
        self.assertNotIn('NEVER INCLUDE OUTCOME',json.dumps([r['messages'] for r in requests]))

    def test_materializes_exact_text_not_summary_truth(self):
        g=game(); r=refs.build_reference_selection_requests(g)[0]
        raw=answer(r,[{'module':'availability_rotation','topic':'availability','spans':[[2,3]]}])
        a=refs.ingest_reference_selection(r,raw,g,'selector')
        self.assertEqual(a['status'],'mechanically_valid',a['errors'])
        block=a['output']['reference_selections'][0]
        self.assertEqual(block['verbatim_excerpt'],g['sources'][0]['pages'][0]['lines'][1:3])
        self.assertNotIn('fact',block);self.assertNotIn('fact_date',block)
        self.assertIn('meaning and relevance require',block['status'])
        self.assertEqual(a['output']['unselected_line_ranges'],[[1,1],[4,4]])
        self.assertEqual(a['raw_response'],raw)

    def test_bad_references_and_factual_fields_are_audited_without_losing_siblings(self):
        g=game();r=refs.build_reference_selection_requests(g)[0]
        good={'module':'matchup','topic':'other_relevant_context','spans':[[4,4]]}
        raw=answer(r,[good,{**good,'spans':[[999,999]]},{**good,'fact':'He averages exactly20/6/8.'},
                      {**good,'topic':'scored40points'},{**good,'spans':[[True,3]]}])
        a=refs.ingest_reference_selection(r,raw,g,'selector')
        self.assertEqual(a['status'],'mechanically_valid')
        self.assertEqual(len(a['output']['reference_selections']),1)
        self.assertEqual(len(a['output']['rejected_references']),4)
        self.assertEqual(a['output']['reference_selections'][0]['verbatim_excerpt'][0]['text'],g['sources'][0]['pages'][0]['lines'][3]['text'])

    def test_wrong_chunk_truncated_and_malformed_responses_never_complete(self):
        g=game();r=refs.build_reference_selection_requests(g)[0]
        for raw,finish in [('broken JSON','stop'),({**answer(r),'reviewed_chunk_id':'foreign'},'stop'),
                           (answer(r),'length'),({**answer(r),'fact':'A fabricated summary'},'stop')]:
            a=refs.ingest_reference_selection(r,raw,g,'selector',finish_reason=finish)
            self.assertEqual(a['status'],'rejected');self.assertEqual(a['raw_response'],raw)

    def test_canonical_tables_are_retained_once_even_when_selector_selects_nothing(self):
        g=game();g['sources'][0]['pages'][0]['lines']=[{'line':i,'text':text} for i,text in enumerate([
            'PLAYER / 2024-25 REGULAR SEASON AVERAGES','#5 Sam Merrill','GP/GS PPG RPG APG SPG BPG MPG',
            '26/2 7.3 1.8 1.8 0.7 0.2 19.9'],1)]
        requests=refs.build_reference_selection_requests(g,max_chunk_chars=80)
        artifacts=[refs.ingest_reference_selection(r,answer(r),g,'selector') for r in requests]
        tables=[f for a in artifacts for f in a['output']['canonical_table_findings']]
        self.assertEqual(len(tables),1)
        original=next(t for t in research.code_tables(g)[0] if t.get('decoder')=='player_season_averages_v1')
        self.assertEqual(tables[0]['fact'],original['fact'])
        modules=refs.build_reference_module_requests(g,artifacts,max_chunk_chars=80)
        self.assertEqual(sum(len(json.loads(r['messages'][1]['content'])['code_decoded_tables']) for r in modules),1)
        shown=next(t for r in modules for t in json.loads(r['messages'][1]['content'])['code_decoded_tables'])
        self.assertEqual(shown['id'],original['id'])
        self.assertEqual(shown['fact'],original['fact'])

    def test_full_coverage_and_artifact_integrity_required_for_module_inputs(self):
        g=game();requests=refs.build_reference_selection_requests(g)
        artifacts=[refs.ingest_reference_selection(r,answer(r),g,'selector') for r in requests]
        with self.assertRaisesRegex(ValueError,'Incomplete/duplicate'):
            refs.build_reference_module_requests(g,artifacts[:-1])
        corrupted=copy.deepcopy(artifacts);corrupted[0]['output']['disposition']='changed'
        with self.assertRaisesRegex(ValueError,'artifact changed'):
            refs.build_reference_module_requests(g,corrupted)

    def test_hierarchy_preserves_all_references_and_source_text_without_fact_cap(self):
        g=game();g['sources'][0]['pages']=[{'page':i,'lines':[{'line':1,'text':('Original source paragraph. '*25)+str(i)}]} for i in range(1,21)]
        requests=refs.build_reference_selection_requests(g)
        artifacts=[refs.ingest_reference_selection(r,answer(r,[{'module':'matchup','topic':'team_style','spans':[[1,1]]}]),g,'selector') for r in requests]
        modules=refs.build_reference_module_requests(g,artifacts,max_input_chars=9000)
        parts=[r for r in modules if r['module']=='matchup']
        self.assertGreater(len(parts),1)
        expected={u['evidence_id'] for a in artifacts for u in a['output']['reference_selections']}
        actual={u['evidence_id'] for r in parts for u in json.loads(r['messages'][1]['content'])['reference_selections']}
        self.assertEqual(actual,expected)
        self.assertTrue(all(sum(len(m['content']) for m in r['messages'])<=9000 for r in modules))
        visible={(p['page'],l['text']) for r in parts for s in json.loads(r['messages'][1]['content'])['source_context'] for p in s['pages'] for l in p['lines']}
        self.assertEqual(visible,{(p['page'],p['lines'][0]['text']) for p in g['sources'][0]['pages']})
        with self.assertRaises(ValueError):refs.build_reference_module_requests(g,artifacts,max_input_chars=1000)

    def test_sparse_unknown_and_unresolved_lines_survive_without_numeric_admission(self):
        g=game();g['sources'][0]['pages'][0]['flags']=['sparse_or_empty_text_needs_visual_review']
        requests=refs.build_reference_selection_requests(g);raw=answer(requests[0]);raw['unresolved']=[{'spans':[[4,4]],'reason':'ambiguous_time_scope'}]
        a=refs.ingest_reference_selection(requests[0],raw,g,'selector')
        self.assertTrue(a['output']['coverage_unknowns']);self.assertEqual(a['output']['unselected_line_ranges'],[[1,4]])
        others=[refs.ingest_reference_selection(r,answer(r),g,'selector') for r in requests[1:]]
        modules=refs.build_reference_module_requests(g,[a]+others)
        payload=json.loads(modules[0]['messages'][1]['content'])
        self.assertEqual(payload['coverage']['unresolved_references'][0]['spans'],[[4,4]])
        self.assertTrue(payload['page_coverage_gaps'])


if __name__=='__main__':unittest.main()
