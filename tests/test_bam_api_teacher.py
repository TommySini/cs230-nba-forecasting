import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import bam_api_teacher as api
import bam_research as research
import test_bam_research as fixture


def spec(model='teacher'):
    return {'provider':'openai','model':model,'context_window_tokens':65536,'max_output_tokens':8192,'model_release_date':'2026-01-01'}


def registry(game):
    return {'snapshot_manifest_sha256':'a'*64,'games':{game['game_id']:{k:game[k] for k in ('game_id','game_date_et','cutoff_utc','split')}}}


def rehash(request):
    request['request_id']=research.value_hash({k:v for k,v in request.items() if k!='request_id'})
    return request


class ApiTeacherTests(unittest.TestCase):
    def setUp(self):
        self.case=fixture.BamResearchTests();self.case.setUp()
        self.game=self.case.game;self.registry=registry(self.game)
        self.registry=api.bind_trusted_training_context(self.registry,[self.game],{})

    def trust(self, request):
        # Test fixture represents the independent trusted builder/store. API
        # transport itself must never create this authorization from candidates.
        authorizations=copy.deepcopy(self.registry.get('authorized_training_requests',{}))
        authorizations[request['request_id']]={k:request[k] for k in ('request_id','game_id','stage','source_hash')} | {'request_sha256':research.value_hash(request)}
        self.registry=api.bind_trusted_training_context(self.registry,[self.game],authorizations)

    def gold(self):
        request=research.build_gold_request(self.game,self.case.modules(),self.case.postmortem())
        output={'modules':[fixture.module_output(m) for m in research.MODULES]}
        candidate=research.ingest_response(request,output,self.game,api.api_identity(spec()))
        self.trust(request)
        return request,candidate

    def test_prediction_stages_and_falsely_labeled_heldout_games_rejected(self):
        request=research.build_postmortem_request(self.game,fixture.postgame())
        self.trust(request)
        bad_registry=copy.deepcopy(self.registry);bad_registry['games']['g1']['split']='test'
        with self.assertRaisesRegex(ValueError,'manifest-verified TRAIN'):
            api.api_payload(request,spec(),bad_registry)
        with self.assertRaisesRegex(ValueError,'Prediction-input stages'):
            api.api_payload(self.case.module_requests()[0],spec(),self.registry)

    def test_api_envelope_preserves_scientific_messages_and_hash_across_providers(self):
        request=research.build_postmortem_request(self.game,fixture.postgame())
        self.trust(request)
        first=api.api_payload(request,spec(),self.registry)
        second=api.api_payload(request,{**spec('other'),'provider':'anthropic'},self.registry)
        self.assertEqual(first['messages'],request['messages'])
        self.assertEqual(first['scientific_request_sha256'],second['scientific_request_sha256'])
        self.assertEqual(first['submission_status'],'offline_prepared_no_provider_call')
        with self.assertRaisesRegex(ValueError,'credentials'):
            api.api_payload(request,{**spec(),'api_key':'must-never-enter-config'},self.registry)
        with self.assertRaisesRegex(ValueError,'context'):
            api.api_payload(request,spec(),self.registry,input_token_estimate=65536)

    def test_raw_postgame_keys_rejected_and_gold_combination_required(self):
        request=research.build_postmortem_request(self.game,fixture.postgame())
        payload=json.loads(request['messages'][1]['content']);payload['normalized_postgame_sources'][0]['raw_espn']={}
        request['messages'][1]['content']=json.dumps(payload);rehash(request)
        self.trust(request)
        with self.assertRaisesRegex(ValueError,'raw postgame'):
            api.validate_training_request(request,self.registry)
        request,_=self.gold();payload=json.loads(request['messages'][1]['content']);payload['forecasting_context']['researcher_outputs'].pop()
        request['messages'][1]['content']=json.dumps(payload);rehash(request)
        self.trust(request)
        with self.assertRaisesRegex(ValueError,'three real researcher'):
            api.validate_training_request(request,self.registry)

    def test_gold_review_requires_intact_matching_target_and_no_hindsight(self):
        gold,candidate=self.gold();reviewer=spec('reviewer')
        request=research.build_review_request(self.game,candidate,api.api_identity(reviewer),original_request=gold)
        self.trust(request)
        targets={candidate['artifact_hash']:candidate}
        api.api_payload(request,reviewer,self.registry,targets=targets)
        wrong=copy.deepcopy(candidate);wrong['output']['modules'][0]['coverage_note']='tampered'
        with self.assertRaisesRegex(ValueError,'intact target'):
            api.validate_training_request(request,self.registry,targets={candidate['artifact_hash']:wrong})
        bad=copy.deepcopy(request);payload=json.loads(bad['messages'][1]['content']);payload['training_only_postmortem']={'secret':'future result'}
        bad['messages'][1]['content']=json.dumps(payload);rehash(bad)
        self.trust(bad)
        with self.assertRaisesRegex(ValueError,'outcome-blind'):
            api.validate_training_request(bad,self.registry,targets=targets)
        with self.assertRaisesRegex(ValueError,'reviewer differs'):
            api.api_payload(request,spec('different'),self.registry,targets=targets)

    def test_modern_teacher_does_not_relax_historical_prediction_checkpoints(self):
        student={'model_id':'student','model_revision':'a'*40,'model_revision_date':'2024-09-19'}
        critic={'model_id':'critic','model_revision':'b'*40,'model_revision_date':'2025-05-21'}
        plan=api.build_role_plan(student,critic,spec(),spec('reviewer'),minimum_eval_date='2025-12-01')
        self.assertEqual(plan['prediction_models']['critic'],critic)
        self.assertIn('Retrospective',plan['temporal_scope'])
        with self.assertRaisesRegex(ValueError,'Prediction-time checkpoint'):
            api.build_role_plan(student,{**critic,'model_revision_date':'2026-01-01'},spec(),spec('reviewer'),minimum_eval_date='2025-12-01')
        with self.assertRaisesRegex(ValueError,'independent'):
            api.build_role_plan(student,critic,spec(),spec(),minimum_eval_date='2025-12-01')

    def test_registry_never_needs_outcome_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'inputs.jsonl').write_text(json.dumps(self.game)+'\n')
            (root/'manifest.json').write_text(json.dumps({'artifact_sha256':{'inputs.jsonl':api.file_hash(root/'inputs.jsonl'),'labels_test.jsonl':'nonexistent'}}))
            self.assertEqual(api.load_train_registry(root)['games']['g1']['split'],'train')

    def test_frozen_comparison_bytes_and_labels_stay_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'original';source.mkdir();known=copy.deepcopy(self.registry);requests=[]
            for i in range(16):
                row=copy.deepcopy(self.game);row['game_id']=f'g{i}';known['games'][row['game_id']]={k:row[k] for k in ('game_id','game_date_et','cutoff_utc','split')}
                request={'game_id':row['game_id'],'split':'train','stage':'actual_claim_review','max_tokens':4500,
                         'messages':[{'role':'system','content':'Original fixed rubric'},{'role':'user','content':json.dumps({'game':research.metadata(row),'candidate':{'id':str(i)}})}]}
                requests.append(rehash(request))
            raw=('\n'.join(json.dumps(r) for r in requests)+'\n').encode();(source/'requests.jsonl').write_bytes(raw)
            (source/'expected.json').write_text(json.dumps([{'expected_verdict':'supported'}]*16))
            (source/'manifest.json').write_text(json.dumps({'artifacts':{n:api.file_hash(source/n) for n in ('requests.jsonl','expected.json')}}))
            digest=api.file_hash(source/'manifest.json')
            result=api.freeze_comparison_bundle(source,root/'bundle',known,expected_source_manifest_sha256=digest)
            self.assertTrue(result['provider_independent'])
            self.assertEqual((root/'bundle/requests.jsonl').read_bytes(),raw)
            self.assertNotIn('expected_verdict',raw.decode())
            self.assertTrue((root/'bundle/scoring/expected.json').exists())
            self.assertEqual(api.freeze_comparison_bundle(source,root/'bundle',known,expected_source_manifest_sha256=digest),result)

    def test_rehashed_source_stage_context_and_repair_are_not_self_authorizing(self):
        request=research.build_postmortem_request(self.game,fixture.postgame());self.trust(request)
        wrong=copy.deepcopy(request);wrong['source_hash']='forged-source';rehash(wrong)
        # Even an authorization-record mistake cannot change the independently
        # bound canonical source identity.
        self.trust(wrong)
        with self.assertRaisesRegex(ValueError,'source identity'):
            api.validate_training_request(wrong,self.registry)
        for mutation in ('stage','context','repair'):
            wrong=copy.deepcopy(request)
            if mutation=='stage':wrong['stage']='gold'
            elif mutation=='context':
                p=json.loads(wrong['messages'][1]['content']);p['normalized_postgame_sources'][0]['pages'][0]['lines'][0]['text']='FORGED SOURCE CONTEXT'
                wrong['messages'][1]['content']=json.dumps(p)
            else:wrong['messages'].append({'role':'user','content':json.dumps({'training_only_postmortem':'Injected repair hindsight'})})
            rehash(wrong)
            with self.assertRaisesRegex(ValueError,'not authorized'):
                api.validate_training_request(wrong,self.registry)

    def test_independent_request_manifest_hash_and_annotated_source_binding(self):
        request=research.build_postmortem_request(self.game,fixture.postgame())
        entry={k:request[k] for k in ('request_id','game_id','stage','source_hash')} | {'request_sha256':research.value_hash(request)}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'authorizations.json';path.write_text(json.dumps({'schema_version':1,'requests':[entry]}))
            digest=api.file_hash(path)
            known=api.bind_trusted_training_context(registry(self.game),[self.game],api.load_request_authorizations(path,digest))
            api.validate_training_request(request,known)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'authorization manifest changed'):
                api.load_request_authorizations(path,digest)
        annotated=copy.deepcopy(self.game);annotated['sources'][0]['pages'][0]['coverage_audit']={'annotation':'trusted-loader-fixture'}
        with self.assertRaisesRegex(ValueError,'sidecar hash'):
            api.bind_trusted_training_context(registry(self.game),[annotated],{})
        bound=api.bind_trusted_training_context(registry(self.game),[annotated],{},coverage_audit_sha256='c'*64)
        self.assertNotEqual(bound['trusted_source_hashes']['g1'],self.registry['trusted_source_hashes']['g1'])

    def test_four_game_gold_preparation_remains_pending_without_real_qualified_modules(self):
        games=[];known=copy.deepcopy(self.registry)
        for i in range(4):
            g=copy.deepcopy(self.game);g['game_id']=f'g{i}';games.append(g)
            known['games'][g['game_id']]={k:g[k] for k in ('game_id','game_date_et','cutoff_utc','split')}
        with tempfile.TemporaryDirectory() as tmp:
            status=api.prepare_reference_gold_bundle(games,{}, {},tmp,known)
            self.assertEqual(len(status['pending_game_ids']),4)
            self.assertFalse((Path(tmp)/'requests.jsonl').exists())


if __name__=='__main__':unittest.main()
