"""Resumability and scientific input-boundary tests; no Modal jobs are launched."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import bam_research as research
import bam_training as training
import run_bam_pipeline as pipeline


class FakeCall:
    def __init__(self,identity,result,interrupt=False):
        self.object_id=identity;self.result=result;self.interrupt=interrupt;self.gets=0
    def get(self,timeout=None):
        self.gets+=1
        if self.interrupt:
            self.interrupt=False
            raise KeyboardInterrupt('simulated local process interruption')
        return self.result


class FakeFunction:
    def __init__(self,results,interrupt=False):
        self.results=iter(results);self.calls={};self.spawns=0;self.interrupt=interrupt
    def spawn(self,*args):
        self.spawns+=1
        call=FakeCall('fc-'+str(self.spawns),next(self.results),self.interrupt)
        self.calls[call.object_id]=call
        return call


class CloudTests(unittest.TestCase):
    def test_retry_mode_reconnects_after_ambiguous_get_error_and_preserves_history(self):
        from bam_budget import BudgetLedger
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); function=FakeFunction([{'status':'complete','answer':42}])
            budget=BudgetLedger(root/'budget.json',cap_usd=20)
            cloud=pipeline.DurableCloud(root/'calls',functions={'generate':function},
                call_loader=function.calls.__getitem__,retry_failed=True,budget=budget,
                cost_policy={'generate':{'timeout_seconds':3600,'hourly_rate_usd':5}})
            original_get=FakeCall.get
            def retrieval(call,timeout=None):
                if call.gets==0:
                    call.gets+=1
                    raise ConnectionError('Local retrieval failed; remote call still running')
                return original_get(call,timeout)
            with patch.object(FakeCall,'get',retrieval):
                with self.assertRaises(ConnectionError): cloud.invoke('generate',[],{'request':1})
                error_path=next((root/'calls').rglob('error.json')); retained=error_path.read_bytes()
                self.assertEqual(cloud.invoke('generate',[],{'request':1})['answer'],42)
                self.assertEqual(cloud.invoke('generate',[],{'request':1})['answer'],42)
            self.assertEqual(function.spawns,1)
            self.assertEqual(function.calls['fc-1'].gets,2)
            self.assertEqual(error_path.read_bytes(),retained)
            self.assertEqual(len(list((error_path.parent/'errors').glob('*.json'))),1)
            self.assertEqual(budget.snapshot()['totals']['outstanding_reserved_cents'],0)

    def test_inspected_terminal_failure_requires_matching_call_and_reason(self):
        for separate in (True,False):
            with self.subTest(separate=separate),tempfile.TemporaryDirectory() as directory:
                root=Path(directory); function=FakeFunction([{'status':'complete'},{'status':'complete','retry':True}])
                cloud=pipeline.DurableCloud(root,functions={'generate':function},
                    call_loader=function.calls.__getitem__,retry_failed=True)
                with patch.object(FakeCall,'get',side_effect=RuntimeError('remote status initially unknown')):
                    with self.assertRaises(RuntimeError): cloud.invoke('generate',[],{'request':1})
                attempt=next(root.rglob('call.json')).parent
                marker=attempt/('terminal_failure.json' if separate else 'error.json')
                record=pipeline.read_json(marker) if marker.exists() else {}
                record.update(terminal_failure_confirmed=True,call_id='wrong',reason='Inspected terminal worker failure')
                training.write_json(marker,record)
                with self.assertRaises(ValueError): cloud.invoke('generate',[],{'request':1})
                self.assertEqual(function.spawns,1)
                record['call_id']='fc-1';record['reason']=''
                training.write_json(marker,record)
                with self.assertRaises(ValueError): cloud.invoke('generate',[],{'request':1})
                record['reason']='Verified original call terminated; retained logs inspected'
                training.write_json(marker,record)
                self.assertTrue(cloud.invoke('generate',[],{'request':1})['retry'])
                self.assertEqual(function.spawns,2)
                self.assertTrue((attempt/'error.json').exists())

    def test_retry_selection_verifies_failed_result_checksum_before_respawn(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); function=FakeFunction([{'status':'complete'}])
            cloud=pipeline.DurableCloud(root,functions={'generate':function},
                call_loader=function.calls.__getitem__,retry_failed=True)
            cloud.invoke('generate',[],{'request':1})
            next(root.rglob('result.json')).write_text('{"status":"failed"}')
            with self.assertRaisesRegex(ValueError,'cache changed'): cloud.invoke('generate',[],{'request':1})
            self.assertEqual(function.spawns,1)

    def test_budget_admission_precedes_remote_spawn(self):
        from bam_budget import BudgetExceeded, BudgetLedger
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); function=FakeFunction([{}])
            budget=BudgetLedger(root/'budget.json',cap_usd=6)
            cloud=pipeline.DurableCloud(root/'calls',functions={'generate':function},
                call_loader=function.calls.__getitem__,budget=budget,
                cost_policy={'generate':{'timeout_seconds':3600,'hourly_rate_usd':5}})
            with self.assertRaises(BudgetExceeded): cloud.invoke('generate',[],{'request':1})
            self.assertEqual(function.spawns,0)

    def test_budget_reconnect_and_cached_result_do_not_reserve_twice(self):
        from bam_budget import BudgetLedger
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); function=FakeFunction([{'answer':42}],interrupt=True)
            budget=BudgetLedger(root/'budget.json',cap_usd=20)
            cloud=pipeline.DurableCloud(root/'calls',functions={'generate':function},
                call_loader=function.calls.__getitem__,budget=budget,
                cost_policy={'generate':{'timeout_seconds':3600,'hourly_rate_usd':5}})
            with self.assertRaises(KeyboardInterrupt): cloud.invoke('generate',[],{'request':1})
            self.assertEqual(budget.snapshot()['totals']['outstanding_reserved_cents'],575)
            self.assertEqual(cloud.invoke('generate',[],{'request':1}),{'answer':42})
            first=budget.snapshot()
            self.assertEqual(cloud.invoke('generate',[],{'request':1}),{'answer':42})
            self.assertEqual(budget.snapshot(),first)
            self.assertEqual(function.spawns,1)
            self.assertEqual(first['totals']['outstanding_reserved_cents'],0)

    def test_ambiguous_spawn_does_not_automatically_spawn_again(self):
        from bam_budget import BudgetLedger
        class AmbiguousFunction:
            spawns=0
            def spawn(self,*args):
                self.spawns+=1
                raise ConnectionError('Server may have received request')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); function=AmbiguousFunction()
            budget=BudgetLedger(root/'budget.json',cap_usd=20)
            cloud=pipeline.DurableCloud(root/'calls',functions={'generate':function},
                call_loader=lambda _:None,budget=budget,
                cost_policy={'generate':{'timeout_seconds':3600,'hourly_rate_usd':5}})
            with self.assertRaises(ConnectionError): cloud.invoke('generate',[],{'request':1})
            with self.assertRaises(ValueError): cloud.invoke('generate',[],{'request':1})
            self.assertEqual(function.spawns,1)
            self.assertEqual(budget.snapshot()['totals']['outstanding_reserved_cents'],575)

    def test_spawn_cached_and_recovered_without_respawn(self):
        with tempfile.TemporaryDirectory() as directory:
            function=FakeFunction([{'status':'complete','answer':42}],interrupt=True)
            cloud=pipeline.DurableCloud(directory,functions={'generate':function},call_loader=function.calls.__getitem__)
            with self.assertRaises(KeyboardInterrupt): cloud.invoke('generate',['payload'],{'request':'one'})
            self.assertEqual(function.spawns,1)
            result=cloud.invoke('generate',['payload'],{'request':'one'})
            self.assertEqual(result['answer'],42)
            self.assertEqual(cloud.invoke('generate',['payload'],{'request':'one'}),result)
            self.assertEqual(function.spawns,1)
            self.assertEqual(function.calls['fc-1'].gets,2)

    def test_explicit_failed_resume_preserves_prior_result(self):
        with tempfile.TemporaryDirectory() as directory:
            function=FakeFunction([{'status':'failed','error':'interrupted'},{'status':'complete'}])
            cloud=pipeline.DurableCloud(directory,functions={'train':function},call_loader=function.calls.__getitem__)
            self.assertEqual(cloud.invoke('train',[],{'data':'same'})['status'],'failed')
            cloud.retry_failed=True
            self.assertEqual(cloud.invoke('train',[],{'data':'same'})['status'],'complete')
            self.assertEqual(function.spawns,2)
            self.assertEqual(len(list(Path(directory).rglob('result.json'))),2)
            cloud.invoke('train',[],{'data':'same'})
            self.assertEqual(function.spawns,2)

    def test_cached_response_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            function=FakeFunction([{'response':'safe'}])
            cloud=pipeline.DurableCloud(directory,functions={'generate':function},call_loader=function.calls.__getitem__)
            cloud.invoke('generate',[],{'request':1})
            next(Path(directory).rglob('result.json')).write_text('{"response":"changed"}')
            with self.assertRaises(ValueError): cloud.invoke('generate',[],{'request':1})

    def test_binary_result_roundtrip(self):
        value={'payload':b'\x00\xff','nested':[{'data':b'abc'}]}
        self.assertEqual(pipeline.unpacked(json.loads(json.dumps(pipeline.packed(value)))),value)

    def test_upload_is_deterministic_and_explicitly_excludes_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('research_examples.jsonl','research_inputs.jsonl','labels_test.jsonl'):
                (root/name).write_text(name)
            names=['research_examples.jsonl','research_inputs.jsonl']
            one=pipeline.payload_archive(root,names)
            with patch('time.time',return_value=9999999999): two=pipeline.payload_archive(root,names)
            self.assertEqual(one,two)
            with tarfile.open(fileobj=io.BytesIO(one),mode='r:gz') as archive:
                self.assertEqual(set(archive.getnames()),set(names))


def game():
    return {'game_id':'g1','game_date_et':'2025-01-02','cutoff_utc':'2025-01-02T23:00:00Z',
            'home_abbr':'BOS','away_abbr':'MIA','split':'train','p_boost':.6,'forecast_eligible':True,
            'boosting_fold':'older','home_win':1,'recap':'POSTGAME MUST NEVER ENTER',
            'sources':[{'source_id':'s1','publisher':'BOS','url':'https://example.com/notes.pdf',
                        'sha256':'a'*64,'archive_time':'2025-01-02T19:00:00Z',
                        'pages':[{'page':1,'lines':[{'line':1,'text':'A player is questionable tonight.'}]}]}]}


def approved_modules(g):
    selection=research.build_selection_requests(g)[0]
    selected=research.ingest_response(selection,{'reviewed_chunk_id':selection['chunk_id'],'findings':[],
                    'disposition':'All supplied source lines reviewed.'},g,'student')
    requests=research.build_module_requests(g,[selected])
    artifacts=[]
    for request in requests:
        output={'module':request['module'],'findings':[],'unknowns':['No reliable role-specific signal established.'],
                'coverage_note':'One-sided source coverage; all source pages were reviewed.'}
        candidate=research.ingest_response(request,output,g,'student')
        review_request=research.build_review_request(g,candidate,'independent')
        review=research.ingest_response(review_request,{'decisions':[],'report_verdict':'supported',
                    'report_reason':'Coverage and uncertainty are explicit.'},g,'independent')
        artifacts.append(research.apply_semantic_review(candidate,review))
    return artifacts


class InputBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.game=game();self.artifacts=approved_modules(self.game)
        self.provenance=[{'training_game_ids':[],'training_max_date':None,'adapter_sha256':None,'mode':'cold_prompted'} for _ in range(3)]
        self.lineage={'older':{'last_fit_date':'2024-01-01','policy':'earlier_calendar_months_only'}}

    def test_forecast_allowlist_omits_outcomes_and_gold(self):
        provenance=[{'training_game_ids':['private_training_game'],'training_max_date':'2024-01-01',
                     'adapter_sha256':'private_adapter_hash','mode':'expanding_fold'}]*3
        row=pipeline.prediction_row(self.game,self.artifacts,provenance,self.lineage)
        content=json.dumps(row['messages'])
        self.assertNotIn('POSTGAME MUST NEVER ENTER',content)
        self.assertNotIn('home_win',content)
        self.assertNotIn('gold',content)
        for forbidden in ('private_training_game','private_adapter_hash','parent_hashes','fold_id','training_game_ids','quotations'):
            self.assertNotIn(forbidden,content)
        self.assertIn('private_training_game',json.dumps(row['research_provenance']))
        self.assertEqual(row['p_boost'],.6)

    def test_teacher_output_cannot_be_substituted_for_student(self):
        artifacts=copy.deepcopy(self.artifacts)
        artifacts[0]['model_role']='gold_teacher'
        with self.assertRaises(ValueError): pipeline.prediction_row(self.game,artifacts,self.provenance,self.lineage)
        artifacts=copy.deepcopy(self.artifacts);artifacts[0]['status']='mechanically_valid'
        with self.assertRaises(ValueError): pipeline.prediction_row(self.game,artifacts,self.provenance,self.lineage)

    def test_own_gold_and_future_student_lineage_fail(self):
        for ids,day in [(['g1'],'2024-01-01'),(['another'],'2025-01-03')]:
            provenance=[{'training_game_ids':ids,'training_max_date':day,'adapter_sha256':'hash','mode':'expanding_fold'}]*3
            with self.assertRaises(ValueError): pipeline.prediction_row(self.game,self.artifacts,provenance,self.lineage)

    def test_research_only_game_cannot_enter_forecast(self):
        with self.assertRaises(ValueError):
            pipeline.prediction_row(self.game | {'forecast_eligible':False},self.artifacts,self.provenance,self.lineage)

    def test_coverage_annotation_preserves_source_and_rejects_changed_text(self):
        candidate=copy.deepcopy(self.game)
        page=candidate['sources'][0]['pages'][0];page['text']='Canonical source text'
        before=copy.deepcopy(page)
        audit={'game_id':'g1','source_id':'s1','source_sha256':'a'*64,'page':1,
            'text_sha256':hashlib.sha256(page['text'].encode()).hexdigest(),
            'classification':'numeric_image_untranscribed','mandatory_unknown':True,
            'coverage_note':'Numeric image not transcribed.'}
        pipeline.attach_coverage_audits(candidate,[audit])
        self.assertEqual({k:v for k,v in page.items() if k!='coverage_audit'},before)
        self.assertEqual(page['coverage_audit'],audit)
        with self.assertRaises(ValueError): pipeline.attach_coverage_audits(candidate,[audit | {'text_sha256':'0'*64}])

    def test_coverage_sidecar_requires_bound_snapshot_and_visual_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);snapshot=root/'snapshot';snapshot.mkdir()
            (snapshot/'manifest.json').write_text('{}');(root/'page.png').write_bytes(b'visual')
            candidate=copy.deepcopy(self.game);candidate['sources'][0]['pages'][0]['text']='text'
            entry={'game_id':'g1','source_id':'s1','source_sha256':'a'*64,'page':1,
                'text_sha256':hashlib.sha256(b'text').hexdigest(),
                'classification':'numeric_image_untranscribed','mandatory_unknown':True,
                'coverage_note':'Numeric image not transcribed.','review_image_path':'page.png',
                'review_image_sha256':training.file_hash(root/'page.png')}
            training.write_json(root/'sidecar.json',{'schema_version':1,
                'snapshot_manifest_sha256':training.file_hash(snapshot/'manifest.json'),'page_audits':[entry]})
            config={'coverage_audit_path':'sidecar.json','coverage_audit_sha256':training.file_hash(root/'sidecar.json')}
            with patch.object(pipeline.bam_data,'load_snapshot_game',return_value=candidate):
                audits,identity=pipeline.load_coverage_audits(config,snapshot,root)
                self.assertEqual(identity['pages'],1);self.assertEqual(audits['g1'],[entry])
                (root/'page.png').write_bytes(b'changed')
                with self.assertRaises(ValueError): pipeline.load_coverage_audits(config,snapshot,root)

    def test_reference_subsets_must_be_training_only(self):
        rows=[{'game_id':'train','split':'train'},{'game_id':'test','split':'test'}]
        self.assertEqual(len(pipeline.selected_games(rows,['train'])),1)
        with self.assertRaises(ValueError): pipeline.selected_games(rows,['test'])
        with self.assertRaises(ValueError): pipeline.selected_games(rows,['missing'])

    def test_model_specific_tokenizer_and_reasoning_budget(self):
        worker=pipeline.Pipeline.__new__(pipeline.Pipeline)
        worker.student={'model_id':'student/model','model_revision':'a'*40}
        worker.teacher={'model_id':'teacher/model','model_revision':'b'*40}
        worker.gold_reviewer={'model_id':'reviewer/model','model_revision':'c'*40}
        worker.config={'generation':{'max_model_len':32768},'generation_by_model':{
            'teacher/model':{'expects_reasoning':True,'reasoning_token_reserve':4096}}}
        student_tokenizer=object();teacher_tokenizer=object()
        worker.tokenizers={pipeline.model_identity(worker.student):student_tokenizer,
                           pipeline.model_identity(worker.teacher):teacher_tokenizer}
        with patch.object(research,'count_request_tokens',side_effect=lambda request,tokenizer:1 if tokenizer is student_tokenizer else 2):
            self.assertEqual(worker.tokens({'model_role':'research_student'}),1)
            self.assertEqual(worker.tokens({'model_role':'gold_teacher'}),2)
            self.assertEqual(worker.tokens({'model_role':'semantic_reviewer','reviewer_model':pipeline.model_identity(worker.teacher)}),2)
            self.assertEqual(worker.request_model({'model_role':'semantic_reviewer','reviewer_model':pipeline.model_identity(worker.gold_reviewer)}),worker.gold_reviewer)
        artifact={'output':{'findings':[]}}
        expected=32768-4096-research.semantic_review_max_tokens(artifact)
        self.assertEqual(worker.review_budget(worker.teacher,artifact),expected)

    def test_long_context_review_preserves_source_and_reserves_actual_completion(self):
        worker=pipeline.Pipeline.__new__(pipeline.Pipeline)
        model={'model_id':'long-context/reviewer','model_revision':'d'*40}
        worker.config={'generation':{'max_model_len':65536,'expects_reasoning':True,
                                     'reasoning_token_reserve':4096}}
        selection=research.build_selection_requests(self.game)[0]
        finding={'id':'f1','module':'availability_rotation','fact':'A player is questionable tonight.',
            'fact_date':None,'time_scope':'current','interpretation':'Final status remains uncertain.',
            'counterevidence':'No absence is confirmed.','uncertainty':'Final status is unknown.',
            'booster_overlap':'unknown','citations':[{'source_id':'s1','page':1,'lines':[1]}]}
        selected=research.ingest_response(selection,{'reviewed_chunk_id':selection['chunk_id'],
            'findings':[finding],'disposition':'All supplied source lines reviewed.'},self.game,'student')
        original=research.build_module_requests(self.game,[selected])[0]
        candidate=research.ingest_response(original,self.artifacts[0]['output'],self.game,'student')
        worker.tokens=lambda request,selected_model:40000
        review=worker.review_request(self.game,candidate,model,original)
        self.assertIn('A player is questionable tonight.',json.dumps(review))
        self.assertEqual(review['reviewer_model'],pipeline.model_identity(model))
        limit=65536-4096-review['max_tokens']
        worker.tokens=lambda request,selected_model:limit
        worker.review_request(self.game,candidate,model,original)
        worker.tokens=lambda request,selected_model:limit+1
        with self.assertRaises(ValueError): worker.review_request(self.game,candidate,model,original)
        worker.config['generation']['max_model_len']=32768
        worker.tokens=lambda request,selected_model:40000
        with self.assertRaises(ValueError): worker.review_request(self.game,candidate,model,original)

    def test_incomplete_raw_output_is_retained_but_transport_errors_stay_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            worker=pipeline.Pipeline.__new__(pipeline.Pipeline)
            worker.root=Path(directory);worker.game=lambda gid:self.game
            request=research.build_selection_requests(self.game)[0]
            answer=json.dumps({'reviewed_chunk_id':request['chunk_id'],
                               'findings':[], 'disposition':'Source page reviewed.'})
            for finish,raw in [('length',answer[:-4]),('stop',answer)]:
                artifact=worker.ingest(request,{'status':'error','text':'','raw_text':raw,
                    'finish_reason':finish},'student/model')
                self.assertEqual(artifact['status'],'rejected')
                self.assertEqual(artifact['raw_response'],raw)
                self.assertEqual(artifact['finish_reason'],finish if finish=='length' else 'transport_error')

    def test_heldout_preparation_skips_unused_finals_but_keeps_partial_research(self):
        for split in ('dev','test'):
            for hierarchy in (False,True):
                with self.subTest(split=split,hierarchy=hierarchy),tempfile.TemporaryDirectory() as directory:
                    worker=pipeline.Pipeline.__new__(pipeline.Pipeline)
                    g=copy.deepcopy(self.game);g['split']=split
                    worker.root=Path(directory);worker.rows=[g];worker.by_id={g['game_id']:g}
                    worker.batch_games=64;worker.game=lambda gid:g;worker.tokens=lambda request:100
                    worker.student={'model_id':'student'};worker.teacher={'model_id':'critic'}
                    selection_request=research.build_selection_requests(g)[0]
                    selection=research.ingest_response(selection_request,{'reviewed_chunk_id':selection_request['chunk_id'],
                        'findings':[],'disposition':'All supplied source lines reviewed.'},g,'student')
                    pipeline.training.write_json(worker.path(g['game_id'],'selection.json'),[selection])
                    finals=research.build_module_requests(g,[selection])
                    partial={**finals[0],'stage':'module_partial','request_id':'partial-request'}
                    calls=[]
                    def resolve(requests,model,*,reviewer):
                        calls.append((requests,model,reviewer))
                        self.assertTrue(hierarchy)
                        self.assertEqual(requests,[partial])
                        return {'partial-request':{'stage':'module_partial','status':'approved'}}
                    worker.resolve=resolve
                    with patch.object(research,'build_module_requests',return_value=[partial] if hierarchy else finals), \
                         patch.object(research,'build_module_synthesis_requests',return_value=finals) as synthesis:
                        worker.modules()
                    self.assertEqual(len(calls),int(hierarchy))
                    self.assertEqual(synthesis.call_count,int(hierarchy))
                    saved=pipeline.read_json(worker.path(g['game_id'],'module_requests.json'))
                    self.assertEqual(saved,finals)
                    self.assertEqual(len(research.export_student_inputs(g,saved)),3)
                    self.assertEqual(pipeline.read_json(worker.path(g['game_id'],'prompted_modules.json')),[])

    def test_identical_partial_retry_stops_without_promoting_rejected_facts(self):
        with tempfile.TemporaryDirectory() as directory:
            worker=pipeline.Pipeline.__new__(pipeline.Pipeline)
            worker.root=Path(directory);worker.game=lambda gid:self.game
            requests=research.build_selection_requests(self.game)
            def respond(batch,model):
                responses={}
                for request in batch:
                    bad={'id':'bad','module':research.MODULES[0],'fact':'Unsupported','fact_date':'2025-01-02',
                         'time_scope':'current','interpretation':'Unknown','counterevidence':'Unknown',
                         'uncertainty':'Unknown','booster_overlap':'unknown',
                         'citations':[{'source_id':'s1','page':1,'lines':[999]}]}
                    responses[request['request_id']]={'text':json.dumps({'reviewed_chunk_id':request['chunk_id'],
                        'findings':[bad],'disposition':'All page lines reviewed.'}),'finish_reason':'stop'}
                return responses
            worker.raw=respond
            result=worker.resolve(requests,{'model_id':'student','model_revision':'a'*40},selection=True)
            artifact=result[requests[0]['request_id']]
            self.assertEqual(artifact['status'],'mechanically_valid')
            self.assertEqual(artifact['output']['findings'],[])
            self.assertEqual(len(artifact['output']['rejected_findings']),1)
            self.assertTrue(list((Path(directory)/'selection_repair_exhausted').glob('*.json')))
            self.assertEqual(len(list((Path(directory)/'artifacts').glob('*.json'))),2)
            record=json.loads(next((Path(directory)/'selection_repair_exhausted').glob('*.json')).read_text())
            self.assertEqual(record['reason'],'identical_partial_retry')
            self.assertEqual(record['repairs_attempted'],1)

    def test_snapshot_prepare_never_opens_absent_test_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);snapshot=root/'snapshot';snapshot.mkdir();post=root/'post';post.mkdir()
            (snapshot/'inputs.jsonl').write_text('{}\n')
            manifest={'artifact_sha256':{'inputs.jsonl':training.file_hash(snapshot/'inputs.jsonl'),'labels_test.jsonl':'not-present'}}
            training.write_json(snapshot/'manifest.json',manifest)
            training.write_json(post/'manifest.json',{'snapshot_manifest_sha256':training.file_hash(snapshot/'manifest.json')})
            worker=pipeline.Pipeline.__new__(pipeline.Pipeline)
            worker.snapshot=snapshot;worker.postgame=post;worker.snapshot_manifest=manifest
            worker.protocol={'snapshot_manifest_sha256':training.file_hash(snapshot/'manifest.json')}
            worker.root=root/'run';worker.rows=[self.game];worker.game=lambda gid:self.game
            worker.save_requests=lambda requests:None
            outputs=worker.prepare_research()
            self.assertTrue(outputs[0].exists())
            self.assertFalse((snapshot/'labels_test.jsonl').exists())

    def test_complete_research_chain_with_fake_cloud_keeps_gold_only_in_targets(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**options):
                ids=[]
                for message in messages:
                    ids+=[{'system':1,'user':2,'assistant':3}[message['role']]]+[ord(c)+10 for c in message['content']]+[4]
                if options.get('add_generation_prompt'): ids+=[3]
                return ids
        class FakeCloud:
            def invoke(self,name,args,identity):
                if name!='generate': raise AssertionError('Research preparation must only call generation')
                responses=[]
                runtime={'model':{'model_spec':args[1]},
                    'worker_sha256':training.file_hash(pipeline.ROOT/'scripts/bam_inference.py'),
                    'tokenizer_helper_sha256':training.file_hash(pipeline.ROOT/'scripts/bam_training.py')}
                def module(name):
                    return {'module':name,'findings':[],'unknowns':['No supported specific signal established.'],
                            'coverage_note':'All source pages reviewed; one-sided evidence.'}
                for request in args[0]:
                    stage=request['stage']
                    if stage=='selection':
                        value={'reviewed_chunk_id':request['chunk_id'],'findings':[],'disposition':'All page lines read.'}
                    elif stage in ('module','module_partial'): value=module(request['module'])
                    elif stage=='semantic_review':
                        value={'decisions':[{'id':identity,'verdict':'supported','reason':'Matches source.'} for identity in request['expected_ids']],
                               'report_verdict':'supported','report_reason':'Coverage and uncertainty are explicit.'}
                    elif stage=='postmortem':
                        value={'general':[{'observation':'The game ended.','foreseeable_signal':'No specific supported signal.',
                                           'unforeseeable':'Outcome variation.','citations':[{'source_id':'post','page':1,'lines':[1]}]}],
                               'timeline':['Completed game.'],'market_view':'Unavailable.'}
                    elif stage=='gold': value={'modules':[module(name) for name in research.MODULES]}
                    else: raise AssertionError(stage)
                    response={'request_id':request['request_id'],'request_sha256':training.value_hash(request),
                                      'prompt_token_ids_sha256':training.value_hash(training.chat_token_ids(worker.tokenizer,request['messages'],generation_prompt=True)),
                                      'execution_sha256':training.value_hash(runtime),
                                      'text':json.dumps(value),'finish_reason':'stop'}
                    response['response_sha256']=training.value_hash(response)
                    responses.append(response)
                return {'responses':responses,'identity':runtime}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);snapshot=root/'snapshot';snapshot.mkdir();post=root/'post';post.mkdir()
            (snapshot/'inputs.jsonl').write_text('{}\n')
            manifest={'artifact_sha256':{'inputs.jsonl':training.file_hash(snapshot/'inputs.jsonl')}}
            training.write_json(snapshot/'manifest.json',manifest)
            record={'game_id':'g1','split':'train','normalized':True,'sources':[{'source_id':'post','publisher':'Normalized results',
                    'url':'https://example.com/post','sha256':'b'*64,'archive_time':'2025-01-03T04:00:00Z',
                    'pages':[{'page':1,'lines':[{'line':1,'text':'The game ended.'}]}]}]}
            training.write_rows(post/'records.jsonl',[record])
            training.write_json(post/'manifest.json',{'snapshot_manifest_sha256':training.file_hash(snapshot/'manifest.json'),
                    'artifact_sha256':{'records.jsonl':training.file_hash(post/'records.jsonl')}})
            worker=pipeline.Pipeline.__new__(pipeline.Pipeline)
            worker.snapshot=snapshot;worker.postgame=post;worker.snapshot_manifest=manifest
            worker.protocol={'snapshot_manifest_sha256':training.file_hash(snapshot/'manifest.json')}
            worker.root=root/'run';worker.training=worker.root/'training';worker.report=root/'report'
            worker.run_id='synthetic';worker.rows=[self.game];worker.by_id={'g1':self.game};worker.game=lambda gid:self.game
            worker.batch_games=8;worker.request_batch_size=64;worker.game_ids=['g1']
            worker.config={'generation':{'max_model_len':32768},'research':{'max_seq_length':16384}}
            worker.student={'model_id':'student/model','model_revision':'a'*40}
            worker.teacher={'model_id':'teacher/model','model_revision':'b'*40}
            worker.gold_reviewer=worker.student
            worker.tokenizer=Tokenizer();worker.tokens=lambda request,model=None:128;worker.cloud=FakeCloud()
            worker.tokenizers={pipeline.model_identity(worker.student):worker.tokenizer,pipeline.model_identity(worker.teacher):worker.tokenizer}
            training.write_json(worker.root/'protocol.json',worker.protocol)
            for stage in pipeline.STAGES[:5]:
                result=worker.run(stage)
                self.assertFalse(result['pipeline_complete'])
            inputs=training.read_rows(worker.training/'research_inputs.jsonl')
            targets=training.read_rows(worker.training/'research_examples.jsonl')
            self.assertEqual(len(inputs),3);self.assertEqual(len(targets),3)
            self.assertTrue(all(len(row['messages'])==2 for row in inputs))
            self.assertTrue(all(len(row['messages'])==3 and row['review_status']=='accepted' for row in targets))
            self.assertTrue(all(row['messages']==target['messages'][:-1] for row,target in zip(inputs,targets)))
            self.assertFalse((worker.training/'labels_test.jsonl').exists())
            # Actual student validation failure is repaired with its own adapter,
            # retaining the initial report rather than substituting teacher gold.
            worker.config.update(minimum_eval_date='2024-09-25',forecast={'max_seq_length':16384})
            training.write_json(snapshot/'boosting_lineage.json',self.lineage)
            prompted={a['module']:a['output'] for a in pipeline.read_json(worker.path('g1','prompted_modules.json'))}
            generated=[]
            for index,source in enumerate(inputs):
                generated.append({k:v for k,v in source.items() if k!='messages'} | {
                    'input_sha256':training.value_hash(source['messages']),
                    'report':'invalid JSON' if index==0 else json.dumps(prompted[source['module']]),
                    'research_provenance':self.provenance[index]})
            training.write_rows(worker.training/'generated_modules.jsonl',generated)
            initial_hash=training.file_hash(worker.training/'generated_modules.jsonl')
            repairs=[]
            def same_adapter_repair(stage,names,fold=None,repair_batch=None):
                self.assertEqual(stage,'research-repair');repairs.append(repair_batch)
                requests=training.read_rows(worker.training/names[0]);outputs=[]
                for request in requests:
                    original=next(row for row in generated if row['module']==request['module'])
                    source=next(row for row in inputs if row['module']==request['module'])
                    training.validate_student_repair(request,source,original,original)
                    outputs.append(original | {'report':json.dumps(prompted[request['module']]),
                        'repair_history':[{'attempt':1,'parent_output_sha256':training.value_hash(original)}]})
                path=worker.training/'research/repairs'/repair_batch/'outputs.jsonl'
                training.write_rows(path,outputs)
                marker=path.parent/'export.json';training.write_json(marker,{'synthetic':True})
                return marker
            worker.remote_stage=same_adapter_repair
            worker.review_student()
            self.assertEqual(len(repairs),1)
            self.assertEqual(training.file_hash(worker.training/'generated_modules.jsonl'),initial_hash)
            selected=pipeline.read_json(worker.path('g1','selected_student_generations.json'))
            self.assertEqual(sum(bool(row.get('repair_history')) for row in selected),1)
            predictions=training.read_rows(worker.training/'prediction_inputs.jsonl')
            self.assertEqual(len(predictions),1)
            self.assertNotIn('invalid JSON',json.dumps(predictions[0]['messages']))


if __name__=='__main__': unittest.main()
