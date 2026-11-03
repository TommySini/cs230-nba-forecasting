"""CPU tests for independent-game, chronology, completion and heldout boundaries."""
import copy
import importlib.util
import json
import sys
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

SPEC = importlib.util.spec_from_file_location('bam_training', Path(__file__).resolve().parents[1]/'scripts/bam_training.py')
bam = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bam)


def research(gid, day, split='train', target=False, module='availability', **extra):
    messages = [{'role':'system','content':'Use pregame sources.'},{'role':'user','content':'Source quotation.'}]
    if target:
        messages.append({'role':'assistant','content':'Reviewed report.'})
    return {'game_id':gid,'game_date_et':day,'split':split,'module':module,
            'messages':messages,'source_hashes':['source'],
            **({'teacher_hash':'teacher','review_status':'accepted'} if target else {}),**extra}


def forecast(gid, day, split, p=.6):
    return {'game_id':gid,'game_date_et':day,'split':split,'p_boost':p,
            'messages':[{'role':'system','content':'Forecast.'},{'role':'user','content':'Pregame report.'}],
            'booster_provenance':{'training_max_date':'2023-01-01'},
            'research_provenance':{'mode':'cold_prompted','training_game_ids':[],
                                   'training_max_date':None,'adapter_sha256':None}}


class ToyTokenizer:
    chat_template = 'fixture-chat'
    special_tokens_map = {'eos_token': 'fixture-eos'}
    def get_vocab(self): return {'fixture': 1}
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_dict=False):
        result = [1]
        for m in messages:
            result += [{'system':2,'user':3,'assistant':4}[m['role']]]
            result += [ord(c)+10 for c in m['content']]
            result += [5]
        if add_generation_prompt:
            result += [4]
        return result


class ResearchBoundaries(unittest.TestCase):
    def test_bounded_output_retry_keeps_same_input_adapter_and_raw_attempts(self):
        config={'research':{'max_new_tokens':4,'output_token_budgets':[4,8]}}
        items=[{'request_sha256':'one','input_ids':[1,2],'input_sha256':'source'},
               {'request_sha256':'two','input_ids':[3,4],'input_sha256':'source2'}]
        provenance={'adapter_sha256':'same-fold-adapter'}; calls=[]
        def generate(rows,budget):
            calls.append(([r['request_sha256'] for r in rows],budget))
            return [[7,8,8,8] if r['request_sha256']=='one' and budget==4 else [7,9] for r in rows]
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            result=bam.bounded_research_completions(items,provenance,config,directory,max_context=20,
                eos_ids=[9],generate_batch=generate,decode=lambda ids:str(ids))
            self.assertFalse(result['failures']);self.assertEqual(calls,[(['one','two'],4),(['one'],8)])
            failed=json.loads((Path(directory)/'one/attempt_1_4.json').read_text())['record']
            self.assertEqual(failed['status'],'token_limit');self.assertEqual(failed['raw_token_ids'],[7,8,8,8])
            self.assertEqual(failed['research_provenance'],provenance)
            again=bam.bounded_research_completions(items,provenance,config,directory,max_context=20,
                eos_ids=[9],generate_batch=lambda *_:self.fail('Cache should prevent any generation'),decode=str)
            self.assertEqual(again,result)

    def test_larger_output_context_failure_retains_successful_sibling(self):
        config={'research':{'max_new_tokens':4,'output_token_budgets':[4,8]}}
        items=[{'request_sha256':'one','input_ids':[1]*6,'input_sha256':'source'},
               {'request_sha256':'two','input_ids':[3],'input_sha256':'source2'}]
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            result=bam.bounded_research_completions(items,{},config,directory,max_context=10,eos_ids=[9],
                generate_batch=lambda rows,budget:[[1]*4 if r['request_sha256']=='one' else [9] for r in rows],decode=str)
            self.assertIn('two',result['completed']);self.assertEqual(len(result['failures']),1)
            record=json.loads((Path(directory)/'one/attempt_2_8.json').read_text())['record']
            self.assertEqual(record['status'],'context_limit');self.assertEqual(record['prompt_tokens'],6)
            self.assertIn('input unchanged',record['error'])

    def test_no_implicit_output_expansion_or_changed_source_cache(self):
        self.assertEqual(bam.research_output_budgets({'research':{'max_new_tokens':4096}}),[4096])
        for value in ([4096,2048],[4096,4096],[4096,9000],[2048,4096,8192]):
            with self.assertRaises(ValueError):bam.research_output_budgets({'research':{'max_new_tokens':4096,'output_token_budgets':value}})

    def repair_fixture(self):
        source=research('a','2024-02-01')
        provenance={'mode':'expanding_fold','training_game_ids':['earlier'],
                    'training_max_date':'2024-01-01','adapter_sha256':'same_adapter'}
        generated={k:v for k,v in source.items() if k!='messages'} | {'report':'Initial rejected report',
                    'input_sha256':bam.value_hash(source['messages']),'research_provenance':provenance}
        request={k:generated[k] for k in ('game_id','game_date_et','split','module','source_hashes','research_provenance')}
        request.update(messages=source['messages'],original_input_sha256=generated['input_sha256'],attempt=1,
                       previous_response=generated['report'],parent_output_sha256=bam.value_hash(generated),
                       repair_history=[],feedback={'errors':['Citation does not support the scope.']})
        return source,generated,request

    def test_student_repair_preserves_source_adapter_and_parent(self):
        source,generated,request=self.repair_fixture()
        bam.validate_student_repair(request,source,generated,generated)
        for changes in ({'attempt':3},{'home_win':1},{'game_date_et':'2024-03-01'},{'split':'test'},{'original_input_sha256':'wrong'},
                        {'parent_output_sha256':'wrong'},{'previous_response':'Teacher gold'},
                        {'research_provenance':generated['research_provenance'] | {'adapter_sha256':'different'}}):
            with self.assertRaises(ValueError):
                bam.validate_student_repair(request | changes,source,generated,generated)

    def test_student_second_repair_requires_preserved_first_repair(self):
        source,initial,request=self.repair_fixture()
        previous=initial | {'report':'First repair','repair_history':[{'attempt':1}]}
        second=request | {'attempt':2,'previous_response':previous['report'],
                           'parent_output_sha256':bam.value_hash(previous),'repair_history':previous['repair_history']}
        bam.validate_student_repair(second,source,initial,previous)
        with self.assertRaises(ValueError): bam.validate_student_repair(second,source,initial,initial)

    def test_chat_tokenizer_rejects_mapping_or_nested_output(self):
        class BadTokenizer:
            def __init__(self, output): self.output=output
            def apply_chat_template(self, *args, **kwargs):
                self.return_dict=kwargs['return_dict']
                return self.output
        for value in ({'input_ids':[1,2]}, [[1,2]], [], [True,2], ['1',2]):
            tokenizer=BadTokenizer(value)
            with self.assertRaises(ValueError): bam.chat_token_ids(tokenizer,[],generation_prompt=True)
            self.assertFalse(tokenizer.return_dict)

    def test_real_pinned_local_qwen_tokenizer(self):
        path=Path(__file__).resolve().parents[1]/'data/models/qwen2.5-14b-instruct-4bit-20240918'
        if not (path/'tokenizer.json').exists():
            self.skipTest('Pinned local Qwen tokenizer is not present in this checkout')
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        messages=research('synthetic','2024-01-01',target=True)['messages']
        ids=bam.prompt_tokens(tokenizer,messages[:-1],1024)
        completion=bam.completion_tokens(tokenizer,messages,1024)
        self.assertTrue(ids and all(type(token) is int for token in ids))
        self.assertEqual(completion['input_ids'][:completion['prefix_index']+1],ids)

    def test_separate_fold_invocations_share_scientific_lock_and_detect_tampering(self):
        inputs=[research('a','2024-01-01'),research('b','2024-02-01')]
        config={'model_id':'test/model','model_revision':'a'*40,'minimum_eval_date':'2024-09-25',
                'research':{'folds':2,'mode':'prompted'}}
        provenance={'mode':'cold_prompted','training_game_ids':[],'training_max_date':None,
                    'adapter_sha256':None,'base_model':bam._model_config(config)}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            bam.write_rows(root/'research_inputs.jsonl',inputs)
            bam.write_rows(root/'research_examples.jsonl',[])
            for index,row in enumerate(inputs):
                output={k:v for k,v in row.items() if k!='messages'} | {
                    'report':'Synthetic cached report','input_sha256':bam.value_hash(row['messages']),
                    'research_provenance':provenance}
                path=root/f'research/fold_{index}/generated.jsonl'
                bam.write_rows(path,[output]);bam.write_json(path.with_suffix('.manifest.json'),{'sha256':bam.file_hash(path)})
            with patch.object(bam,'_load_model',side_effect=AssertionError('Cached folds should never load a GPU model')):
                first=bam.generate_research(root,config | {'research':config['research'] | {'fold_ids':[0]}})
                second=bam.generate_research(root,config | {'research':config['research'] | {'fold_ids':[1]}})
            self.assertFalse(first['complete']);self.assertTrue(second['complete'])
            self.assertEqual(second['modules'],2)
            lock=json.loads((root/'research/generation/request.json').read_text())
            self.assertNotIn('fold_ids',lock['config']['research'])
            (root/'generated_modules.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError): bam.generate_research(root,config)

    def test_game_group_and_date_split(self):
        rows = [research('a','2024-01-01'),research('a','2024-01-01',module='matchup')]
        bam.validate_groups(rows)
        rows[1]['split']='dev'
        with self.assertRaises(ValueError): bam.validate_groups(rows)
        with self.assertRaises(ValueError):
            bam.validate_groups([research('a','2024-01-01'),research('b','2024-01-01','dev')])

    def test_expanding_folds_use_only_earlier_dates_and_research_only_games(self):
        inputs = [research(str(i),f'2024-01-0{i}') for i in range(2,7)]
        inputs += [research('dev','2024-02-01','dev'),research('test','2024-03-01','test')]
        gold = [research(str(i),f'2024-01-0{i}',target=True) for i in range(1,7)]
        plans = bam.expanding_folds(inputs,gold,3)
        self.assertIn('1',{r['game_id'] for r in plans[0]['fit']})
        predicted = []
        for plan in plans:
            predicted += plan['predict']
            self.assertLess(max(r['game_date_et'] for r in plan['fit']),min(r['game_date_et'] for r in plan['predict']))
            self.assertFalse({r['game_id'] for r in plan['fit']} & {r['game_id'] for r in plan['predict']})
        self.assertCountEqual([r['game_id'] for r in predicted],[r['game_id'] for r in inputs])

    def test_cold_initial_fold(self):
        inputs = [research(str(i),f'2024-01-0{i}') for i in range(1,4)]
        gold = [research(str(i),f'2024-01-0{i}',target=True) for i in range(1,4)]
        self.assertEqual(bam.expanding_folds(inputs,gold,2)[0]['mode'],'cold_prompted')

    def test_unreviewed_and_heldout_gold_rejected(self):
        row = research('a','2024-01-01',target=True)
        for changes in ({'review_status':'pending'},{'split':'dev'},{'teacher_hash':''}):
            with self.assertRaises(ValueError):
                bam.validate_research_rows([row | changes],targets=True)

    def test_duplicate_module_and_raw_outcome_rejected(self):
        row = research('a','2024-01-01')
        with self.assertRaises(ValueError): bam.validate_research_rows([row,row],targets=False)
        with self.assertRaises(ValueError): bam.validate_research_rows([row | {'home_win':1}],targets=False)

    def test_completion_mask_and_prefix_position(self):
        tokenizer = ToyTokenizer()
        messages = research('a','2024-01-01',target=True)['messages']
        encoded = bam.completion_tokens(tokenizer,messages,1000)
        n = encoded['prefix_index']+1
        self.assertTrue(all(v==-100 for v in encoded['labels'][:n]))
        self.assertEqual(encoded['labels'][n:],encoded['input_ids'][n:])
        self.assertEqual(encoded['input_ids'][:n],bam.prompt_tokens(tokenizer,messages[:-1],1000))
        modified = copy.deepcopy(messages)
        modified[-1]['content']='Completely different outcome-revealing teacher explanation.'
        changed = bam.completion_tokens(tokenizer,modified,1000)
        self.assertEqual(changed['input_ids'][:n],encoded['input_ids'][:n])

    def test_no_silent_truncation(self):
        with self.assertRaises(ValueError):
            bam.completion_tokens(ToyTokenizer(),research('a','2024-01-01',target=True)['messages'],10)


class ForecastBoundaries(unittest.TestCase):
    def test_forecast_own_gold_and_future_gold_rejected(self):
        row = forecast('a','2024-10-01','train')
        for provenance in (
            {'mode':'expanding_fold','training_game_ids':['a'],'training_max_date':'2024-01-01'},
            {'mode':'expanding_fold','training_game_ids':['b'],'training_max_date':'2024-10-02'},
            {'mode':'final_train','training_game_ids':['b'],'training_max_date':'2024-01-01'},
        ):
            with self.assertRaises(ValueError):
                bam.validate_prediction_rows([row | {'research_provenance':provenance}])

    def test_in_sample_booster_research_only_and_old_eval_rejected(self):
        row = forecast('a','2024-10-01','dev')
        for changes in ({'booster_provenance':{'training_max_date':'2024-10-01'}},
                        {'forecast_eligible':False},{'home_win':1}):
            with self.assertRaises(ValueError): bam.validate_prediction_rows([row | changes])
        with self.assertRaises(ValueError): bam.validate_prediction_rows([row],'2024-11-01')

    def test_frozen_zero_control_can_win_selection(self):
        rows = [forecast('a','2024-01-01','train',.5),forecast('b','2024-02-01','dev',.5)]
        selected,candidates = bam.fit_frozen_control(np.ones((2,2)),rows,{'a':1},{'b':0},[.01])
        self.assertEqual(selected['name'],'zero_correction')
        self.assertTrue(np.array_equal(selected['theta'],np.zeros(3)))
        self.assertEqual(candidates[0]['dev']['brier'],.25)

    def test_date_bootstrap_exact_zero_and_joint_units(self):
        rows = [forecast('a','2024-01-01','test'),forecast('b','2024-01-01','test'),forecast('c','2024-01-02','test')]
        labels = {'a':1,'b':0,'c':1}
        result = bam.paired_date_bootstrap(rows,[r['p_boost'] for r in rows],labels,samples=100)
        self.assertEqual(result['ci95'],[0.,0.])
        self.assertEqual(result['date_clusters'],2)
        self.assertEqual(result['hybrid_minus_boost_brier'],0.)

    def test_labels_are_exact_game_join(self):
        with tempfile.TemporaryDirectory() as directory:
            bam.write_rows(Path(directory)/'labels_train.jsonl',[{'game_id':'a','home_win':1}])
            self.assertEqual(bam._labels(directory,'train',['a']),{'a':1})
            with self.assertRaises(ValueError): bam._labels(directory,'train',['b'])

    def test_metrics_match_manual_brier(self):
        result = bam.metrics([.8,.3],[1,0])
        self.assertAlmostEqual(result['brier'],.065)
        self.assertEqual(result['accuracy'],1.)
        with self.assertRaises(ValueError): bam.metrics([float('nan')],[1])

    def test_immutable_revision_required(self):
        with self.assertRaises(ValueError):
            bam._model_config({'model_id':'model','model_revision':'main','minimum_eval_date':'2024-10-01'})

    def test_prefix_hidden_has_no_completion_dependency(self):
        # The helper interface accepts only pregame IDs; simulate causal reader on CPU.
        import torch
        class Output: pass
        class Backbone:
            def __call__(self,input_ids,**kwargs):
                output = Output()
                output.last_hidden_state=torch.stack((input_ids.float(),input_ids.float()+1),dim=-1)
                return output
        class Model:
            base_model=Backbone()
        original_tensor = torch.tensor
        with patch.object(torch,'tensor',side_effect=lambda data,**kw:original_tensor(data)):
            hidden=bam._prefix_hidden(Model(),[1,2,3])
        self.assertTrue(torch.allclose(hidden,torch.tensor([.6,.8])))

    def test_frozen_pipeline_does_not_open_test_labels_until_selection(self):
        rows = [forecast('a','2024-01-01','train',.5),
                forecast('b','2024-10-01','dev',.5),forecast('c','2024-11-01','test',.5)]
        config = {'model_id':'test/model','model_revision':'a'*40,'minimum_eval_date':'2024-09-25',
                  'forecast':{'train_lora':False,'l2_candidates':[.01]}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            bam.write_rows(root/'prediction_inputs.jsonl',rows)
            bam.write_rows(root/'labels_train.jsonl',[{'game_id':'a','home_win':1}])
            bam.write_rows(root/'labels_dev.jsonl',[{'game_id':'b','home_win':0}])
            (root/'reader').mkdir()
            np.savez_compressed(root/'reader/features.npz',game_ids=np.array(['a','b','c']),embeddings=np.ones((3,2)))
            bam.write_json(root/'reader/features.json',{'inputs_sha256':bam.value_hash(rows),
                           'features_sha256':bam.file_hash(root/'reader/features.npz')})
            # No labels_test file exists: successful fit proves it was not opened.
            selected=bam.train_forecaster(root,config)
            self.assertEqual(selected['frozen']['name'],'zero_correction')
            bam.write_rows(root/'labels_test.jsonl',[{'game_id':'c','home_win':1}])
            result=bam.evaluate(root,config)
            self.assertEqual(result['metrics']['frozen_reader']['brier'],.25)
            self.assertEqual(result,bam.evaluate(root,config))
            with self.assertRaises(ValueError): bam.train_forecaster(root,config)
            altered=dict(result);altered['metrics']={'fake':'tampered'}
            bam.write_json(root/'evaluation.json',altered)
            with self.assertRaisesRegex(ValueError,'artifact changed'): bam.evaluate(root,config)


class OptimizerResumeIntegration(unittest.TestCase):
    def setUp(self):
        import torch
        from types import SimpleNamespace
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__();self.embedding=torch.nn.Embedding(256,4);self.dropout=torch.nn.Dropout(.2)
                self.output=torch.nn.Linear(4,256)
            def encode(self,ids):return self.dropout(self.embedding(ids))
            def forward(self,input_ids,attention_mask,labels,use_cache):
                logits=self.output(self.encode(input_ids))
                return SimpleNamespace(loss=torch.nn.functional.cross_entropy(logits.reshape(-1,256),labels.reshape(-1),ignore_index=-100))
            def save_pretrained(self,path,**kwargs):
                Path(path).mkdir(parents=True,exist_ok=True);torch.save(self.state_dict(),Path(path)/'weights.pt')
        self.Model=TinyModel;self.tokenizer=ToyTokenizer()
        self.config={'model_id':'fixture/model','model_revision':'a'*40,'minimum_eval_date':'2024-01-01','seed':230,
                     'research':{'epochs':2,'gradient_accumulation_steps':2,'learning_rate':.01},
                     'forecast':{'epochs':2,'gradient_accumulation_steps':2,'learning_rate':.01,'head_learning_rate':.01,
                                 'brier_loss_weight':1.,'explanation_loss_weight':.3,'max_seq_length':1024}}
        self.patches=contextlib.ExitStack()
        self.patches.enter_context(patch.object(bam,'_adapter_state',side_effect=lambda model:model.state_dict()))
        self.patches.enter_context(patch.object(bam,'_apply_adapter',side_effect=lambda model,state:model.load_state_dict(state,strict=True)))
        self.patches.enter_context(patch.object(bam,'_prefix_hidden',side_effect=lambda model,ids:
            torch.nn.functional.normalize(model.encode(torch.tensor(ids)).mean(0),dim=0)))
        self.patches.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.addCleanup(self.patches.close)

    def new(self,seed=23):
        import torch
        torch.manual_seed(seed);model=self.Model();head=torch.nn.Linear(4,1)
        torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
        return model,head

    def test_research_actual_loop_resumes_exact_data_dropout_and_optimizer(self):
        import torch
        rows=[research(str(i),'2024-01-01',target=True) for i in range(5)]
        encoded=[bam.completion_tokens(self.tokenizer,row['messages'],1024) for row in rows]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model,_=self.new()
            complete=bam._research_steps(model,self.tokenizer,rows,encoded,root/'full',self.config)
            expected=copy.deepcopy(model.state_dict());model,_=self.new()
            with self.assertRaisesRegex(RuntimeError,'checkpointed'):
                bam._research_steps(model,self.tokenizer,rows,encoded,root/'resume',self.config,stop_after_steps=2)
            restored,_=self.new(99)
            resumed=bam._research_steps(restored,self.tokenizer,rows,encoded,root/'resume',self.config)
            self.assertEqual(complete,resumed)
            for key,value in expected.items():self.assertTrue(torch.equal(value,restored.state_dict()[key]),key)

    def forecast_fixture(self,root):
        rows=[forecast(str(i),'2024-01-01','train',p=.2) for i in range(3)]+[forecast('d','2024-02-01','dev',p=.2)]
        labels={str(i):1 for i in range(3)};dev={'d':1}
        bam.write_rows(root/'forecast_explanations_train.jsonl',[{'game_id':str(i),'review_status':'accepted','explanation':'Synthetic source summary.'} for i in range(3)])
        return rows,labels,dev

    def test_forecast_loop_restores_head_and_selected_best_provenance(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);full=root/'full';partial=root/'resume'
            rows,labels,dev=self.forecast_fixture(full);self.forecast_fixture(partial)
            model,head=self.new();complete=bam._forecast_steps(full,self.config,rows,labels,dev,model,self.tokenizer,head)
            expected=copy.deepcopy(bam._joint_state(model,head));model,head=self.new()
            with self.assertRaisesRegex(RuntimeError,'checkpointed'):
                bam._forecast_steps(partial,self.config,rows,labels,dev,model,self.tokenizer,head,stop_after_steps=3)
            model,head=self.new(99)
            resumed=bam._forecast_steps(partial,self.config,rows,labels,dev,model,self.tokenizer,head)
            self.assertEqual(complete,resumed)
            self.assertEqual(resumed['name'],'forecast_lora')
            for key,value in expected.items():self.assertTrue(torch.equal(value,bam._joint_state(model,head)[key]),key)

    def test_forecast_changed_dev_labels_or_selected_weights_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows,labels,dev=self.forecast_fixture(root);model,head=self.new()
            with self.assertRaises(RuntimeError):
                bam._forecast_steps(root,self.config,rows,labels,dev,model,self.tokenizer,head,stop_after_steps=3)
            model,head=self.new(99)
            with self.assertRaisesRegex(ValueError,'identity changed'):
                bam._forecast_steps(root,self.config,rows,labels,{'d':0},model,self.tokenizer,head)
            candidate=root/'forecast_candidates/epoch_001/head.pt'
            self.assertTrue(candidate.exists());candidate.write_bytes(b'corrupted selection')
            with self.assertRaisesRegex(ValueError,'selected candidate changed'):
                bam._forecast_steps(root,self.config,rows,labels,dev,model,self.tokenizer,head)

    def test_separate_backward_matches_combined_objective_with_shared_dropout(self):
        import torch
        def gradients(separate):
            model,head=self.new();head.weight.data.fill_(.1)
            prefix=torch.nn.functional.normalize(model.encode(torch.tensor([1,2,3])).mean(0),dim=0)
            brier=(torch.sigmoid(head(prefix)).squeeze()-1).square()
            if separate:(brier/8).backward()
            ce=model(input_ids=torch.tensor([[4,5,6]]),attention_mask=None,labels=torch.tensor([[4,5,6]]),use_cache=False).loss
            if separate:(.3*ce/8).backward()
            else:((brier+.3*ce)/8).backward()
            return [p.grad.clone() for p in list(model.parameters())+list(head.parameters())]
        a,b=gradients(False),gradients(True)
        for combined,separate in zip(a,b):self.assertTrue(torch.allclose(combined,separate,atol=1e-7,rtol=1e-6))


if __name__=='__main__':
    unittest.main()
