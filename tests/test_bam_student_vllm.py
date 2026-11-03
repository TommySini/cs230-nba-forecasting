import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as N
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import bam_student_vllm as b
import bam_training as t
from bam_student_vllm_probe_worker import prompt


def output(ids=(1,2),tokens=(9,),finish='stop',stop=9,adapter=3):
    return N(prompt_token_ids=list(ids),lora_request=N(lora_int_id=adapter) if adapter else None,
             outputs=[N(token_ids=list(tokens),text='answer',finish_reason=finish,stop_reason=stop)],finished=True)


class TestStudentVLLM(unittest.TestCase):
    def test_process_local_adapter_id_fits_vllm_int32(self):
        import numpy as np
        for digest in ('f'*64, '9'*64):
            self.assertEqual(int(np.int32(b.adapter_handle(digest))),1)
        self.assertIsNone(b.adapter_handle(None))
        with self.assertRaises(ValueError):b.adapter_handle('short')
        with self.assertRaisesRegex(ValueError,'collide'):
            b.validate_adapter_bindings(['a'*64,'b'*64])
        self.assertEqual(b.validate_adapter_bindings(['a'*64]*4),{1:'a'*64})

    def test_eos_is_observed_never_fabricated(self):
        self.assertEqual(b.normalize_output(output(),[1,2],[9],32,3)['status'],'complete')
        record=b.normalize_output(output(tokens=(4,),stop=9),[1,2],[9],32,3)
        self.assertEqual(record['raw_token_ids'],[4]);self.assertEqual(record['verified_eos_token_id'],9)
        for changed in (output(tokens=(4,),stop=None),output(finish='length'),output(stop='EOS',tokens=(4,))):
            self.assertEqual(b.normalize_output(changed,[1,2],[9],32,3)['status'],'incomplete')

    def test_changed_tokens_or_missing_adapter_reject(self):
        for changed in (output(ids=(2,1)),output(adapter=None)):
            with self.assertRaises(ValueError):b.normalize_output(changed,[1,2],[9],32,3)

    def test_context_fails_before_backend(self):
        engine=object.__new__(b.StudentVLLM);engine.max_context=5;engine.adapter=None
        with self.assertRaisesRegex(ValueError,'truncation'):
            engine.generate([{'input_ids':[1,2,3]}],3)

    def test_adapter_identity_is_exact_and_architecture_bound(self):
        config={'model_id':'model','research':{'lora_rank':8,'lora_alpha':16}}
        spec={'base_model_name_or_path':'model','peft_type':'LORA','task_type':'CAUSAL_LM','r':8,'lora_alpha':16,
              'bias':'none','target_modules':['q_proj','k_proj','v_proj','o_proj']}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'adapter_config.json').write_text(json.dumps(spec));(p/'adapter_model.safetensors').write_bytes(b'fixture')
            digest=t.value_hash({x.name:t.file_hash(x) for x in p.iterdir()})
            self.assertEqual(b.adapter_identity(p,digest,config)['rank'],8)
            (p/'adapter_model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):b.adapter_identity(p,digest,config)

    def test_trained_target_future_and_diagnostic_boundaries(self):
        config={'model_id':'model','model_revision':'a'*40,'minimum_eval_date':'2025-01-01'}
        row={'game_id':'g','game_date_et':'2025-02-01','split':'train','module':'availability','source_hashes':['s'],
             'messages':[{'role':'system','content':'source only'},{'role':'user','content':'source'}]}
        provenance={'mode':'expanding_fold','base_model':t._model_config(config),'training_game_ids':['g'],
                    'training_max_date':'2025-01-01','adapter_sha256':'sha'}
        manifest={**provenance,'adapter_files_sha256':'sha'}
        with self.assertRaisesRegex(ValueError,'target'):b.validate_lineage([row],provenance,config,{'sha256':'sha'},training_manifest=manifest)
        provenance.update(mode='synthetic_trained_adapter',training_game_ids=[],training_max_date=None)
        with self.assertRaisesRegex(ValueError,'masquerade'):b.validate_lineage([row],provenance,config,{'sha256':'sha'},diagnostic=True)
        row['synthetic_only']=True;b.validate_lineage([row],provenance,config,{'sha256':'sha'},diagnostic=True)

    def test_real_pinned_tokenizer_prompt_contract(self):
        from transformers import AutoTokenizer
        path=Path(__file__).resolve().parents[1]/'data/models/bam_tokenizer/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8'
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        for length in (4096,8192):
            item=prompt(tokenizer,length,1)
            self.assertTrue(all(type(x) is int for x in item['input_ids']))
            self.assertEqual(item['input_ids'],t.chat_token_ids(tokenizer,item['messages'],generation_prompt=True))
            self.assertLess(abs(len(item['input_ids'])-length),160)

    def test_durable_small_chunks_resume_without_regeneration(self):
        config={'model_id':'model','model_revision':'a'*40,'minimum_eval_date':'2025-01-01',
                'research':{'max_new_tokens':8,'output_token_budgets':[8,16],'max_seq_length':100}}
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):return [1,2,3]
        class Backend:
            adapter=None;max_context=100;tokenizer=Tokenizer()
            identity={'training_base_precision':'nf4','fixture':True}
            calls=[]
            def generate(self,items,budget):
                self.calls.append(len(items))
                return [{'status':'complete','text':'source-only report','request_sha256':i['request_sha256'],
                         'prompt_token_ids_sha256':t.value_hash(i['input_ids']),
                         'execution_sha256':t.value_hash(self.identity),'output_budget':budget} for i in items]
        rows=[{'game_id':str(i),'game_date_et':'2025-02-01','split':'train','module':'availability','source_hashes':['s'],
               'messages':[{'role':'system','content':'source only'},{'role':'user','content':'source'}]} for i in range(6)]
        provenance={'mode':'cold_prompted','base_model':t._model_config(config),'training_game_ids':[],
                    'training_max_date':None,'adapter_sha256':None}
        backend=Backend()
        with tempfile.TemporaryDirectory() as root:
            result=b.generate_reports(root,rows,backend,provenance,config)
            self.assertTrue(result['complete']);self.assertEqual(backend.calls,[4,2])
            b.generate_reports(root,rows,backend,provenance,config)
            self.assertEqual(backend.calls,[4,2])
            changed=copy.deepcopy(rows);changed[0]['messages'][1]['content']='other source'
            with self.assertRaises(ValueError):b.generate_reports(root,changed,backend,provenance,config)
            source=Path(root)/'attempts'/t.value_hash(rows[0])/'8.json'
            target=Path(root)/'attempts'/t.value_hash(rows[1])/'8.json'
            target.write_bytes(source.read_bytes())
            with self.assertRaisesRegex(ValueError,'different request'):
                b.generate_reports(root,rows,backend,provenance,config)

    def test_attempt_binding_rejects_changed_prompt_runtime_budget(self):
        item={'request_sha256':'request','input_ids':[1,2]};backend=N(identity={'backend':'expected'})
        record={'request_sha256':'request','prompt_token_ids_sha256':t.value_hash([1,2]),
                'execution_sha256':t.value_hash(backend.identity),'output_budget':8,'status':'complete'}
        b.validate_attempt_binding(record,item,backend,8)
        for field in ('request_sha256','prompt_token_ids_sha256','execution_sha256','output_budget'):
            with self.assertRaises(ValueError):b.validate_attempt_binding(record|{field:'changed'},item,backend,8)


if __name__=='__main__':unittest.main()
