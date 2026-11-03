"""No-cloud tests for transfer boundaries, immutable requests, and full prompts."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
import contextlib
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import bam_cloud as cloud
import bam_inference as inference


def archive(entries):
    output=io.BytesIO()
    with tarfile.open(fileobj=output,mode='w:gz') as stream:
        for name,content,kind in entries:
            info=tarfile.TarInfo(name)
            if kind=='file':
                content=content.encode();info.size=len(content)
                stream.addfile(info,io.BytesIO(content))
            else:
                info.type=tarfile.SYMTYPE;info.linkname=content
                stream.addfile(info)
    return output.getvalue()


class TransferBoundaries(unittest.TestCase):
    def extract(self,payload,directory,**kwargs):
        return cloud.safe_extract(payload,hashlib.sha256(payload).hexdigest(),directory,**kwargs)

    def test_normal_transfer_and_identical_resume(self):
        payload=archive([('research_inputs.jsonl','{}\n','file'),('metadata/config.json','{}','file')])
        with tempfile.TemporaryDirectory() as directory:
            first=self.extract(payload,directory)
            self.assertEqual(first,self.extract(payload,directory))
            self.assertEqual((Path(directory)/'research_inputs.jsonl').read_text(),'{}\n')

    def test_traversal_absolute_symlink_and_duplicate_rejected(self):
        cases=[ [('..'+'/escape','bad','file')], [('/absolute','bad','file')],
               [('link','/tmp/outside','symlink')], [('same','a','file'),('same','b','file')],
               [('.venv/bin/python','bad','file')], [('a\\b','bad','file')] ]
        with tempfile.TemporaryDirectory() as directory:
            for entries in cases:
                with self.assertRaises(ValueError): self.extract(archive(entries),directory)

    def test_invalid_later_member_causes_no_partial_mutation(self):
        payload=archive([('okay','a','file'),('../bad','b','file')])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError): self.extract(payload,directory)
            self.assertFalse((Path(directory)/'okay').exists())

    def test_changed_existing_file_and_wrong_hash_fail(self):
        payload=archive([('same','a','file')])
        with tempfile.TemporaryDirectory() as directory:
            self.extract(payload,directory)
            with self.assertRaises(ValueError): self.extract(archive([('same','b','file')]),directory)
            with self.assertRaises(ValueError): cloud.safe_extract(payload,'0'*64,directory)
            self.assertEqual((Path(directory)/'same').read_text(),'a')

    def test_expansion_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError): self.extract(archive([('large','abcd','file')]),directory,max_bytes=3)

    def test_run_and_model_identity(self):
        for identity in ('../run','a/b','','.hidden'):
            with self.assertRaises(ValueError): cloud.checked_id(identity)
        self.assertEqual(cloud.checked_id('run-2026_1'),'run-2026_1')
        with self.assertRaises(ValueError): cloud.checked_model({'model_id':'Qwen/Model','model_revision':'main'})
        spec={'model_id':'Qwen/Model','model_revision':'a'*40}
        self.assertEqual(cloud.checked_model(spec),spec)

    def test_explicit_parallelism_routes_and_rejects_wrong_hardware(self):
        spec={'model_id':'owner/model'}
        self.assertEqual(cloud.generation_function(spec,{}),'generate')
        config={'generation_by_model':{'owner/model':{'tensor_parallel_size':2}}}
        self.assertEqual(cloud.generation_function(spec,config),'generate_heavy')
        with self.assertRaises(ValueError):
            cloud._generate([],spec,'run',config,'unused',function_name='generate')
        with self.assertRaises(ValueError): cloud.generation_function(spec,{'generation':{'tensor_parallel_size':True}})

    def test_immutable_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'request.json'
            cloud.lock_request(path,{'data':1})
            cloud.lock_request(path,{'data':1})
            with self.assertRaises(ValueError): cloud.lock_request(path,{'data':2})

    def test_result_manifest_excludes_adapter_unless_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'adapter').mkdir()
            (root/'adapter/adapter_model.safetensors').write_text('weights')
            (root/'report.json').write_text('{}')
            self.assertEqual(set(cloud.artifact_manifest(root)),{'report.json'})
            self.assertEqual(len(cloud.artifact_manifest(root,True)),2)

    def test_parent_commit_observes_child_artifact_before_progress_line(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            committed=[]
            code="from pathlib import Path; Path('artifact.json').write_text('{}'); print('checkpoint saved',flush=True)"
            cloud._run_logged([sys.executable,'-u','-c',code],root,root/'execution.log',
                              commit=lambda:committed.append((root/'artifact.json').read_text()))
            self.assertEqual(committed,['{}'])

    def test_warning_storm_does_not_commit_every_line_and_markers_commit_immediately(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); committed=[]
            code="import json; [print('warning',flush=True) for _ in range(300)]; print(json.dumps({'checkpoint_complete':{'step':1}}),flush=True); print(json.dumps({'completed':2,'total':2,'batch_seconds':.1}),flush=True)"
            with contextlib.redirect_stdout(io.StringIO()), patch.object(cloud.time,'monotonic',return_value=100):
                cloud._run_logged([sys.executable,'-u','-c',code],root,root/'execution.log',commit=lambda:committed.append(True))
            self.assertEqual(len(committed),3)  # checkpoint, completed batch, final
            self.assertEqual((root/'execution.log').read_text().count('warning'),300)

    def test_failed_child_commits_final_log_and_checkpoint_tensors_are_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);committed=[]
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(RuntimeError):
                cloud._run_logged([sys.executable,'-c',"print('failure'); raise SystemExit(2)"],root,root/'execution.log',commit=lambda:committed.append(True))
            self.assertTrue(committed)
            (root/'checkpoints').mkdir();(root/'checkpoints/optimizer.pt').write_text('state')
            (root/'checkpoints/manifest.json').write_text('{}')
            self.assertNotIn('checkpoints/optimizer.pt',cloud.artifact_manifest(root))
            self.assertIn('checkpoints/manifest.json',cloud.artifact_manifest(root))

    def test_ordinary_log_commit_is_periodic(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);committed=[]
            with contextlib.redirect_stdout(io.StringIO()),patch.object(cloud.time,'monotonic',side_effect=[0,25,25,25]):
                cloud._run_logged([sys.executable,'-c',"print('first'); print('second')"],root,root/'execution.log',
                                  commit=lambda:committed.append(True),commit_interval=20)
            self.assertEqual(len(committed),2)  # one elapsed-time commit plus final


class PromptBoundaries(unittest.TestCase):
    def test_parallelism_and_explicit_kernel_selection_are_validated(self):
        spec={'model_id':'owner/model'}
        config={'generation':{'tensor_parallel_size':2,
            'kernel_config':{'linear_backend':'cutlass','moe_backend':'triton'}}}
        self.assertEqual(inference.generation_options(config,spec)['tensor_parallel_size'],2)
        for options in ({'tensor_parallel_size':3},{'tensor_parallel_size':True},
                        {'kernel_config':{'linear_backend':'unknown'}}):
            with self.assertRaises(ValueError): inference.generation_options({'generation':options},spec)

    def request(self,**kwargs):
        return {'request_id':'r1','messages':[{'role':'system','content':'S'},{'role':'user','content':'U'}],**kwargs}

    def test_max_output_bound_and_unique_ids(self):
        with self.assertRaises(ValueError): inference.validate_requests([self.request(max_tokens=8193)])
        with self.assertRaises(ValueError): inference.validate_requests([self.request(),self.request()])
        inference.validate_requests([self.request(max_tokens=8192)])

    def test_context_failure_preserves_request_hash_and_complete_token_count(self):
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs): return list(range(20))
        row=self.request(max_tokens=10)
        result=inference.prepare_requests([row],Tokenizer(),25)[0]
        self.assertEqual(result['status'],'error')
        self.assertEqual(result['prompt_tokens'],20)
        self.assertEqual(result['request_sha256'],inference.value_hash(row))
        valid=inference.prepare_requests([row],Tokenizer(),30)[0]
        self.assertEqual(len(valid['input_ids']),20)
        self.assertNotIn('status',valid)


if __name__=='__main__':
    unittest.main()
