"""A missing verification verdict must produce actionable retry feedback."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from research_vllm_worker import validation_feedback
import research_vllm_worker as worker


class VerificationRetryTests(unittest.TestCase):
    def test_missing_last_of_eight_is_explicit(self):
        reply=json.dumps({'decisions':[{'id':str(i),'verdict':'supported','reason':'source'} for i in range(7)]})
        message=validation_feedback(ValueError('Missing verification decisions'),reply,{'facts':[{}]*8},'verify')
        self.assertIn('exactly 8 decisions',message)
        self.assertIn('Missing IDs: ["7"]',message)
        self.assertIn('["0", "1", "2", "3", "4", "5", "6", "7"]',message)

    def test_malformed_reply_requires_all_ids(self):
        message=validation_feedback(ValueError('Invalid JSON'),'not json',{'facts':[{},{}]},'verify')
        self.assertIn('Missing IDs: ["0", "1"]',message)

    def test_extraction_error_is_not_changed(self):
        self.assertEqual(validation_feedback(ValueError('Invalid source line'),'{}',{},'extract'),'Invalid source line')

    def test_actual_retry_preserves_old_attempt_and_requires_eighth_verdict(self):
        task={'id':'test-unit-v000','facts':[{'id':f'fact-{i}'} for i in range(8)]}
        messages=[]
        def tokenize(items,**kwargs):
            messages.append(items)
            return [1,2]
        calls=[]
        def generate(prompts,sampling,use_tqdm):
            count=7 if not calls else 8
            calls.append(count)
            text=json.dumps({'decisions':[{'id':str(i),'verdict':'unclear','reason':'Not established'} for i in range(count)]})
            return [SimpleNamespace(outputs=[SimpleNamespace(text=text,finish_reason='stop',token_ids=[1])])]
        config={'verification_max_tokens':100,'max_prompt_tokens':1000,'seed':230,'max_attempts':3}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'protocol.json').write_text('{}')
            old=root/'attempts/verify/test-unit-v000-a0.json'
            old.parent.mkdir(parents=True)
            old.write_text('original failed attempt')
            with patch.dict(sys.modules,{'vllm':SimpleNamespace(SamplingParams=lambda **kwargs:kwargs)}),patch.object(worker.r,'DATA',root),patch.object(worker.r,'REPORT',root),patch.object(worker.r,'verification_messages',return_value=[{'role':'user','content':'Check all eight claims'}]):
                worker.execute([task],'verify',config,SimpleNamespace(generate=generate),SimpleNamespace(eos_token_id=0,apply_chat_template=tokenize),'new-execution-hash')
            response=json.loads((root/'verify/test-unit-v000.json').read_text())
            self.assertEqual(calls,[7,8])
            self.assertEqual(response['status'],'complete')
            self.assertEqual(len(response['parsed']['decisions']),8)
            self.assertIn('Missing IDs: ["7"]',messages[-1][-1]['content'])
            self.assertEqual(old.read_text(),'original failed attempt')


if __name__=='__main__':
    unittest.main()
