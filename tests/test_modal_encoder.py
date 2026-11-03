"""Reject foreign or corrupted results at the encoder transfer boundary."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_modal_encoder as subject
from run_modal_encoder import validate_result
from run_game_note_research import value_hash, write_json, read_json, sha


class EncoderTransferTests(unittest.TestCase):
    def setUp(self):
        self.request={'model_revision':'pinned','model_files':{'weights':'hash'},
                      'versions':{'mlx':'fixed'},'prefill_step_size':512,'seed':230,
                      'rows':[{'game_id':'game1','chunks':[[1,2],[3,4]],'request_sha256':'prompt'}]}
        identity={k:self.request[k] for k in ('model_revision','model_files','versions','prefill_step_size','seed')}
        identity.update(backend='mlx_cuda',worker_sha256='worker')
        self.result={'execution':identity,'execution_sha256':value_hash(identity),
                     'rows':[{'game_id':'game1','input_sha256':value_hash(self.request['rows'][0]),
                              'request_sha256':'prompt','chunks':2,'execution_sha256':value_hash(identity),
                              'embedding':[1.]+[0.]*5119}]}

    def test_intact_result_validates_and_missing_chunks_or_changed_inputs_fail(self):
        validate_result(self.request,self.result,'worker')
        for key,value in [('chunks',1),('input_sha256','other'),('request_sha256','other'),
                          ('game_id','another'),('execution_sha256','unknown')]:
            result=deepcopy(self.result);result['rows'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_result(self.request,result,'worker')

    def test_hash_consistent_but_different_model_or_runtime_is_rejected(self):
        for key,value in [('model_revision','other'),('versions',{'mlx':'other'}),
                          ('worker_sha256','unknown'),('backend','other')]:
            result=deepcopy(self.result);result['execution'][key]=value
            result['execution_sha256']=value_hash(result['execution'])
            result['rows'][0]['execution_sha256']=result['execution_sha256']
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_result(self.request,result,'worker')

    def test_nonfinite_wrong_length_and_unnormalized_vectors_are_rejected(self):
        for vector in ([float('nan')]+[0.]*5119,[1.,0.],[2.]+[0.]*5119):
            result=deepcopy(self.result);result['rows'][0]['embedding']=vector
            with self.assertRaises(ValueError):
                validate_result(self.request,result,'worker')

    def test_saved_features_reject_mixed_backend_wrong_vectors_and_foreign_tokens(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):
                return [int(v) for v in messages[0]['content'].split()]
        for mutation in ('none','backend','vector','tokens'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);data=root/'data';report=root/'report';batch=root/'batch'
                (root/'scripts').mkdir();(root/'scripts/encode_research_mlx.py').write_text('fixed worker')
                chunks=[{'messages':[{'role':'user','content':'1 2'}]},
                        {'messages':[{'role':'user','content':'3 4'}]}]
                row={'game_id':'game1','chunks':chunks}
                request=deepcopy(self.request)
                request['rows'][0]['request_sha256']=value_hash(chunks)
                result=deepcopy(self.result)
                result['execution']['worker_sha256']=sha(root/'scripts/encode_research_mlx.py')
                result['execution_sha256']=value_hash(result['execution'])
                encoded=result['rows'][0]
                encoded.update(request_sha256=value_hash(chunks),input_sha256=value_hash(request['rows'][0]),
                               execution_sha256=result['execution_sha256'])
                write_json(report/'protocol.json',{'source':'fixed'})
                saved={'encoder_backend':'mlx_cuda','execution_sha256':result['execution_sha256'],
                       'protocol_sha256':sha(report/'protocol.json'),'request_sha256':value_hash(chunks),
                       'embedding':encoded['embedding']}
                features=np.asarray([encoded['embedding']])
                if mutation=='backend':
                    saved['encoder_backend']='mlx_metal'
                if mutation=='vector':
                    saved['embedding']=[0.,1.]+[0.]*5118
                if mutation=='tokens':
                    request['rows'][0]['chunks'][0]=[2,1]
                    encoded['input_sha256']=value_hash(request['rows'][0])
                write_json(data/'embeddings/game1.json',saved)
                write_json(batch/'request.json',request);write_json(batch/'result.json',result)
                write_json(report/'encoder_execution_manifest.json',{'backend':'mlx_cuda','games':1,
                    'artifacts':{str(p.relative_to(root)):sha(p) for p in (batch/'request.json',batch/'result.json')}})
                with patch.object(subject,'ROOT',root),patch.object(subject,'request_for',return_value=request), \
                     patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()):
                    if mutation=='none':
                        subject.validate_saved_embeddings({'model_path':'model'},[row],features,data,report)
                    else:
                        with self.assertRaises((ValueError,AssertionError)):
                            subject.validate_saved_embeddings({'model_path':'model'},[row],features,data,report)


if __name__=='__main__':
    unittest.main()
