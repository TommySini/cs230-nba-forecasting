"""Encode prepared token sequences with the unchanged frozen MLX checkpoint."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def write(path,value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main(request_path, output):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    request=json.loads(Path(request_path).read_text())
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    model_path=Path('/models')/Path(request['model_path']).name
    for relative,expected in request['model_files'].items():
        if digest(model_path/Path(relative).name)!=expected:
            raise ValueError('Encoder checkpoint changed')
    versions={name:version(name) for name in request['versions']}
    if versions!=request['versions']:
        raise ValueError('Encoder runtime versions changed')
    if mx.default_device()!=mx.gpu:
        raise ValueError('CUDA GPU encoder required')
    mx.set_memory_limit(request['memory_limit_gb']*1024**3)
    mx.set_cache_limit(4*1024**3)
    mx.random.seed(request['seed'])
    model,_=load(str(model_path));model.eval()
    identity={'backend':'mlx_cuda','versions':versions,'model_files':request['model_files'],
              'model_revision':request['model_revision'],'worker_sha256':digest(__file__),
              'prefill_step_size':request['prefill_step_size'],'seed':request['seed'],
              'pooling':'Mean of final-token vectors from all complete chunks, then L2 normalization in float64',
              'device_info':mx.device_info()}
    identity_hash=value_hash(identity)
    write(output/'execution.json',identity)
    results=[]
    for row in request['rows']:
        if set(row)!={'game_id','chunks','request_sha256'}:
            raise ValueError('Unexpected encoder payload fields')
        path=output/(row['game_id'].replace(':','-')+'.json')
        if path.exists():
            result=json.loads(path.read_text())
            if result['input_sha256']!=value_hash(row) or result['execution_sha256']!=identity_hash:
                raise ValueError('Cached encoder result changed')
            results.append(result);continue
        started=time.monotonic();vectors=[]
        for ids in row['chunks']:
            if not ids or len(ids)>request['token_budget'] or any(type(t)is not int or t<0 for t in ids):
                raise ValueError('Invalid encoder input; no truncation permitted')
            cache=make_prompt_cache(model)
            for start in range(0,len(ids),request['prefill_step_size']):
                hidden=model.model(mx.array(ids[start:start+request['prefill_step_size']])[None],cache=cache)
                mx.eval(hidden)
            vectors.append(np.asarray(hidden[0,-1].tolist(),dtype=np.float64))
            del cache,hidden;mx.clear_cache()
        pooled=np.asarray(vectors,dtype=np.float64).mean(axis=0)
        norm=np.linalg.norm(pooled)
        if pooled.shape!=(5120,) or not np.isfinite(pooled).all() or norm<=0:
            raise ValueError('Invalid contextual vector')
        result={'game_id':row['game_id'],'request_sha256':row['request_sha256'],
                'input_sha256':value_hash(row),'execution_sha256':identity_hash,
                'chunks':len(vectors),'embedding':(pooled/norm).tolist(),
                'seconds':time.monotonic()-started}
        write(path,result);results.append(result)
        print(json.dumps({'encoded':len(results),'total':len(request['rows']),
                          'game_id':row['game_id'],'seconds':result['seconds']}),flush=True)
    write(output/'result.json',{'execution':identity,'execution_sha256':identity_hash,'rows':results})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request');parser.add_argument('output')
    args=parser.parse_args();main(args.request,args.output)
