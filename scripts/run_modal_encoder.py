"""Submit immutable, label-free encoder inputs to the existing project deployment."""
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from run_game_note_research import ROOT, read_json, write_json, sha, value_hash


def request_for(config, rows):
    manifest=read_json(ROOT/'data/probes/research_compute_bundle/probe/manifest.json')
    protocol=read_json(ROOT/'reports/game_note_research/protocol.json')
    if manifest['model_files']!=protocol['model_sha256']:
        raise ValueError('Encoder checkpoint manifest differs from the frozen research checkpoint')
    return {**{key:config[key] for key in ('model_path','model_revision','memory_limit_gb','seed','prefill_step_size')},
            'model_files':manifest['model_files'],'versions':manifest['versions'],
            'token_budget':config['embedding_token_budget'],'rows':rows}


def validate_result(request, result, expected_worker_sha):
    identity=result['execution']
    if result['execution_sha256']!=value_hash(identity):
        raise ValueError('Encoder execution identity changed')
    for key in ('model_revision','model_files','versions','prefill_step_size','seed'):
        if identity[key]!=request[key]:
            raise ValueError('Encoder execution differs from request: '+key)
    if identity['backend']!='mlx_cuda' or identity['worker_sha256']!=expected_worker_sha:
        raise ValueError('Unexpected encoder backend or worker')
    if [row['game_id'] for row in result['rows']]!=[row['game_id'] for row in request['rows']]:
        raise ValueError('Encoder output membership/order changed')
    for source,row in zip(request['rows'],result['rows']):
        if (row['input_sha256']!=value_hash(source) or row['request_sha256']!=source['request_sha256']
            or row['chunks']!=len(source['chunks']) or row['execution_sha256']!=result['execution_sha256']):
            raise ValueError('Encoder response provenance changed')
        vector=np.asarray(row['embedding'],dtype=np.float64)
        if vector.shape!=(5120,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector),1,atol=1e-12,rtol=0):
            raise ValueError('Invalid normalized embedding')


def submit(request, output, environment='main'):
    import modal
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    expected=sha(ROOT/'scripts/encode_research_mlx.py')
    if (output/'request.json').exists() and read_json(output/'request.json')!=request:
        raise ValueError('Cannot change a saved encoder request')
    write_json(output/'request.json',request)
    if (output/'result.json').exists():
        result=read_json(output/'result.json')
        validate_result(request,result,expected)
        return result
    call_path=output/'call.json'
    if call_path.exists():
        saved=read_json(call_path)
        if saved['worker_sha256']!=expected or saved['request_sha256']!=value_hash(request):
            raise ValueError('Cannot resume a changed encoder call')
        call=modal.FunctionCall.from_id(saved['function_call_id'])
    else:
        function=modal.Function.from_name('cs230-nba-research-vllm','encode_reports',environment_name=environment)
        run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        call=function.spawn(request,expected,run_id)
        write_json(call_path,{'function_call_id':call.object_id,'run_id':run_id,
                             'worker_sha256':expected,'request_sha256':value_hash(request)})
        print('Submitted encoder '+call.object_id,flush=True)
    started=time.monotonic()
    result=call.get()
    write_json(output/'result.json',result)
    validate_result(request,result,expected)
    print(f'Validated {len(result["rows"])} cloud embeddings in {time.monotonic()-started:.1f}s',flush=True)
    return result


def encode_all(config, rows, data, report):
    from concurrent.futures import ThreadPoolExecutor
    from transformers import AutoTokenizer
    from run_game_note_research import encode_messages
    plan=read_json(report/'encoder_execution_plan.json')
    if plan['backend']!='mlx_cuda' or plan['workers']!=2:
        raise ValueError('Unexpected frozen encoder execution plan')
    tokenizer=AutoTokenizer.from_pretrained(ROOT/config['model_path'],local_files_only=True)
    inputs=[]
    for row in rows:
        chunks=[]
        for chunk in row['chunks']:
            ids=encode_messages(tokenizer,chunk['messages'])
            if len(ids)!=chunk['tokens'] or len(ids)>config['embedding_token_budget']:
                raise ValueError('Cloud encoder tokenization differs; no truncation permitted')
            chunks.append(ids)
        inputs.append({'game_id':row['game_id'],'chunks':chunks,'request_sha256':value_hash(row['chunks'])})
    directory=ROOT/plan['output_directory']
    # Both bounded workers use identical checkpoint, runtime, pooling and inputs.
    requests=[request_for(config,inputs[index::2]) for index in range(2)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(submit,request,directory/f'batch_{index}') for index,request in enumerate(requests)]
        results=[future.result() for future in futures]
    combined={}
    artifacts={}
    for index,result in enumerate(results):
        for name in ('request.json','result.json','call.json'):
            file=directory/f'batch_{index}'/name
            artifacts[str(file.relative_to(ROOT))]=sha(file)
        for encoded in result['rows']:
            if encoded['game_id'] in combined:
                raise ValueError('Duplicated game in cloud encoder outputs')
            combined[encoded['game_id']]=encoded
    if set(combined)!={row['game_id'] for row in rows}:
        raise ValueError('Cloud encoder cohort incomplete')
    for row in rows:
        encoded=combined[row['game_id']]
        path=data/'embeddings'/(row['game_id'].replace(':','-')+'.json')
        saved={'game_id':row['game_id'],'request_sha256':encoded['request_sha256'],
               'protocol_sha256':sha(report/'protocol.json'),'chunks':encoded['chunks'],
               'facts':row['facts'],'embedding':encoded['embedding'],
               'encoder_backend':'mlx_cuda','execution_sha256':encoded['execution_sha256']}
        if path.exists() and read_json(path)!=saved:
            raise ValueError('Refusing to mix or replace a different saved encoder vector')
        write_json(path,saved)
    write_json(report/'encoder_execution_manifest.json',{
        'backend':'mlx_cuda','games':len(rows),'workers':2,'artifacts':artifacts,
        'executions':{result['execution_sha256']:result['execution'] for result in results},
        'source_worker_sha256':sha(ROOT/'scripts/encode_research_mlx.py')})


def validate_saved_embeddings(config, rows, features, data, report):
    from transformers import AutoTokenizer
    from run_game_note_research import encode_messages
    execution=read_json(report/'encoder_execution_manifest.json')
    if execution['backend']!='mlx_cuda' or execution['games']!=len(rows):
        raise ValueError('Cloud encoder manifest differs from cohort')
    for relative,expected in execution['artifacts'].items():
        if sha(ROOT/relative)!=expected:
            raise ValueError('Frozen cloud encoder artifact changed')
    tokenizer=AutoTokenizer.from_pretrained(ROOT/config['model_path'],local_files_only=True)
    observed={}
    for relative in execution['artifacts']:
        if Path(relative).name!='result.json':
            continue
        directory=(ROOT/relative).parent
        request,result=read_json(directory/'request.json'),read_json(directory/'result.json')
        validate_result(request,result,sha(ROOT/'scripts/encode_research_mlx.py'))
        if request_for(config,request['rows'])!=request:
            raise ValueError('Cloud encoder execution settings differ from frozen config')
        for source,encoded in zip(request['rows'],result['rows']):
            if source['game_id'] in observed:
                raise ValueError('Duplicate cloud embedding')
            observed[source['game_id']]=(source,encoded)
    if set(observed)!={row['game_id'] for row in rows}:
        raise ValueError('Incomplete cloud embedding cohort')
    for index,row in enumerate(rows):
        source,encoded=observed[row['game_id']]
        expected={'game_id':row['game_id'],'request_sha256':value_hash(row['chunks']),
                  'chunks':[encode_messages(tokenizer,chunk['messages']) for chunk in row['chunks']]}
        if source!=expected:
            raise ValueError('Cloud token inputs differ from actual forecast prompts')
        saved=read_json(data/'embeddings'/(row['game_id'].replace(':','-')+'.json'))
        if (saved.get('encoder_backend')!='mlx_cuda' or saved.get('execution_sha256')!=encoded['execution_sha256']
            or saved['protocol_sha256']!=sha(report/'protocol.json') or saved['request_sha256']!=expected['request_sha256']):
            raise ValueError('Mixed or untraceable cloud/local embeddings')
        np.testing.assert_array_equal(saved['embedding'],encoded['embedding'])
        np.testing.assert_array_equal(features[index],encoded['embedding'])


def probe():
    config=read_json(ROOT/'config/game_note_research.json')
    directory=ROOT/'data/probes/encoder_device_probe'
    inputs=read_json(directory/'inputs.json')
    if inputs['split']!='train':
        raise ValueError('Encoder timing probe must use a training packet')
    request=request_for(config,[{'game_id':inputs['game_id'],'chunks':inputs['token_ids'],
                                'request_sha256':value_hash(inputs['chunks'])}])
    result=submit(request,directory/'modal')
    local=read_json(directory/'local_result.json')
    pooled=np.asarray(local['vectors'],dtype=np.float64).mean(axis=0)
    pooled/=np.linalg.norm(pooled)
    remote=np.asarray(result['rows'][0]['embedding'])
    comparison={'game_id':inputs['game_id'],'split':'train','model_revision':config['model_revision'],
                'local_seconds':sum(local['chunk_seconds']), 'cloud_seconds':result['rows'][0]['seconds'],
                'cosine_similarity':float(np.dot(pooled,remote)),
                'max_absolute_component_difference':float(np.max(np.abs(pooled-remote))),
                'local_result_sha256':sha(directory/'local_result.json'),
                'cloud_result_sha256':sha(directory/'modal/result.json'),
                'scope':'Training-only compute probe. Same weights, tokens and pooling; different hardware arithmetic. No labels or head fitting.'}
    write_json(directory/'comparison.json',comparison)
    import json
    print(json.dumps(comparison,indent=2),flush=True)


if __name__=='__main__':
    probe()
