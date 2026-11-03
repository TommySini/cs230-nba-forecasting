"""Label-agnostic vLLM batch worker with immutable, resumable per-request artifacts."""
from __future__ import annotations
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import time

from bam_training import chat_token_ids, file_hash, value_hash, write_json, write_rows


def validate_requests(requests):
    seen = set()
    for row in requests:
        identity = row.get('request_id', row.get('id'))
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError('Every request needs a unique string request_id (or id)')
        seen.add(identity)
        messages = row.get('messages')
        if not isinstance(messages, list) or not messages:
            raise ValueError('Missing chat messages')
        if any(m.get('role') not in ('system','user','assistant') or not isinstance(m.get('content'),str)
               or not m['content'].strip() for m in messages):
            raise ValueError('Invalid chat messages')
        if messages[-1]['role'] != 'user':
            raise ValueError('Generation prompt must finish with a user message')
        budget = row.get('max_tokens',4096)
        if not isinstance(budget,int) or not 1 <= budget <= 8192:
            raise ValueError('max_tokens must be within 1..8192; no implicit output expansion')
        if 'json_schema' in row and not isinstance(row['json_schema'],dict):
            raise ValueError('json_schema must be an object')
    return requests


def generation_options(config, model_spec):
    """Model-specific sampling is explicit in the immutable experiment config."""
    options = dict(config.get('generation', config))
    overrides = config.get('generation_by_model', {}).get(model_spec['model_id'], {})
    options.update(overrides)
    reserve = options.get('reasoning_token_reserve', 0)
    if type(reserve) is not int or not 0 <= reserve <= 8192:
        raise ValueError('Reasoning reserve must be an explicit integer within 0..8192')
    if options.get('expects_reasoning', False) and reserve == 0:
        raise ValueError('Reasoning models require an explicit output-token reserve')
    if type(options.get('tensor_parallel_size',1)) is not int or options.get('tensor_parallel_size',1) not in (1,2):
        raise ValueError('Only explicit one- or two-GPU tensor parallelism is supported')
    if options.get('kernel_config') is not None and options['kernel_config'] != {'linear_backend':'cutlass','moe_backend':'triton'}:
        raise ValueError('Unsupported explicit inference kernel configuration')
    if type(options.get('nccl_only',False)) is not bool:
        raise ValueError('nccl_only must be an explicit boolean')
    return options


def separate_completion(raw, finish_reason, *, expects_reasoning=False):
    """Keep transport/analysis artifacts apart from the complete final JSON answer.

    Never salvage an unfinished answer or infer a missing verdict. Removing one
    complete Markdown fence changes transport formatting only; raw bytes remain.
    """
    result = {'raw_text': raw, 'text': '', 'reasoning_text': None,
              'removed_markdown_fence': False}
    if finish_reason != 'stop':
        return {**result, 'status': 'error', 'error': 'Generation did not finish naturally'}
    answer = raw.strip()
    if expects_reasoning:
        if answer.count('</think>') != 1:
            return {**result, 'status': 'error', 'error': 'Missing or ambiguous completed reasoning boundary'}
        reasoning, answer = answer.split('</think>', 1)
        if reasoning.startswith('<think>'):
            reasoning = reasoning[len('<think>'):]
        if '<think>' in reasoning or '<think>' in answer or '</think>' in answer:
            return {**result, 'status': 'error', 'error': 'Malformed reasoning boundary'}
        result['reasoning_text'] = reasoning.strip()
        answer = answer.strip()
    elif '<think>' in answer or '</think>' in answer:
        return {**result, 'status': 'error', 'error': 'Unexpected reasoning output requires explicit model configuration'}
    fence = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', answer, flags=re.DOTALL)
    if fence:
        answer = fence.group(1).strip()
        result['removed_markdown_fence'] = True
    if not answer:
        return {**result, 'status': 'error', 'error': 'No final answer after completed reasoning'}
    return {**result, 'text': answer, 'status': 'complete'}


def prepare_requests(requests, tokenizer, max_model_len, *, reasoning_token_reserve=0):
    """Tokenize complete requests; oversize inputs become explicit error rows."""
    validate_requests(requests)
    prepared = []
    for row in requests:
        ids = chat_token_ids(tokenizer, row['messages'], generation_prompt=True)
        identity = row.get('request_id',row.get('id'))
        item = {'request_id':identity,'request_sha256':value_hash(row),'request':row,
                'prompt_token_ids_sha256':value_hash(ids),'prompt_tokens':len(ids),
                'max_tokens':row.get('max_tokens',4096) + reasoning_token_reserve,
                'requested_answer_tokens':row.get('max_tokens',4096),
                'reasoning_token_reserve':reasoning_token_reserve}
        if not ids or len(ids)+item['max_tokens']>max_model_len:
            item.update(status='error',error='Complete prompt plus requested output exceeds model context; truncation prohibited')
        else:
            item['input_ids']=ids
        prepared.append(item)
    return prepared


def run_requests(root, requests, model_manifest, config, *, commit=None):
    options = generation_options(config, model_manifest['model_spec'])
    runtime_environment = {}
    if options.get('nccl_only'):
        runtime_environment = {'VLLM_ALLREDUCE_USE_FLASHINFER':'0','VLLM_ALLREDUCE_USE_SYMM_MEM':'0'}
        os.environ.update(runtime_environment)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    root = Path(root)
    model_path = Path(model_manifest['snapshot_path'])
    for name,digest in model_manifest['files'].items():
        if file_hash(model_path/name)!=digest:
            raise ValueError('Pinned checkpoint content changed: '+name)
    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    settings = dict(dtype='float16' if model_manifest['model_spec'].get('quantization')=='awq' else 'bfloat16',
        max_model_len=options.get('max_model_len',32768), max_num_seqs=options.get('max_num_seqs',12),
        max_num_batched_tokens=options.get('max_num_batched_tokens',8192),
        gpu_memory_utilization=options.get('gpu_memory_utilization',.88),
        tensor_parallel_size=options.get('tensor_parallel_size',1),
        seed=config.get('seed',230), generation_config='vllm', enforce_eager=True,
        enable_prefix_caching=True, enable_chunked_prefill=True, trust_remote_code=False)
    if options.get('kernel_config') is not None:
        settings['kernel_config']=options['kernel_config']
    if options.get('nccl_only'):
        settings['disable_custom_all_reduce']=True
    if model_manifest['model_spec'].get('quantization'):
        settings['quantization']=model_manifest['model_spec']['quantization']
    identity = {'model':model_manifest,'settings':settings,'runtime_environment':runtime_environment,'worker_sha256':file_hash(__file__),
                'tokenizer_helper_sha256':file_hash(Path(__file__).with_name('bam_training.py')),
                'versions':{p:version(p) for p in ('vllm','torch','transformers','tokenizers')},
                'sampling': {k: options.get(k, default) for k, default in
                    [('temperature', 0.), ('top_p', 1.), ('top_k', -1), ('min_p', 0.)]},
                'completion_policy': {'expects_reasoning': options.get('expects_reasoning', False),
                    'reasoning_token_reserve': options.get('reasoning_token_reserve', 0),
                    'complete_markdown_fence_removal': True}}
    execution_hash = value_hash(identity)
    identity_path = root/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        raise ValueError('Generation runtime changed; use a new run_id')
    write_json(identity_path,identity)
    prepared = prepare_requests(requests,tokenizer,settings['max_model_len'],
                                reasoning_token_reserve=options.get('reasoning_token_reserve', 0))
    responses, pending = {}, []
    for item in prepared:
        path = root/'responses'/(item['request_sha256']+'.json')
        if path.exists():
            cached=json.loads(path.read_text())
            if cached['request_sha256']!=item['request_sha256'] or cached['execution_sha256']!=execution_hash:
                raise ValueError('Cached request/runtime mismatch')
            if cached.get('response_sha256')!=value_hash({k:v for k,v in cached.items() if k!='response_sha256'}):
                raise ValueError('Cached response content changed')
            responses[item['request_id']]=cached
            continue
        if item.get('status')=='error':
            result={k:v for k,v in item.items() if k not in ('request','input_ids')}
            result.update(execution_sha256=execution_hash,text='',finish_reason=None,output_tokens=0)
            result['response_sha256']=value_hash(result)
            write_json(path,result)
            responses[item['request_id']]=result
        else:
            pending.append(item)
    if commit:
        commit()
    llm=None
    chunk_size=int(options.get('request_chunk_size',32))
    if chunk_size<1:
        raise ValueError('Positive request_chunk_size required')
    for start in range(0,len(pending),chunk_size):
        batch=pending[start:start+chunk_size]
        if llm is None:
            llm=LLM(model=str(model_path),tokenizer=str(model_path),**settings)
        sampling=[]
        for item in batch:
            kwargs={}
            if item['request'].get('json_schema') is not None:
                # Structured outputs are optional and fail loudly on an incompatible runtime.
                from vllm.sampling_params import StructuredOutputsParams
                kwargs['structured_outputs']=StructuredOutputsParams(json=item['request']['json_schema'])
            sampling.append(SamplingParams(**identity['sampling'],
                            max_tokens=item['max_tokens'],seed=config.get('seed',230),**kwargs))
        began=time.monotonic()
        try:
            outputs=llm.generate([{'prompt_token_ids':i['input_ids']} for i in batch],sampling,use_tqdm=False)
            if len(outputs)!=len(batch):
                raise ValueError('vLLM returned incomplete batch')
            results=[]
            for item,output in zip(batch,outputs):
                completion=output.outputs[0]
                results.append({**separate_completion(completion.text, completion.finish_reason,
                    expects_reasoning=options.get('expects_reasoning', False)),
                    'finish_reason':completion.finish_reason,'output_tokens':len(completion.token_ids)})
        except Exception as error:
            # Never discard prior completed batches. Errors require a new request/run to retry.
            results=[{'text':'','finish_reason':None,'output_tokens':0,'status':'error',
                      'error':f'{type(error).__name__}: {error}'} for _ in batch]
        elapsed=time.monotonic()-began
        for item,result in zip(batch,results):
            result.update({k:v for k,v in item.items() if k not in ('request','input_ids')})
            result.update(execution_sha256=execution_hash,batch_seconds=elapsed,batch_size=len(batch))
            result['response_sha256']=value_hash(result)
            write_json(root/'responses'/(item['request_sha256']+'.json'),result)
            responses[item['request_id']]=result
        if commit:
            commit()
        print(json.dumps({'completed':len(responses),'total':len(requests),'batch_seconds':elapsed}),flush=True)
    ordered=[responses[row.get('request_id',row.get('id'))] for row in requests]
    write_rows(root/'responses.jsonl',ordered)
    summary={'requests':len(requests),'complete':sum(r['status']=='complete' for r in ordered),
             'errors':sum(r['status']!='complete' for r in ordered),'execution_sha256':execution_hash,
             'responses_sha256':file_hash(root/'responses.jsonl')}
    write_json(root/'summary.json',summary)
    if commit:
        commit()
    return {'summary':summary,'identity':identity,'responses':ordered}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args()
    request=json.loads(args.request.read_text())
    run_requests(args.root,request['requests'],request['model_manifest'],request['config'])


if __name__=='__main__':
    main()
