"""One synthetic BF16 LoRA benchmark; no NBA training or production mutation."""
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import argparse
import json
import math
from pathlib import Path
import time

import bam_training as training
from bam_student_vllm import StudentVLLM, adapter_identity

CHECKPOINT = 'step_000000000006_399d05befda5410daf90d2572926ad1e'
MANIFEST_SHA = 'd1e8b3ac3d79d755d121731fa2c2eda3f33e703b02afe026425febff168799a1'
ADAPTER_SHA = 'e304892f42062377a66ad8c2f55d25ebb8b84f1e34767524f5c63c33a518625f'
CONFIG_SHA = '37d865844e5f04636422f8a861d82032222a61e4af6171d307c8490928b911a0'


def trained_synthetic_adapter(root, config):
    import torch
    torch.set_num_threads(1)
    from safetensors.torch import save_file
    source = Path('/runs/bam/training/bam_resume_smoke_v4_20261102/synthetic_resume_smoke/research/full/checkpoints') / CHECKPOINT
    if training.file_hash(source / 'manifest.json') != MANIFEST_SHA or training.file_hash(source / 'adapter.pt') != ADAPTER_SHA:
        raise ValueError('Synthetic trained checkpoint differs from verified CUDA replay')
    config_path = Path('/runs/bam/probes/bam_student_decode_probe_20261102/synthetic_adapter/adapter_config.json')
    if training.file_hash(config_path) != CONFIG_SHA: raise ValueError('Synthetic adapter architecture changed')
    tensors = torch.load(source / 'adapter.pt', map_location='cpu', weights_only=True)
    if not isinstance(tensors, dict) or not any('lora_B' in k and torch.count_nonzero(v).item() for k,v in tensors.items()):
        raise ValueError('Expected an actually updated synthetic adapter')
    target = root / 'synthetic_trained_adapter'; target.mkdir()
    (target / 'adapter_config.json').write_bytes(config_path.read_bytes())
    save_file({k:v.contiguous() for k,v in tensors.items()}, str(target / 'adapter_model.safetensors'))
    digest = training.value_hash({p.name:training.file_hash(p) for p in sorted(target.iterdir())})
    adapter_identity(target, digest, config)
    return target, digest, {'source_checkpoint': str(source), 'manifest_sha256': MANIFEST_SHA,
                           'adapter_tensor_sha256': ADAPTER_SHA, 'optimizer_updates': 6,
                           'actual_nba_sft': False, 'synthetic_only': True}


def prompt(tokenizer, approximate_tokens, variant):
    # Construct new synthetic evidence of a chosen length; never truncate a real source.
    sentence = f'Synthetic line {variant}: Example Player is available; the source does not establish any injury or outcome. '
    pattern = tokenizer.encode(sentence, add_special_tokens=False)
    count = max(1, (approximate_tokens-120)//len(pattern))
    for _ in range(4):
        messages = [{'role':'system','content':'Use only the synthetic source. No NBA game or outcome exists.'},
                    {'role':'user','content':sentence * count +
                     '\nWrite a source-only research note of about 400 words with facts, interpretation and uncertainties. Do not invent statistics.'}]
        ids = training.chat_token_ids(tokenizer, messages, generation_prompt=True)
        if abs(len(ids)-approximate_tokens) < len(pattern): break
        count = max(1, count + round((approximate_tokens-len(ids))/len(pattern)))
    return {'input_ids':ids, 'request_sha256':training.value_hash(messages), 'messages':messages}


def adapter_effect(backend, item):
    from vllm import SamplingParams
    records = []
    for enabled in (False, True):
        raw = backend.llm.generate([{'prompt_token_ids':item['input_ids']}],
                  SamplingParams(temperature=0., max_tokens=1, logprobs=20, seed=230),
                  lora_request=backend.lora if enabled else None, use_tqdm=False)[0]
        if list(raw.prompt_token_ids) != item['input_ids']: raise ValueError('Control prompt changed')
        observed = getattr(getattr(raw, 'lora_request', None), 'lora_int_id', None)
        if observed != (backend.adapter_id if enabled else None): raise ValueError('Control adapter identity differs')
        values = {str(k):float(v.logprob) for k,v in raw.outputs[0].logprobs[0].items()}
        records.append({'adapter':enabled,'logprobs':values,'tokens':list(raw.outputs[0].token_ids)})
    common = sorted(records[0]['logprobs'].keys() & records[1]['logprobs'].keys())
    deltas = {k:records[1]['logprobs'][k]-records[0]['logprobs'][k] for k in common}
    if not deltas or not all(math.isfinite(v) for v in deltas.values()): raise ValueError('No comparable finite logits')
    maximum = max(abs(v) for v in deltas.values())
    return {'paired_results':records,'common_token_logprob_deltas':deltas,'max_abs_logprob_delta':maximum,
            'adapter_effect_verified':maximum > 1e-6, 'prompt_token_ids_sha256':training.value_hash(item['input_ids']),
            'one_token_diagnostic_only':True}


def run(root, config):
    import torch
    from vllm import SamplingParams
    root = Path(root); began = time.monotonic()
    adapter, sha, origin = trained_synthetic_adapter(root, config)
    key = config['model_id'].replace('/','--') + '--' + config['model_revision']
    manifest = json.loads((Path('/models/bam/manifests')/(key+'.json')).read_text())
    backend = StudentVLLM(config, manifest, adapter_path=adapter, expected_adapter_hash=sha)
    result = {'synthetic_only':True,'actual_nba_sft':False,'adapter_origin':origin,'identity':backend.identity,
              'runtime_fix':{'multiprocess_method':'spawn','pre_engine_cpu_tensor_threads':1},
              'load_seconds':backend.load_seconds,'cases':[], 'gpu':torch.cuda.get_device_name(),
              'request_count_limit':18,'comparison':'Existing NF4 throughput only; no paired NF4 text equivalence'}
    training.write_json(root/'result.json',result)
    # Two outputs establish adapter use; no natural-completion claim for the one-token diagnostic.
    control = prompt(backend.tokenizer,256,99)
    result['adapter_effect'] = adapter_effect(backend,control)
    training.write_json(root/'result.json',result)
    print(json.dumps({'batch_complete':'adapter_effect','verified':result['adapter_effect']['adapter_effect_verified']}),flush=True)
    # Twelve natural completions; persist at most four outstanding responses.
    for index, length in enumerate((4096,8192,8192)):
        items = [prompt(backend.tokenizer,length,index*4+i) for i in range(4)]
        training.write_json(root/f'inputs_{index}.json',items)
        started = time.monotonic(); records = backend.generate(items,1536); elapsed=time.monotonic()-started
        for i,record in enumerate(records): training.write_json(root/f'output_{index}_{i}.json',record)
        case = {'kind':'natural_eos','batch_size':4,'prompt_tokens_each':[len(x['input_ids']) for x in items],
                'seconds':elapsed,'output_tokens':sum(r['output_tokens'] for r in records),
                'complete':sum(r['status']=='complete' for r in records), 'records':records}
        case['aggregate_output_tokens_per_second']=case['output_tokens']/elapsed
        result['cases'].append(case); training.write_json(root/'result.json',result)
        print(json.dumps({'batch_complete':index,'seconds':elapsed,'complete':case['complete']}),flush=True)
    # Four outputs match the earlier NF4 fixed256 benchmark's length constraint.
    items = [prompt(backend.tokenizer,8192,50+i) for i in range(4)]
    training.write_json(root/'inputs_fixed.json',items)
    tick=time.monotonic()
    outputs=backend.llm.generate([{'prompt_token_ids':x['input_ids']} for x in items],
        SamplingParams(temperature=0.,max_tokens=256,min_tokens=256,ignore_eos=True,seed=230),
        lora_request=backend.lora,use_tqdm=False)
    elapsed=time.monotonic()-tick
    if len(outputs)!=4: raise ValueError('Fixed batch dropped requests')
    for item,out in zip(items,outputs):
        if list(out.prompt_token_ids)!=item['input_ids'] or len(out.outputs[0].token_ids)!=256:
            raise ValueError('Fixed benchmark token contract failed')
        if getattr(getattr(out,'lora_request',None),'lora_int_id',None)!=backend.adapter_id:
            raise ValueError('Fixed benchmark adapter omitted')
    fixed={'kind':'fixed256_benchmark_only','natural_completion_claim':False,'batch_size':4,'seconds':elapsed,
           'prompt_tokens_each':[len(x['input_ids']) for x in items], 'aggregate_output_tokens_per_second':1024/elapsed,
           'output_token_ids':[list(o.outputs[0].token_ids) for o in outputs]}
    result['cases'].append(fixed); result['elapsed_seconds']=time.monotonic()-began
    result['total_requests']=18; result['status']='complete'
    training.write_json(root/'result.json',result)
    print(json.dumps({'batch_complete':'probe_finished','seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--config',required=True)
    a=p.parse_args();run(a.root,json.loads(Path(a.config).read_text()))
