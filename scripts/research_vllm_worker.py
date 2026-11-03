"""Isolated CUDA execution of the frozen, label-free research requests."""
import argparse
from collections import defaultdict
from importlib.metadata import version
import json
from pathlib import Path
import time

import run_game_note_research as r


def validation_feedback(error, text, task, stage):
    """Tell a retry which required decisions were omitted; never invent them."""
    if stage != 'verify':
        return str(error)
    expected = [str(i) for i in range(len(task['facts']))]
    try:
        decisions = r.parse_json(text).get('decisions', [])
        returned = [d.get('id') for d in decisions if isinstance(d, dict)]
    except (ValueError, TypeError, AttributeError):
        returned = []
    missing = [i for i in expected if i not in returned]
    return (str(error) + '. Return exactly ' + str(len(expected)) + ' decisions, one for EACH ID '
            + json.dumps(expected) + '. Previous returned IDs: ' + json.dumps(returned)
            + '. Missing IDs: ' + json.dumps(missing) + '. Do not omit the final claim. '
            'Every claim needs its own supported, unsupported, or unclear verdict based on the supplied source.')


def execute(tasks, stage, config, llm, tokenizer, execution_sha):
    from vllm import SamplingParams
    message = r.extraction_messages if stage == 'extract' else r.verification_messages
    validate = r.validate_extraction if stage == 'extract' else r.validate_verification
    budget = config['extraction_max_tokens'] if stage == 'extract' else config['verification_max_tokens']
    pending = []
    for task in tasks:
        request_hash = r.value_hash(message(task))
        path = r.DATA / stage / (task['id'] + '.json')
        if path.exists():
            old = r.read_json(path)
            if old['request_sha256'] != request_hash or old['protocol_sha256'] not in r.response_protocols():
                raise ValueError('Cached response provenance changed')
            if old.get('status') == 'complete' and old.get('finish_reason') == 'stop':
                if validate(r.parse_json(old['text']), task) != old['parsed']:
                    raise ValueError('Cached parsing changed')
                continue
        pending.append((task, request_hash, 0, None))
    started = time.monotonic()
    print(f'{stage}: {len(tasks)-len(pending)}/{len(tasks)} cached; {len(pending)} queued', flush=True)
    eos = tokenizer.eos_token_id
    stop_ids = [eos] if isinstance(eos, int) else list(eos)
    while pending:
        prompts, sampling = [], []
        for task, request_hash, attempt, error in pending:
            messages = message(task)
            if error:
                messages += [{'role': 'user', 'content': 'The prior attempt failed validation: ' + error + '. Return a complete corrected JSON object. Cite only printed source line numbers; use only allowed category/time values and supplied decision IDs.'}]
            ids = r.encode_messages(tokenizer, messages)
            if len(ids) > config['max_prompt_tokens']:
                raise ValueError('Prompt exceeds frozen budget; no truncation permitted')
            prompts.append({'prompt_token_ids': ids})
            sampling.append(SamplingParams(temperature=0, seed=config['seed'],
                max_tokens=min(budget * 2**attempt, 8192), stop_token_ids=stop_ids))
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        if len(outputs) != len(pending):
            raise ValueError('Missing generation outputs')
        retries = []
        for item, prompt, output in zip(pending, prompts, outputs):
            task, request_hash, attempt, _ = item
            completion = output.outputs[0]
            # Protocol identifies the scientific requests; execution identifies
            # actual weights representation, versions, kernels and batch settings.
            result = {'id': task['id'], 'protocol_sha256': r.sha(r.REPORT / 'protocol.json'),
                'execution_sha256': execution_sha, 'request_sha256': request_hash,
                'text': completion.text, 'attempt': attempt, 'finish_reason': completion.finish_reason,
                'prompt_tokens': len(prompt['prompt_token_ids']), 'output_tokens': len(completion.token_ids),
                'prompt_token_ids_sha256': r.value_hash(prompt['prompt_token_ids'])}
            try:
                if completion.finish_reason != 'stop':
                    raise ValueError('Generation reached its output limit')
                result.update(parsed=validate(r.parse_json(completion.text), task), status='complete')
            except (ValueError, TypeError, KeyError) as error:
                result.update(status='failed', error=str(error))
                if attempt + 1 < config['max_attempts']:
                    retries.append((task, request_hash, attempt + 1,
                                    validation_feedback(error, completion.text, task, stage)))
            r.write_json(r.DATA / stage / (task['id']+'.json'), result)
            # Keep failed attempts from earlier executions when resuming.
            r.write_json(r.DATA / 'attempts' / stage / (task['id']+f'-{execution_sha[:12]}-a{attempt}.json'), result)
        pending = retries
    failed = [t['id'] for t in tasks if r.read_json(r.DATA/stage/(t['id']+'.json'))['status'] != 'complete']
    if failed:
        raise ValueError(f'Incomplete {stage}; no pages skipped: {failed[:5]}')
    print(f'{stage}: completed {len(tasks)} requests in {time.monotonic()-started:.1f}s', flush=True)


def main(model_path, benchmark):
    from transformers import AutoTokenizer
    from vllm import LLM
    config = r.read_json(r.CONFIG)
    conversion = r.read_json(Path(model_path)/'conversion.json')
    for name, digest in conversion['output_files'].items():
        if r.sha(Path(model_path)/name) != digest:
            raise ValueError('Converted checkpoint checksum changed: ' + name)
    tokenizer = AutoTokenizer.from_pretrained(r.ROOT / config['model_path'], local_files_only=True)
    units = r.read_rows(r.ROOT / 'probe/units.jsonl' if benchmark else r.DATA / 'units.jsonl')
    # Prove that the remote tokenizer produces the exact prepared extraction IDs.
    token_reference = r.read_json(r.ROOT / 'token_reference.json')
    for unit in units:
        ids = r.encode_messages(tokenizer, r.extraction_messages(unit))
        if r.value_hash(ids) != token_reference[unit['id']]:
            raise ValueError('Tokenizer changed input tokens: ' + unit['id'])
    settings = dict(dtype='float16', max_model_len=19456, max_num_seqs=48,
                    max_num_batched_tokens=8192, gpu_memory_utilization=.88,
                    seed=config['seed'], generation_config='vllm', enforce_eager=True,
                    enable_prefix_caching=True, enable_chunked_prefill=True)
    identity = {'backend': 'vllm', 'versions': {p: version(p) for p in ('vllm','torch','transformers','numpy','tokenizers')},
                'conversion': conversion, 'settings': settings, 'worker_sha256': r.sha(__file__),
                'original_research_protocol_sha256': r.sha(r.REPORT/'protocol.json'),
                'scope': 'Frozen checkpoint dequantized to FP16 matching original scales. Different arithmetic/backend; not bitwise-identical MLX generation. No target labels or boosting inputs.'}
    execution_sha = r.value_hash(identity)
    r.write_json(r.DATA/'executions'/(execution_sha+'.json'), identity)
    llm = LLM(model=str(model_path), tokenizer=str(r.ROOT/config['model_path']), **settings)
    if benchmark:
        start = time.monotonic()
        execute(units, 'extract', config, llm, tokenizer, execution_sha)
        extraction_seconds = time.monotonic()-start
        tasks = r.verification_tasks(units, config)
        start = time.monotonic()
        execute(tasks, 'verify', config, llm, tokenizer, execution_sha)
        verification_seconds = time.monotonic()-start
        from check_research_quality import challenge_cases
        tasks, expected = challenge_cases(units)
        original_data = r.DATA
        r.DATA = original_data / 'challenges'
        execute(tasks, 'verify', config, llm, tokenizer, execution_sha)
        cases = []
        for task in tasks:
            decision = r.read_json(r.DATA/'verify'/(task['id']+'.json'))['parsed']['decisions'][0]
            cases.append({**decision, 'expected_supported': expected[decision['id']],
                'passed': (decision['verdict']=='supported') == expected[decision['id']]})
        r.DATA = original_data
        result = {'execution_sha256': execution_sha, 'units':len(units), 'extraction_seconds':extraction_seconds,
                  'verification_seconds':verification_seconds, 'factual_challenges':cases,
                  'semantic_checker_all_passed':all(c['passed'] for c in cases), 'schema_and_source_checks':'passed'}
        r.write_json(r.DATA/'benchmark_result.json', result)
        print(json.dumps(result, indent=2), flush=True)
    else:
        games = defaultdict(list)
        for unit in units:
            games[unit['game']['game_id']].append(unit)
        progress = []
        for gid, game_units in games.items():
            start = time.monotonic()
            execute(game_units, 'extract', config, llm, tokenizer, execution_sha)
            tasks = r.verification_tasks(game_units, config)
            execute(tasks, 'verify', config, llm, tokenizer, execution_sha)
            progress.append({'game_id':gid, 'units':len(game_units), 'seconds':time.monotonic()-start})
            r.write_json(r.REPORT/'progress.json', {'completed_games':len(progress),'total_games':len(games),'games':progress})
            print(json.dumps({'completed_games':len(progress),'total_games':len(games),**progress[-1]}),flush=True)
        r.merge_reports(units, r.verification_tasks(units, config))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    main(args.model_path, args.benchmark)
