"""Bounded generation dispatch and exact raw-response reuse. No Modal imports.

``invoke`` is the driver's DurableCloud.invoke: it must persist/reconnect call IDs
and hash-check completed results. This helper never spawns behind that boundary.
One controller owns a run; separate concurrent controllers are unsupported.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import tempfile


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


class DispatchFailed(RuntimeError):
    def __init__(self, results, failures, not_started):
        self.results, self.failures, self.not_started = results, failures, not_started
        super().__init__(f'Generation dispatch failed at batches {sorted(failures)}; remaining calls drained and retained')


def dispatch_batches(jobs, invoke, *, max_inflight=4, function_name='generate'):
    """Return results in input order; preserve each job's original args/identity.

    Each job is {args: [...], identity: {...}} for the selected generation function. At most
    four invoke calls run concurrently. After an exception, submit no new work,
    drain existing calls so their durable results survive, then raise with indexed
    results/failures/not_started. Restart with the SAME jobs through DurableCloud.
    Initialize the caller's cloud connection before using worker threads.
    """
    if type(max_inflight) is not int or not 1 <= max_inflight <= 4:
        raise ValueError('Generation concurrency must be within 1..4')
    if function_name not in ('generate','generate_heavy'):
        raise ValueError('Unknown generation function')
    jobs = list(jobs)
    keys = [value_hash(job['identity']) for job in jobs]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate durable batch identity')
    results, failures, following = [None] * len(jobs), {}, 0
    with ThreadPoolExecutor(max_workers=max_inflight) as pool:
        active = {}
        while following < len(jobs) or active:
            while not failures and following < len(jobs) and len(active) < max_inflight:
                job = jobs[following]
                active[pool.submit(invoke, function_name, job['args'], job['identity'])] = following
                following += 1
            if not active:
                break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                index = active.pop(future)
                try:
                    results[index] = future.result()
                except Exception as error:
                    failures[index] = error
    if failures:
        raise DispatchFailed(results, failures, list(range(following, len(jobs))))
    return results


def _immutable(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Immutable cache entry changed: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, ensure_ascii=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _runtime(identity):
    """Require complete pinned model/file/tokenizer/runtime evidence, not a name."""
    for key in ('model', 'settings', 'sampling', 'completion_policy', 'versions',
                'worker_sha256', 'tokenizer_helper_sha256'):
        if not identity.get(key): raise ValueError('Incomplete execution identity: ' + key)
    if not identity['model'].get('files') or not identity['model'].get('model_spec'):
        raise ValueError('Pinned model files and revision required')


def _validate(request, response, identity, prompt_hash):
    _runtime(identity)
    if not request.get('source_hash'):
        raise ValueError('Exact reuse requires the request source_hash')
    if (response.get('request_id') != request.get('request_id')
            or response.get('request_sha256') != value_hash(request)
            or response.get('execution_sha256') != value_hash(identity)
            or response.get('prompt_token_ids_sha256') != prompt_hash
            or response.get('response_sha256') != value_hash({k: v for k, v in response.items() if k != 'response_sha256'})):
        raise ValueError('Exact request/source/token/runtime/response identity mismatch')


class ExactResponseCache:
    """Explicit import of hash-verified prior call files; never scans arbitrary runs.

    ``expected_identity`` is the COMPLETE trusted execution identity for the new
    run, including model file hashes, worker/helper hashes, versions and effective
    sampling/settings. Do not simply copy it from an unverified prior response.
    Request metadata/source hashes must match too. Changed audit metadata is a miss.
    Only naturally completed raw responses are reusable; semantic approval is not.
    """
    def __init__(self, directory):
        self.directory = Path(directory)

    def import_batch(self, requests, result_path, manifest_path, *, origin,
                     expected_identity, prompt_token_hashes):
        raw = Path(result_path).read_bytes(); manifest_raw = Path(manifest_path).read_bytes()
        manifest = json.loads(manifest_raw)
        if manifest.get('sha256') != hashlib.sha256(raw).hexdigest():
            raise ValueError('Prior cloud result manifest checksum mismatch')
        result = json.loads(raw)
        _runtime(expected_identity)
        if result.get('identity') != expected_identity:
            raise ValueError('Prior and current execution identities differ')
        if not origin.get('run_id') or not origin.get('call_id'):
            raise ValueError('Origin run and durable call IDs required')
        by_id = {r['request_id']: r for r in requests}
        responses = result['responses']
        if (len(by_id) != len(requests) or len(responses) != len(requests)
                or {r['request_id'] for r in responses} != set(by_id)):
            raise ValueError('Prior result/request coverage differs')
        for response in responses:
            request = by_id[response['request_id']]
            _validate(request, response, expected_identity, prompt_token_hashes[request['request_id']])
        evidence = {'origin': origin, 'result_path': str(result_path), 'manifest_path': str(manifest_path),
                    'origin_result_sha256': manifest['sha256'],
                    'origin_manifest_sha256': hashlib.sha256(manifest_raw).hexdigest(),
                    'origin_manifest': manifest, 'result': result, 'requests': requests}
        evidence_hash = value_hash(evidence)
        _immutable(self.directory / 'batches' / (evidence_hash + '.json'), evidence)
        count = 0
        for index, response in enumerate(responses):
            if response.get('status') != 'complete' or response.get('finish_reason') != 'stop': continue
            request = by_id[response['request_id']]
            key = self._key(request, expected_identity, prompt_token_hashes[request['request_id']])
            pointer = self.directory / 'entries' / (key + '.json')
            # Keep the first exact completion; later repeats never replace it.
            if not pointer.exists(): _immutable(pointer, {'batch_sha256': evidence_hash, 'response_index': index})
            count += 1
        return count

    @staticmethod
    def _key(request, identity, prompt_hash):
        return value_hash({'request': request, 'identity': identity, 'prompt_token_sha256': prompt_hash})

    def lookup(self, request, *, expected_identity, prompt_token_sha256):
        _runtime(expected_identity)
        path = self.directory / 'entries' / (self._key(request, expected_identity, prompt_token_sha256) + '.json')
        if not path.exists(): return None
        pointer = json.loads(path.read_text()); batch_hash = pointer['batch_sha256']
        if not isinstance(batch_hash, str) or len(batch_hash) != 64 or any(c not in '0123456789abcdef' for c in batch_hash):
            raise ValueError('Invalid cache evidence hash')
        evidence = json.loads((self.directory / 'batches' / (batch_hash + '.json')).read_text())
        if value_hash(evidence) != batch_hash: raise ValueError('Cached origin evidence changed')
        result = evidence['result']
        if result['identity'] != expected_identity: raise ValueError('Cached execution identity changed')
        response = result['responses'][pointer['response_index']]
        _validate(request, response, expected_identity, prompt_token_sha256)
        if response.get('status') != 'complete' or response.get('finish_reason') != 'stop':
            raise ValueError('Incomplete response cannot be reused')
        return {'response': response, 'reuse': {**evidence['origin'],
                'origin_result_sha256': evidence['origin_result_sha256'],
                'origin_manifest_sha256': evidence['origin_manifest_sha256'],
                'cached_evidence_sha256': batch_hash, 'request_sha256': value_hash(request),
                'execution_sha256': value_hash(expected_identity), 'semantic_approval_reused': False}}
