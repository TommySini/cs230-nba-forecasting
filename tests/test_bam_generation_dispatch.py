import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from bam_generation_dispatch import dispatch_batches, DispatchFailed, ExactResponseCache, value_hash
from run_bam_pipeline import DurableCloud


class DispatchTests(unittest.TestCase):
    def test_bounded_parallelism_order_and_original_identity(self):
        barrier = threading.Barrier(4); lock = threading.Lock(); seen = []
        jobs = [{'args': [i], 'identity': {'batch': i}} for i in range(8)]
        def invoke(name, args, identity):
            self.assertEqual(name, 'generate'); self.assertEqual(identity, {'batch': args[0]})
            with lock: seen.append(args[0])
            barrier.wait(timeout=5)
            return args[0] * 2
        self.assertEqual(dispatch_batches(jobs, invoke), list(range(0, 16, 2)))
        self.assertEqual(sorted(seen), list(range(8)))

    def test_failure_drains_started_calls_and_does_not_submit_rest(self):
        barrier = threading.Barrier(4); seen = []; lock = threading.Lock()
        def invoke(name, args, identity):
            with lock: seen.append(args[0])
            barrier.wait(timeout=5)
            if args[0] == 0: raise RuntimeError('retained failure')
            return args[0]
        # Hold successful initial calls until the failed future has reached the
        # coordinator, without relying on timing: use a one-worker failure case
        # for deterministic no-new-submission and four workers for bounded drain.
        with self.assertRaises(DispatchFailed) as failed:
            dispatch_batches([{'args': [i], 'identity': {'i': i}} for i in range(4)], invoke)
        self.assertEqual(sorted(seen), [0, 1, 2, 3])
        self.assertEqual(failed.exception.results[1:], [1, 2, 3])
        calls = []
        def fail(name, args, identity):
            calls.append(args); raise RuntimeError('stop')
        with self.assertRaises(DispatchFailed) as failed:
            dispatch_batches([{'args': [i], 'identity': {'i': i}} for i in range(8)], fail, max_inflight=1)
        self.assertEqual(len(calls), 1); self.assertEqual(failed.exception.not_started, list(range(1, 8)))

    def test_reconnect_uses_existing_durable_call_id_without_duplicate_compute(self):
        class Call:
            object_id = 'fc-existing'; reads = 0
            def get(self, timeout):
                self.reads += 1
                if self.reads == 1: raise RuntimeError('controller disconnected')
                return {'value': 42}
        call = Call()
        class Function:
            spawns = 0
            def spawn(self, *args): self.spawns += 1; return call
        function = Function(); reconnects = []
        def load(identity): reconnects.append(identity); return call
        with tempfile.TemporaryDirectory() as tmp:
            cloud = DurableCloud(tmp, functions={'generate': function}, call_loader=load)
            jobs = [{'args': ['same request'], 'identity': {'batch': 1}}]
            with self.assertRaises(DispatchFailed): dispatch_batches(jobs, cloud.invoke)
            self.assertEqual(dispatch_batches(jobs, cloud.invoke), [{'value': 42}])
            self.assertEqual(dispatch_batches(jobs, cloud.invoke), [{'value': 42}])
        self.assertEqual(function.spawns, 1); self.assertEqual(reconnects, ['fc-existing'])

    def test_invalid_limit_and_duplicate_identity_rejected_before_invocation(self):
        def invoke(*args): self.fail('must not invoke')
        for limit in (0, 5, True):
            with self.assertRaises(ValueError): dispatch_batches([], invoke, max_inflight=limit)
        with self.assertRaises(ValueError):
            dispatch_batches([{'args': [], 'identity': {}}] * 2, invoke)


class ExactCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.cache = ExactResponseCache(self.root / 'cache')
        self.identity = {'model': {'model_spec': {'model_id': 'pinned', 'model_revision': 'a' * 40},
                         'files': {'weights': 'b' * 64, 'tokenizer.json': 'c' * 64}},
                         'settings': {'seed': 230}, 'sampling': {'temperature': 0},
                         'completion_policy': {'expects_reasoning': False}, 'versions': {'vllm': '0.32'},
                         'worker_sha256': 'd' * 64, 'tokenizer_helper_sha256': 'e' * 64}
        self.request = {'request_id': 'request1', 'source_hash': 'f' * 64, 'max_tokens': 4096,
                        'messages': [{'role': 'user', 'content': 'identical pregame input'}]}
        self.prompt_hash = '1' * 64
        self.response = {'request_id': 'request1', 'request_sha256': value_hash(self.request),
                         'execution_sha256': value_hash(self.identity), 'prompt_token_ids_sha256': self.prompt_hash,
                         'status': 'complete', 'finish_reason': 'stop', 'text': '{"findings":[]}', 'raw_text': '{"findings":[]}'}
        self.response['response_sha256'] = value_hash(self.response)
        self.save_origin()

    def save_origin(self):
        result = {'identity': self.identity, 'responses': [self.response]}
        raw = json.dumps(result).encode(); (self.root / 'result.json').write_bytes(raw)
        (self.root / 'manifest.json').write_text(json.dumps({'sha256': hashlib.sha256(raw).hexdigest()}))

    def store(self):
        return self.cache.import_batch([self.request], self.root / 'result.json', self.root / 'manifest.json',
                     origin={'run_id': 'previous', 'call_id': 'fc-original'}, expected_identity=self.identity,
                     prompt_token_hashes={'request1': self.prompt_hash})

    def lookup(self, request=None, identity=None, prompt=None):
        return self.cache.lookup(request or self.request, expected_identity=identity or self.identity,
                                 prompt_token_sha256=prompt or self.prompt_hash)

    def test_exact_reuse_preserves_response_and_origin_without_approval(self):
        self.assertEqual(self.store(), 1)
        reused = self.lookup()
        self.assertEqual(reused['response'], self.response)
        self.assertEqual(reused['reuse']['call_id'], 'fc-original')
        self.assertFalse(reused['reuse']['semantic_approval_reused'])

    def test_changed_source_metadata_prompt_budget_runtime_or_tokenizer_is_miss(self):
        self.store()
        for key, value in [('source_hash', 'changed'), ('max_tokens', 8192), ('parent_hashes', ['new'])]:
            request = {**self.request, key: value}; self.assertIsNone(self.lookup(request=request))
        for key, value in [('versions', {'vllm': 'changed'}), ('settings', {'seed': 99}),
                           ('sampling', {'temperature': 1}), ('worker_sha256', 'changed')]:
            self.assertIsNone(self.lookup(identity={**self.identity, key: value}))
        identity = copy.deepcopy(self.identity); identity['model']['files']['tokenizer.json'] = 'changed'
        self.assertIsNone(self.lookup(identity=identity)); self.assertIsNone(self.lookup(prompt='changed'))

    def test_tampered_origin_manifest_response_and_cache_rejected(self):
        (self.root / 'result.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'manifest'): self.store()
        self.response['text'] = 'tampered'; self.save_origin()
        with self.assertRaisesRegex(ValueError, 'identity mismatch'): self.store()
        self.response['response_sha256'] = value_hash({k: v for k, v in self.response.items() if k != 'response_sha256'})
        self.save_origin(); self.store()
        batch = next((self.root / 'cache/batches').glob('*.json')); data = json.loads(batch.read_text())
        data['origin']['call_id'] = 'changed'; batch.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'origin evidence'): self.lookup()

    def test_truncated_output_never_reused_and_untrusted_runtime_rejected(self):
        self.response.update(status='error', finish_reason='length')
        self.response['response_sha256'] = value_hash({k: v for k, v in self.response.items() if k != 'response_sha256'})
        self.save_origin(); self.assertEqual(self.store(), 0); self.assertIsNone(self.lookup())
        self.identity.pop('worker_sha256')
        with self.assertRaisesRegex(ValueError, 'Incomplete'): self.lookup()


if __name__ == '__main__': unittest.main()
