"""Tiny synthetic tests: no forecasting data, downloaded models or cloud calls."""
import contextlib
import copy
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import bam_train_only as worker


class Tokenizer:
    chat_template = 'synthetic'
    special_tokens_map = {'eos_token': 'EOS'}
    def get_vocab(self):
        return {str(i): i for i in range(40)}
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_dict=False):
        result = [1]
        for message in messages:
            result += [{'system': 2, 'user': 3, 'assistant': 4}[message['role']]]
            result += [5 + ord(char) % 30 for char in message['content']]
            result += [0]
        if add_generation_prompt:
            result += [4]
        return result


class TinyAdapterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('embedding', torch.sin(torch.arange(40 * 6).reshape(40, 6).float()))
        self.register_buffer('projection', torch.cos(torch.arange(6 * 40).reshape(6, 40).float()))
        self.lora_A = torch.nn.Parameter(torch.randn(6, 4) * .15)
        self.lora_B = torch.nn.Parameter(torch.randn(4, 6) * .15)
        self.dropout = torch.nn.Dropout(.1)
        self.calls = []
    @property
    def base_model(self):
        return self
    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False, return_dict=True):
        self.calls.append({'input_ids': input_ids.detach().tolist(), 'labels': None if labels is None else labels.detach().tolist()})
        embeddings = self.embedding[input_ids]
        hidden = embeddings.cumsum(dim=1) / torch.arange(1, embeddings.shape[1] + 1).reshape(1, -1, 1)
        hidden = hidden + self.dropout(hidden) @ self.lora_A @ self.lora_B
        if labels is None:
            return SimpleNamespace(last_hidden_state=hidden)
        logits = hidden @ self.projection
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 40), labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss)


def fixture():
    rows = [{'game_id': str(i), 'split': 'train', 'p_boost': .2 + .2 * i,
             'messages': [{'role': 'system', 'content': 'Pregame'}, {'role': 'user', 'content': 'Team ' + str(i)}]}
            for i in range(3)]
    labels = {'0': 1, '1': 0, '2': 1}
    gold = {str(i): {'game_id': str(i), 'split': 'train', 'review_status': 'accepted', 'explanation': 'Supported ' + str(i)} for i in range(3)}
    return rows, labels, gold


def config(**overrides):
    return {'model_id': worker.MODEL_ID, 'model_revision': worker.MODEL_REVISION, 'minimum_eval_date': '2024-09-25',
            'seed': 230, 'forecast': {'lora_rank': 8}, 'train_only': {
                'max_optimizer_steps': 4, 'max_runtime_seconds': 300, 'max_seq_length': 256,
                'evaluate_every_steps': 1, 'learning_rate': .025, 'head_learning_rate': .1,
                'explanation_weight': .3, 'gradient_clip': 1., 'minimum_relative_brier_reduction': .5,
                'minimum_relative_ce_reduction': .1, **overrides}}


def make_model(seed=42):
    torch.manual_seed(seed); random.seed(seed)
    model = TinyAdapterModel()
    head = torch.nn.Linear(6, 1)
    torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
    return model, head


def admission_fixture(root, digest='b' * 64, region='us-west-2'):
    root = Path(root)
    binding = {'job_id': 'pilot-test', 'config_sha256': worker.common.value_hash(config()),
               'model': {key: config()[key] for key in ('model_id', 'model_revision')},
               'manifest_sha256': digest, 'region': region}
    report = {'schema_version': 1, 'checked_at': datetime.now(timezone.utc).isoformat(), 'binding': binding,
              'prerequisites_satisfied': True, 'blockers': [], 'reservation_created': False, 'launch_authorized': False,
              'financials': {'cost_bounds_complete': True, 'bounded_reservation_cents': 500,
                             'verified_credit_available_cents': 9500, 'ledger_available_cents': 18000}}
    report_path = root / 'preflight.json'; report_path.write_text(json.dumps(report))
    receipt = {'platform': 'aws_ec2', 'status': 'admitted', 'reservation_id': 'synthetic',
               **binding, 'max_runtime_seconds': 300, 'external_termination_armed': True, 'reserved_cents': 500,
               'preflight_report_sha256': worker.common.file_hash(report_path),
               'expires_at_utc': (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()}
    path = root / 'admission.json'; path.write_text(json.dumps(receipt))
    return path, report_path, receipt


class TrainOnlyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(patch.object(worker.common, '_adapter_state', side_effect=lambda m: {n: p for n, p in m.named_parameters()}))
        self.stack.enter_context(patch.object(worker.common, '_apply_adapter', side_effect=lambda m, state: m.load_state_dict(state, strict=False)))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.addCleanup(self.stack.close)
        self.rows, self.labels, self.gold = fixture()
        self.tokenizer = Tokenizer()

    def run_core(self, directory, model, head, **kwargs):
        return worker.train_steps(directory, config(), self.rows, self.labels, self.gold, model, self.tokenizer, head,
                                  manifest_sha256='b' * 64, **kwargs)

    def test_zero_head_and_both_gradient_paths_are_measured(self):
        with tempfile.TemporaryDirectory() as directory:
            model, head = make_model()
            result = self.run_core(directory, model, head)
            curve = json.loads((Path(directory) / 'learning_curve_train.json').read_text())
            self.assertTrue(result['baseline']['zero_head_equivalence_verified'])
            self.assertEqual(curve['steps'][0]['optimization']['brier_adapter_gradient_norm'], 0.)
            self.assertGreater(curve['steps'][0]['optimization']['joint_adapter_gradient_norm'], 0.)
            self.assertTrue(result['head_and_adapters_updated'])
            self.assertTrue(result['brier_gradient_reached_adapters_after_zero_head'])
            self.assertLess(result['last_inference_train']['brier'], result['baseline']['brier'])
            self.assertFalse(result['generalization_claim'])

    def test_probability_forwards_never_receive_teacher_forced_tokens(self):
        model, head = make_model()
        tokens = [worker.common.prompt_tokens(self.tokenizer, row['messages'], 256) for row in self.rows]
        encoded = {i: worker.common.completion_tokens(self.tokenizer, row['messages'] + [
            {'role': 'assistant', 'content': self.gold[row['game_id']]['explanation']}], 256) for i, row in enumerate(self.rows)}
        optimizer = torch.optim.SGD(list(model.parameters()) + list(head.parameters()), lr=.1)
        worker.joint_update(model, head, optimizer, self.rows, self.labels, tokens, encoded, [0, 1, 2],
                            weight=.3, gradient_clip=1., deadline=time.monotonic() + 30)
        self.assertEqual(len(model.calls), 6)
        for i in range(3):
            self.assertEqual(model.calls[i]['input_ids'], [tokens[i]])
            self.assertIsNone(model.calls[i]['labels'])
            self.assertTrue(all(label == -100 for label in model.calls[i + 3]['labels'][0][:len(tokens[i])]))
            self.assertEqual(model.calls[i + 3]['input_ids'][0][:len(tokens[i])], tokens[i])

    def test_sparse_gold_gradient_matches_separately_normalized_objective(self):
        def prepare():
            model, head = make_model()
            model.dropout.p = 0.
            head.weight.data.fill_(.1)
            return model, head
        model, head = prepare()
        tokens = [worker.common.prompt_tokens(self.tokenizer, row['messages'], 256) for row in self.rows]
        encoded = {1: worker.common.completion_tokens(self.tokenizer, self.rows[1]['messages'] + [
            {'role': 'assistant', 'content': self.gold['1']['explanation']}], 256)}
        optimizer = torch.optim.SGD(list(model.parameters()) + list(head.parameters()), lr=.1)
        record = worker.joint_update(model, head, optimizer, self.rows, self.labels, tokens, encoded, [0, 1, 2],
                                    weight=.3, gradient_clip=1e6, deadline=time.monotonic() + 30)
        expected, expected_head = prepare()
        other_optimizer = torch.optim.SGD(list(expected.parameters()) + list(expected_head.parameters()), lr=.1)
        brier = sum((worker._probability(expected, expected_head, tokens[i], row['p_boost']) - self.labels[row['game_id']]).square()
                    for i, row in enumerate(self.rows)) / 3
        ce = worker._completion_loss(expected, encoded[1], expected_head.weight.device)
        self.assertAlmostEqual(record['gold_completion_ce'], ce.item())
        (brier + .3 * ce).backward(); other_optimizer.step()
        for actual, wanted in zip(list(model.parameters()) + list(head.parameters()), list(expected.parameters()) + list(expected_head.parameters())):
            self.assertTrue(torch.allclose(actual, wanted, atol=1e-7, rtol=1e-6))

    def test_resume_preserves_adapter_head_optimizer_dropout_and_shuffle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, head = make_model()
            complete = self.run_core(root / 'complete', model, head)
            state = copy.deepcopy(worker.common._joint_state(model, head))
            model, head = make_model()
            with self.assertRaisesRegex(RuntimeError, 'checkpointed'):
                self.run_core(root / 'resume', model, head, stop_after_steps=2)
            model, head = make_model(88)
            resumed = self.run_core(root / 'resume', model, head)
            self.assertEqual(complete, resumed)
            for name, value in state.items():
                self.assertTrue(torch.equal(value, worker.common._joint_state(model, head)[name]), name)
            self.assertEqual(json.loads((root / 'complete/learning_curve_train.json').read_text()),
                             json.loads((root / 'resume/learning_curve_train.json').read_text()))

    def test_changed_labels_reject_resume_before_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            model, head = make_model()
            with self.assertRaises(RuntimeError):
                self.run_core(directory, model, head, stop_after_steps=1)
            model, head = make_model(99)
            before = copy.deepcopy(worker.common._joint_state(model, head))
            self.labels['1'] = 1
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                self.run_core(directory, model, head)
            for name, value in before.items():
                self.assertTrue(torch.equal(value, worker.common._joint_state(model, head)[name]))

    def test_rejects_heldout_input_misalignment_and_unreviewed_gold(self):
        for corrupt in ('split', 'labels', 'gold', 'postmortem'):
            with self.subTest(corrupt=corrupt):
                rows, labels, gold = fixture()
                if corrupt == 'split': rows[0]['split'] = 'dev'
                elif corrupt == 'labels': labels['outsider'] = 1
                elif corrupt == 'gold': gold['0']['review_status'] = 'pending'
                else: rows[0]['postmortem'] = 'Outcome'
                with self.assertRaises(ValueError): worker._validate_core_inputs(rows, labels, gold)

    def test_rejects_trainable_base_weights_nonzero_head_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            model, head = make_model()
            model.base_weight = torch.nn.Parameter(torch.ones(1))
            with self.assertRaisesRegex(ValueError, 'Only LoRA'):
                self.run_core(directory, model, head)
            model, head = make_model(); head.bias.data.fill_(.01)
            with self.assertRaisesRegex(ValueError, 'exactly zero'):
                self.run_core(directory, model, head)
            self.rows[0]['messages'][1]['content'] *= 100
            model, head = make_model()
            with self.assertRaisesRegex(ValueError, 'never silently truncate'):
                self.run_core(directory, model, head)

    def test_expired_runtime_discards_partial_gradients_before_step(self):
        model, head = make_model()
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=.1)
        before = copy.deepcopy(worker.common._joint_state(model, head))
        tokens = [worker.common.prompt_tokens(self.tokenizer, row['messages'], 256) for row in self.rows]
        with self.assertRaises(worker.RuntimeLimit):
            worker.joint_update(model, head, optimizer, self.rows, self.labels, tokens, {}, [0, 1, 2],
                                weight=.3, gradient_clip=1., deadline=0.)
        self.assertEqual(optimizer.state_dict()['state'], {})
        for name, value in before.items(): self.assertTrue(torch.equal(value, worker.common._joint_state(model, head)[name]))
        self.assertTrue(all(p.grad is None for p in list(model.parameters()) + list(head.parameters())))

    def test_nonfinite_gradient_fails_without_optimizer_update(self):
        model, head = make_model()
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=.1)
        handle = head.weight.register_hook(lambda grad: grad * float('nan'))
        tokens = [worker.common.prompt_tokens(self.tokenizer, row['messages'], 256) for row in self.rows]
        with self.assertRaisesRegex(ValueError, 'Nonfinite TRAIN gradient'):
            worker.joint_update(model, head, optimizer, self.rows, self.labels, tokens, {}, [0, 1, 2],
                                weight=.3, gradient_clip=1., deadline=time.monotonic() + 30)
        handle.remove()
        self.assertEqual(optimizer.state_dict()['state'], {})

    def test_admission_rejects_unfunded_expired_or_wrong_manifest_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path, preflight, receipt = admission_fixture(directory)
            self.assertEqual(worker._read_execution_admission(path, preflight, 'b' * 64, config(), 'pilot-test'), receipt)
            for change in ({'reserved_cents': 10100}, {'reserved_cents': 200}, {'external_termination_armed': False},
                           {'manifest_sha256': 'wrong'}, {'job_id': 'other'}, {'config_sha256': 'wrong'}, {'model': {}},
                           {'preflight_report_sha256': 'wrong'},
                           {'expires_at_utc': '2020-01-01T00:00:00+00:00'}):
                path.write_text(json.dumps(receipt | change))
                with self.assertRaises(ValueError): worker._read_execution_admission(path, preflight, 'b' * 64, config(), 'pilot-test')

    def test_admission_region_is_bound_to_preflight_and_requires_commercial_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path, preflight, receipt = admission_fixture(directory, region='us-east-1')
            self.assertEqual(worker._read_execution_admission(path, preflight, 'b' * 64, config(), 'pilot-test'), receipt)
            # A valid region cannot be substituted after the plan was admitted.
            path.write_text(json.dumps(receipt | {'region': 'us-west-2'}))
            with self.assertRaisesRegex(ValueError, 'matching complete preflight'):
                worker._read_execution_admission(path, preflight, 'b' * 64, config(), 'pilot-test')
            # Old reports without a region binding must be regenerated.
            report = json.loads(preflight.read_text())
            del report['binding']['region']
            preflight.write_text(json.dumps(report))
            path.write_text(json.dumps(receipt | {'preflight_report_sha256': worker.common.file_hash(preflight)}))
            with self.assertRaisesRegex(ValueError, 'matching complete preflight'):
                worker._read_execution_admission(path, preflight, 'b' * 64, config(), 'pilot-test')
            for region in (None, '', [], 'us-gov-west-1', 'cn-north-1'):
                path, preflight, receipt = admission_fixture(directory, region=region)
                with self.assertRaisesRegex(ValueError, 'matching admitted'):
                    worker._read_execution_admission(path, preflight, 'b' * 64, config(), 'pilot-test')

    def test_requires_bounded_config_and_fixed_positive_joint_weight(self):
        for update in ({'max_optimizer_steps': 0}, {'explanation_weight': 0.}, {'max_runtime_seconds': None},
                       {'minimum_relative_brier_reduction': 2}, {'learning_rate': float('nan')}):
            with self.assertRaises(ValueError): worker.validate_config(config(**update))
        for key, value in (('model_id', 'other/model'), ('model_revision', 'f' * 40)):
            changed = config(); changed[key] = value
            with self.assertRaisesRegex(ValueError, 'approved pinned'):
                worker.validate_config(changed)

    def test_passing_preflight_does_not_replace_independent_launcher_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path, preflight, receipt = admission_fixture(directory)
            path.write_text(preflight.read_text())
            with self.assertRaisesRegex(ValueError, 'matching admitted'):
                worker._read_execution_admission(path, preflight, 'b' * 64, config(), 'pilot-test')
            path.write_text(json.dumps(receipt))
            report = json.loads(preflight.read_text())
            for change in ({'prerequisites_satisfied': False}, {'blockers': ['credit eligibility unknown']},
                           {'checked_at': '2020-01-01T00:00:00Z'}, {'binding': {}}):
                preflight.write_text(json.dumps(report | change))
                path.write_text(json.dumps(receipt | {'preflight_report_sha256': worker.common.file_hash(preflight)}))
                with self.assertRaises(ValueError):
                    worker._read_execution_admission(path, preflight, 'b' * 64, config(), 'pilot-test')

    def test_learning_stops_after_meeting_both_pinned_objectives(self):
        chosen = config(max_optimizer_steps=40, minimum_relative_brier_reduction=.01,
                        minimum_relative_ce_reduction=.0001)
        with tempfile.TemporaryDirectory() as directory:
            model, head = make_model()
            result = worker.train_steps(directory, chosen, self.rows, self.labels, self.gold,
                model, self.tokenizer, head, manifest_sha256='b' * 64)
            self.assertEqual(result['status'], 'learning_criterion_met')
            self.assertGreater(result['optimizer_steps'], 1)
            self.assertLess(result['optimizer_steps'], chosen['train_only']['max_optimizer_steps'])

    def test_validated_sixteen_game_bundle_reaches_model_loader_only_after_admission(self):
        from test_bam_train_contract import fixture_bundle, write_fixture_bundle
        values = fixture_bundle()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = write_fixture_bundle(root, values)
            path, preflight, receipt = admission_fixture(root, digest)
            with patch.object(worker.common, '_load_model', side_effect=RuntimeError('synthetic model loader boundary')) as load:
                with self.assertRaisesRegex(RuntimeError, 'synthetic model loader boundary'):
                    worker.train_bundle(root, digest, config(), root / 'output', path, preflight, 'pilot-test')
                load.assert_called_once()
            artifact = root / 'inputs_train.jsonl'
            artifact.write_text(artifact.read_text().replace('"split": "train"', '"split": "dev"', 1))
            with patch.object(worker.common, '_load_model') as load:
                with self.assertRaisesRegex(ValueError, 'artifact changed'):
                    worker.train_bundle(root, digest, config(), root / 'output', path, preflight, 'pilot-test')
                load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
