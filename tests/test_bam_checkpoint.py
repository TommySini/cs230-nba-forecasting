import copy
import contextlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import bam_checkpoint as checkpoint


def identity():
    return {'model': {'model_id': 'fixture/model', 'model_revision': 'a' * 40},
            **{key: 'b' * 64 for key in ('config_sha256', 'rows_sha256', 'tokenized_sha256',
                                       'worker_sha256', 'tokenizer_sha256')},
            'row_count': 5, 'gradient_accumulation_steps': 2,
            'parameter_names': ['0.weight', '0.bias']}


def cursor(step=1, offset=2):
    return {'epoch': 0, 'next_group_start': offset, 'global_step': step,
            'order': [4, 2, 0, 1, 3], 'epoch_loss_sum': .5, 'epoch_seen': offset, 'history': []}


def setup():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(.3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
    return model, optimizer, random.Random(230)


def update(model, optimizer, indexes):
    optimizer.zero_grad(set_to_none=True)
    for i in indexes:
        value = torch.tensor([[i / 5, .2, -.3]], dtype=torch.float32)
        loss = model(value).square().mean() / len(indexes)
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)


class Checkpoints(unittest.TestCase):
    def save(self, root, model, optimizer, rng, saved_cursor=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return checkpoint.save_checkpoint(root, identity=identity(), cursor=saved_cursor or cursor(),
                adapter_state=model.state_dict(), optimizer=optimizer, named_parameters=model.named_parameters(), shuffle_rng=rng)

    def restore(self, root, model, optimizer, rng, expected=None):
        return checkpoint.restore_checkpoint(root, identity=expected or identity(), adapter_state=model.state_dict(),
            optimizer=optimizer, named_parameters=model.named_parameters(), shuffle_rng=rng,
            apply_adapter=lambda state: model.load_state_dict(state, strict=True))

    def test_exact_resume_matches_dropout_optimizer_and_next_epoch_shuffle(self):
        with tempfile.TemporaryDirectory() as root:
            torch.manual_seed(123); random.seed(123)
            model, optimizer, rng = setup()
            update(model, optimizer, [4, 2])
            self.save(root, model, optimizer, rng)
            # Advance through the remaining partial group and another epoch.
            def finish(m, opt, shuffle):
                update(m, opt, [0, 1]); update(m, opt, [3])
                order = list(range(5)); shuffle.shuffle(order)
                for offset in range(0, len(order), 2):
                    update(m, opt, order[offset:offset + 2])
                return order, random.random(), torch.rand(4)
            order, python_value, torch_value = finish(model, optimizer, rng)
            expected = copy.deepcopy(model.state_dict())
            expected_optimizer = copy.deepcopy(optimizer.state_dict())
            torch.manual_seed(777); random.seed(777)
            resumed, resumed_optimizer, resumed_rng = setup()
            result = self.restore(root, resumed, resumed_optimizer, resumed_rng)
            self.assertEqual(result['cursor'], cursor())
            resumed_order, resumed_python, resumed_torch = finish(resumed, resumed_optimizer, resumed_rng)
            self.assertEqual(order, resumed_order)
            self.assertEqual(python_value, resumed_python)
            self.assertTrue(torch.equal(torch_value, resumed_torch))
            for name, tensor in expected.items():
                self.assertTrue(torch.equal(tensor, resumed.state_dict()[name]), name)
            for key, state in expected_optimizer['state'].items():
                for field, value in state.items():
                    other = resumed_optimizer.state_dict()['state'][key][field]
                    self.assertTrue(torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other)

    def test_changed_identity_fails_before_model_mutation(self):
        with tempfile.TemporaryDirectory() as root:
            model, optimizer, rng = setup(); update(model, optimizer, [4, 2])
            self.save(root, model, optimizer, rng)
            for field in ('rows_sha256', 'config_sha256', 'tokenized_sha256', 'worker_sha256', 'tokenizer_sha256'):
                changed = identity(); changed[field] = 'c' * 64
                untouched = copy.deepcopy(model.state_dict())
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    self.restore(root, model, optimizer, rng, changed)
                self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in untouched.items()))
            changed = identity(); changed['model']['model_revision'] = 'd' * 40
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                self.restore(root, model, optimizer, rng, changed)

    def test_incomplete_write_does_not_advance_pointer_or_emit_success(self):
        with tempfile.TemporaryDirectory() as root:
            model, optimizer, rng = setup(); update(model, optimizer, [4, 2])
            self.save(root, model, optimizer, rng)
            before = (Path(root) / 'latest.json').read_bytes()
            update(model, optimizer, [0, 1])
            log = io.StringIO()
            with patch.object(torch, 'save', side_effect=OSError('simulated interrupted write')):
                with contextlib.redirect_stdout(log), self.assertRaises(OSError):
                    checkpoint.save_checkpoint(root, identity=identity(), cursor=cursor(2, 4),
                        adapter_state=model.state_dict(), optimizer=optimizer, named_parameters=model.named_parameters(), shuffle_rng=rng)
            self.assertEqual((Path(root) / 'latest.json').read_bytes(), before)
            self.assertNotIn('checkpoint_complete', log.getvalue())
            self.assertFalse(list(Path(root).glob('.pending-*')))

    def test_corrupted_file_or_pointer_fails_closed(self):
        for name in ('adapter.pt', 'optimizer.pt', 'rng.pt', 'cursor.json', 'manifest.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as root:
                model, optimizer, rng = setup(); update(model, optimizer, [4, 2])
                pointer = self.save(root, model, optimizer, rng)
                path = Path(root) / pointer['checkpoint'] / name
                path.write_bytes(path.read_bytes() + b'corrupt')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    self.restore(root, model, optimizer, rng)

    def test_optimizer_groups_and_accumulation_boundary_checked(self):
        with tempfile.TemporaryDirectory() as root:
            model, optimizer, rng = setup()
            model(torch.ones(1, 3)).sum().backward()
            with self.assertRaisesRegex(ValueError, 'zero_grad'):
                self.save(root, model, optimizer, rng)
            optimizer.zero_grad(set_to_none=True)
            with self.assertRaisesRegex(ValueError, 'boundary'):
                self.save(root, model, optimizer, rng, cursor(1, 1))
            update(model, optimizer, [4, 2]); self.save(root, model, optimizer, rng)
            optimizer.param_groups[0]['lr'] = .02
            with self.assertRaisesRegex(ValueError, 'hyperparameters'):
                self.restore(root, model, optimizer, rng)

    def test_crash_before_pointer_swap_preserves_old_state_and_can_replay(self):
        with tempfile.TemporaryDirectory() as root:
            model, optimizer, rng = setup(); update(model, optimizer, [4, 2])
            first = self.save(root, model, optimizer, rng)
            old_weights = copy.deepcopy(model.state_dict())
            update(model, optimizer, [0, 1])
            original_json = checkpoint._json
            def interrupted_json(path, value):
                if Path(path).name.startswith('.latest-'):
                    raise OSError('crash before pointer commit')
                return original_json(path, value)
            with patch.object(checkpoint, '_json', side_effect=interrupted_json):
                with self.assertRaises(OSError):
                    self.save(root, model, optimizer, rng, cursor(2, 4))
            self.assertEqual(json.loads((Path(root) / 'latest.json').read_text()), first)
            self.assertEqual(len(list(Path(root).glob('step_*'))), 2)
            self.restore(root, model, optimizer, rng)
            self.assertTrue(all(torch.equal(tensor, model.state_dict()[name]) for name, tensor in old_weights.items()))
            update(model, optimizer, [0, 1])
            second = self.save(root, model, optimizer, rng, cursor(2, 4))
            self.assertEqual(second['global_step'], 2)
            self.assertEqual(len(list(Path(root).glob('step_*'))), 3)

    def test_unpointed_directories_ignored_and_progress_must_increase(self):
        with tempfile.TemporaryDirectory() as root:
            model, optimizer, rng = setup()
            (Path(root) / '.pending-interrupted').mkdir()
            self.assertIsNone(self.restore(root, model, optimizer, rng))
            update(model, optimizer, [4, 2]); pointer = self.save(root, model, optimizer, rng)
            with self.assertRaisesRegex(ValueError, 'increase'):
                self.save(root, model, optimizer, rng)
            # Simulate a complete orphan by removing only its commit pointer.
            (Path(root) / 'latest.json').unlink()
            self.assertIsNone(self.restore(root, model, optimizer, rng))
            replacement = self.save(root, model, optimizer, rng)
            self.assertNotEqual(pointer['checkpoint'], replacement['checkpoint'])
            self.assertTrue((Path(root) / pointer['checkpoint']).is_dir())

    def test_bad_row_order_and_unsafe_pointer_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            model, optimizer, rng = setup()
            bad = cursor(); bad['order'] = [4, 2, 0, 0, 3]
            with self.assertRaisesRegex(ValueError, 'permutation'):
                self.save(root, model, optimizer, rng, bad)
            (Path(root) / 'latest.json').write_text(json.dumps({'checkpoint': '../outside'}))
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                self.restore(root, model, optimizer, rng)

    def test_periodic_checkpoint_due_at_three_minutes(self):
        self.assertFalse(checkpoint.checkpoint_due(100, now=279.9))
        self.assertTrue(checkpoint.checkpoint_due(100, now=280))
        with self.assertRaisesRegex(ValueError, 'five minutes'):
            checkpoint.checkpoint_due(100, interval_seconds=360)

    def test_same_shape_reordered_optimizer_cannot_swap_moments(self):
        class Pair(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(torch.tensor([1., 2.]))
                self.b = torch.nn.Parameter(torch.tensor([3., 4.]))
        with tempfile.TemporaryDirectory() as root:
            model = Pair(); optimizer = torch.optim.AdamW([model.a, model.b], lr=.01)
            model.a.grad = torch.full_like(model.a, 2.)
            model.b.grad = torch.full_like(model.b, 12.)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            expected = identity(); expected['parameter_names'] = ['a', 'b']
            with contextlib.redirect_stdout(io.StringIO()):
                checkpoint.save_checkpoint(root, identity=expected, cursor=cursor(), adapter_state=model.state_dict(),
                    optimizer=optimizer, named_parameters=model.named_parameters(), shuffle_rng=random.Random(230))
            other = Pair(); reversed_optimizer = torch.optim.AdamW([other.b, other.a], lr=.01)
            self.assertEqual(checkpoint.optimizer_parameter_names(reversed_optimizer, other.named_parameters()), ['b', 'a'])
            applied = []
            with self.assertRaisesRegex(ValueError, 'Actual optimizer named-parameter order'):
                checkpoint.restore_checkpoint(root, identity=expected, adapter_state=other.state_dict(),
                    optimizer=reversed_optimizer, named_parameters=other.named_parameters(), shuffle_rng=random.Random(230),
                    apply_adapter=lambda state: applied.append(state))
            self.assertEqual(applied, [])
            self.assertEqual(reversed_optimizer.state_dict()['state'], {})
            with self.assertRaisesRegex(ValueError, 'Actual optimizer named-parameter order'):
                checkpoint.save_checkpoint(Path(root) / 'wrong-order', identity=expected, cursor=cursor(),
                    adapter_state=other.state_dict(), optimizer=reversed_optimizer,
                    named_parameters=other.named_parameters(), shuffle_rng=random.Random(230))

    def test_optimizer_parameter_must_be_actual_named_object(self):
        model, optimizer, _ = setup()
        unrelated, _, _ = setup()
        with self.assertRaisesRegex(ValueError, 'not bound'):
            checkpoint.optimizer_parameter_names(optimizer, unrelated.named_parameters())

    def test_cpu_cuda_runtime_transitions_rejected_before_mutation(self):
        cpu = checkpoint._runtime_signature()
        gpu = {**cpu, 'cuda_available': True, 'cuda_device_count': 1, 'cuda_device_names': ['fixture GPU']}
        for saved_runtime, restored_runtime in ((cpu, gpu), (gpu, cpu)):
            with self.subTest(saved=saved_runtime), tempfile.TemporaryDirectory() as root:
                model, optimizer, rng = setup(); update(model, optimizer, [4, 2])
                with patch.object(checkpoint, '_runtime_signature', return_value=saved_runtime):
                    with patch.object(torch.cuda, 'is_available', return_value=saved_runtime['cuda_available']):
                        with patch.object(torch.cuda, 'get_rng_state_all', return_value=[torch.get_rng_state()] if saved_runtime['cuda_available'] else []):
                            self.save(root, model, optimizer, rng)
                untouched = copy.deepcopy(model.state_dict())
                with patch.object(checkpoint, '_runtime_signature', return_value=restored_runtime):
                    with self.assertRaisesRegex(ValueError, 'CPU/CUDA runtime'):
                        self.restore(root, model, optimizer, rng)
                self.assertTrue(all(torch.equal(tensor, model.state_dict()[name]) for name, tensor in untouched.items()))


if __name__ == '__main__':
    unittest.main()
