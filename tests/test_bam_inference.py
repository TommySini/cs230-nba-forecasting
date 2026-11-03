"""Real failure-mode regressions for final-answer transport and token budgets."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import bam_inference as worker


class CompletionBoundary(unittest.TestCase):
    def test_complete_fence_only_is_removed_and_raw_preserved(self):
        raw = '```json\n{"verdict":"unsupported"}\n```'
        result = worker.separate_completion(raw, 'stop')
        self.assertEqual(result['text'], '{"verdict":"unsupported"}')
        self.assertEqual(result['raw_text'], raw)
        self.assertTrue(result['removed_markdown_fence'])
        prefixed = 'Here is JSON:\n' + raw
        self.assertEqual(worker.separate_completion(prefixed, 'stop')['text'], prefixed)

    def test_reasoning_never_enters_final_answer(self):
        raw = 'Source comparison.\n</think>\n```json\n{"verdict":"unclear"}\n```'
        result = worker.separate_completion(raw, 'stop', expects_reasoning=True)
        self.assertEqual(result['text'], '{"verdict":"unclear"}')
        self.assertEqual(result['reasoning_text'], 'Source comparison.')
        self.assertEqual(result['raw_text'], raw)
        opened = worker.separate_completion('<think>x</think>{"ok":true}', 'stop', expects_reasoning=True)
        self.assertEqual(opened['reasoning_text'], 'x')

    def test_truncation_missing_or_duplicate_boundary_is_never_accepted(self):
        for raw, reason in [('<think>{"verdict":"supported"}', 'stop'),
                            ('x</think>{"verdict":"supported"}', 'length'),
                            ('x</think>y</think>{}', 'stop'), ('x</think>', 'stop')]:
            result = worker.separate_completion(raw, reason, expects_reasoning=True)
            self.assertEqual(result['status'], 'error')
            self.assertEqual(result['text'], '')
            self.assertEqual(result['raw_text'], raw)
        self.assertEqual(worker.separate_completion('x</think>{}', 'stop')['status'], 'error')

    def test_reserve_counts_against_complete_context(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return [1] * 20
        request = {'request_id': 'r', 'messages': [{'role': 'user', 'content': 'Evidence'}], 'max_tokens': 10}
        fail = worker.prepare_requests([request], Tokenizer(), 30, reasoning_token_reserve=5)[0]
        self.assertEqual(fail['status'], 'error')
        self.assertEqual(fail['prompt_tokens'], 20)
        good = worker.prepare_requests([request], Tokenizer(), 35, reasoning_token_reserve=5)[0]
        self.assertEqual(good['max_tokens'], 15)
        self.assertEqual(good['requested_answer_tokens'], 10)

    def test_overrides_do_not_change_student_sampling(self):
        config = {'generation': {'temperature': 0}, 'generation_by_model': {
            'teacher': {'temperature': .6, 'expects_reasoning': True, 'reasoning_token_reserve': 4096}}}
        self.assertEqual(worker.generation_options(config, {'model_id': 'student'})['temperature'], 0)
        self.assertEqual(worker.generation_options(config, {'model_id': 'teacher'})['temperature'], .6)
        for reserve in (-1, 8193, True):
            with self.assertRaises(ValueError):
                worker.generation_options({'generation': {'reasoning_token_reserve': reserve}}, {'model_id': 'x'})


if __name__ == '__main__':
    unittest.main()
