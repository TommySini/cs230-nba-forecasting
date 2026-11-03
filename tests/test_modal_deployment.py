"""Job submission must reuse deployed functions and avoid accidental compute."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import modal_research_vllm as subject


class ModalDeploymentTests(unittest.TestCase):
    def test_status_resolves_named_functions_without_creating_app_or_invoking_jobs(self):
        convert = Mock(object_id='fu-convert')
        execute = Mock(object_id='fu-execute')
        output = io.StringIO()
        with patch.object(subject.modal.Function, 'from_name', side_effect=[convert, execute]) as lookup, \
             patch.object(subject.app, 'run', side_effect=AssertionError('Ephemeral app started')), \
             patch.object(subject.app, 'deploy', side_effect=AssertionError('Unexpected deployment')), \
             redirect_stdout(output):
            subject.main(environment='main', check_deployment=True)
        self.assertEqual(lookup.call_args_list, [
            call(subject.APP_NAME, 'convert_model', environment_name='main'),
            call(subject.APP_NAME, 'execute', environment_name='main'),
        ])
        for function in (convert, execute):
            function.hydrate.assert_called_once_with()
            function.remote.assert_not_called()
        result = json.loads(output.getvalue())
        self.assertFalse(result['compute_started'])
        self.assertEqual(result['functions'], {'convert_model':'fu-convert','execute':'fu-execute'})

    def test_missing_deployment_fails_without_fallback_to_new_app(self):
        missing = Mock()
        missing.hydrate.side_effect = subject.modal.exception.NotFoundError('Not deployed')
        with patch.object(subject.modal.Function, 'from_name', return_value=missing), \
             patch.object(subject.app, 'run', side_effect=AssertionError('Ephemeral fallback')), \
             self.assertRaisesRegex(RuntimeError, 'Deploy .* first'):
            subject.deployed_functions()
        missing.remote.assert_not_called()

    def test_status_cannot_also_launch_a_job(self):
        with patch.object(subject, 'deployed_functions') as lookup, \
             self.assertRaisesRegex(ValueError, 'cannot be combined'):
            subject.main(snapshot='snapshot.tar.gz', check_deployment=True)
        lookup.assert_not_called()

    def test_stale_deployed_worker_is_rejected_before_gpu_submission(self):
        names = ('research_vllm_worker.py', 'convert_research_weights.py')
        with patch.object(subject, 'digest', side_effect=lambda path:path.name):
            subject.verify_deployed_sources({'deployed_source_sha256':{name:name for name in names}})
            with self.assertRaisesRegex(ValueError, 'redeploy before starting GPU'):
                subject.verify_deployed_sources({'deployed_source_sha256':{name:'old' for name in names}})
            with self.assertRaisesRegex(ValueError, 'redeploy before starting GPU'):
                subject.verify_deployed_sources({})

    def test_expanded_retry_budget_requires_inspected_resume(self):
        with patch.object(subject, 'deployed_functions') as lookup:
            for budget in (0, 4, 5):
                with self.assertRaisesRegex(ValueError, 'fourth invocation'):
                    subject.main(invocation_budget=budget)
            lookup.assert_not_called()


if __name__ == '__main__':
    unittest.main()
