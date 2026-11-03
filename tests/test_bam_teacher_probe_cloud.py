"""Remote module hydration must not import unmounted local deployment modules."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import modal


class TeacherProbePackagingTests(unittest.TestCase):
    def test_remote_import_without_bam_cloud(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/bam_teacher_probe_cloud.py'
        spec = importlib.util.spec_from_file_location('isolated_teacher_hydration_test', path)
        module = importlib.util.module_from_spec(spec)
        with patch.object(modal, 'is_local', return_value=False), patch.dict(sys.modules, {'bam_cloud': None}):
            spec.loader.exec_module(module)
        self.assertEqual(module.PINNED_MODEL['model_revision'], 'f07f63f2bbd7540917118ebdf3812696ef303b03')
        self.assertTrue(hasattr(module, 'compare'))


if __name__ == '__main__':
    unittest.main()
