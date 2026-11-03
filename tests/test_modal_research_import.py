"""Cloud provenance must survive response tampering or omitted runtime labels."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import import_modal_research as subject


class CloudProvenanceTests(unittest.TestCase):
    def test_changed_response_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/'response.json'
            path.write_text('{"text":"original"}')
            registry={'response_files':{'response.json':subject.r.sha(path)}}
            path.write_text('{"text":"altered"}')
            with patch.object(subject.r,'ROOT',root),self.assertRaisesRegex(ValueError,'changed'):
                subject.validate_execution({},'response.json',registry)

    def test_removed_backend_identity_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/'response.json'
            path.write_text('{}')
            registry={'response_files':{'response.json':subject.r.sha(path)}}
            with patch.object(subject.r,'ROOT',root),self.assertRaisesRegex(ValueError,'Missing cloud'):
                subject.validate_execution({},'response.json',registry)

    def test_different_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            identity={'conversion':{'source_files':{'model':'changed'}}}
            digest=subject.r.value_hash(identity)
            subject.r.write_json(root/'executions'/(digest+'.json'),identity)
            response={'execution_sha256':digest}
            subject.r.write_json(root/'response.json',response)
            registry={'response_files':{'response.json':subject.r.sha(root/'response.json')},'executions':{digest:identity}}
            (root/'config.json').write_text('{}')
            with patch.object(subject.r,'ROOT',root),patch.object(subject.r,'DATA',root),patch.object(subject.r,'CONFIG',root/'config.json'),patch.object(subject.r,'verify_protocol',return_value={'model_sha256':{'model':'original'}}),self.assertRaisesRegex(ValueError,'different original checkpoint'):
                subject.validate_execution(response,'response.json',registry)


if __name__=='__main__':
    unittest.main()
