"""Monitoring must accept resource metrics already present in training rows."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import wr_common


class MonitorContracts(unittest.TestCase):
    def test_training_resource_overlap_and_completion(self):
        with tempfile.TemporaryDirectory() as d, patch.object(wr_common,'ROOT',Path(d)), patch.object(wr_common,'resources',return_value={'process_rss_hwm_bytes':123,'cuda_peak_allocated_bytes':456}):
            with wr_common.Monitor('train') as monitor:
                monitor.update(completed=1000,total=1000,process_rss_hwm_bytes=12,cuda_peak_allocated_bytes=45)
                monitor.write()
            value=json.loads((Path(d)/'reports/PROGRESS/train.json').read_text())
            self.assertEqual(value['status'],'COMPLETE')
            self.assertEqual(value['completed'],1000)
            self.assertEqual(value['process_rss_hwm_bytes'],123)
            self.assertEqual(value['cuda_peak_allocated_bytes'],456)
            self.assertEqual(len(list((Path(d)/'reports/COSTS').glob('*.json'))),1)

    def test_original_training_exception_is_retained(self):
        with tempfile.TemporaryDirectory() as d, patch.object(wr_common,'ROOT',Path(d)):
            with self.assertRaisesRegex(ValueError,'original training error'):
                with wr_common.Monitor('train') as monitor:
                    monitor.update(**wr_common.resources())
                    raise ValueError('original training error')
            cost=json.loads(next((Path(d)/'reports/COSTS').glob('*.json')).read_text())
            self.assertEqual(cost['status'],'FAILED')
            self.assertIn('original training error',cost['traceback'])
