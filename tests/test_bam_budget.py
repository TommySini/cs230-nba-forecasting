from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from bam_budget import BudgetExceeded, BudgetLedger


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'budget.json'

    def reserve(self, ledger, key='batch', rate=4.743648, seconds=7200):
        return ledger.reserve(key, identity={'request_hash': key},
                              timeout_seconds=seconds, hourly_rate_usd=rate)

    def test_explicit_cap_required_and_invalid_numbers_fail_before_ledger_creation(self):
        with self.assertRaises(TypeError): BudgetLedger(self.path)
        for cap in (None, '150', True, float('nan'), float('inf'), -1, 0, 5):
            with self.assertRaises(ValueError): BudgetLedger(self.path, cap_usd=cap)
        self.assertFalse(self.path.exists())

    def test_timeout_margin_buffer_conservative_rounding_and_reconnect(self):
        ledger = BudgetLedger(self.path, cap_usd=20)
        record = self.reserve(ledger)
        self.assertEqual(record['reservation_cents'], 1092)
        ledger.mark_spawn_intent('batch'); ledger.attach_call('batch', 'fc-original')
        ledger = BudgetLedger(self.path, cap_usd=20)
        self.reserve(ledger); ledger.attach_call('batch', 'fc-original')
        totals = ledger.snapshot()['totals']
        self.assertEqual(totals['outstanding_reserved_cents'], 1092)
        self.assertEqual(totals['available_for_new_calls_cents'], 408)
        with self.assertRaises(BudgetExceeded): self.reserve(ledger, 'second')
        self.assertEqual(len(ledger.snapshot()['calls']), 1)

    def test_call_identity_and_cap_cannot_silently_change(self):
        ledger = BudgetLedger(self.path, cap_usd=100); self.reserve(ledger)
        with self.assertRaises(ValueError): self.reserve(ledger, rate=9)
        with self.assertRaises(ValueError): self.reserve(ledger, seconds=100)
        with self.assertRaises(ValueError):
            ledger.reserve('batch', identity={'changed': True}, timeout_seconds=7200, hourly_rate_usd=4.743648)
        with self.assertRaises(ValueError): BudgetLedger(self.path, cap_usd=200)
        with self.assertRaises(ValueError): self.reserve(ledger, 'bad', rate=float('inf'))

    def test_api_and_modal_share_cap_and_preserve_ambiguous_api_reservation(self):
        ledger=BudgetLedger(self.path,cap_usd=60)
        self.reserve(ledger,'modal')
        quote={'identity':{'provider':'example','model':'pinned','request_hash':'abc'},
               'input_tokens':15000000,'max_output_tokens':4000000,
               'input_usd_per_million':1,'output_usd_per_million':5,
               'pricing_reference':'official price page, batch, 2026-11-02',
               'input_count_basis':'provider count response sha256:example'}
        record=ledger.reserve_tokens('api',**quote)
        self.assertEqual(record['reservation_cents'],4025)
        self.assertEqual(record['cost_basis'],'api_token_quote')
        self.assertNotIn('hourly_rate_usd',record)
        self.assertEqual(ledger.snapshot()['totals']['outstanding_reserved_cents'],5117)
        with self.assertRaises(BudgetExceeded): self.reserve(ledger,'extra')
        ledger.mark_spawn_intent('api')
        ledger.mark_uncertain('api',reason='Provider response lost after submit')
        with self.assertRaises(ValueError): ledger.mark_spawn_intent('api')
        with self.assertRaises(ValueError): ledger.release_unspawned('api',reason='Unknown submit result')
        self.assertEqual(ledger.reserve_tokens('api',**quote)['state'],'uncertain')
        ledger.attach_call('api','batch-1')
        ledger.reconcile('api',estimated_cost_usd=31.25,
                         evidence={'basis':'returned billed token usage at frozen rates','reference':'usage.json'})
        self.assertEqual(ledger.snapshot()['totals']['completed_estimate_cents'],3125)
        self.assertEqual(ledger.snapshot()['totals']['outstanding_reserved_cents'],1092)
        for changes in ({'input_tokens':15000001},{'max_output_tokens':4000001},
                        {'output_usd_per_million':10},{'pricing_reference':'changed'}):
            with self.assertRaises(ValueError): ledger.reserve_tokens('api',**(quote|changes))

    def test_api_token_quote_rejects_invalid_counts_and_missing_cost_evidence(self):
        ledger=BudgetLedger(self.path,cap_usd=60)
        quote={'identity':{'request':'x'},'input_tokens':10,'max_output_tokens':20,
               'input_usd_per_million':1,'output_usd_per_million':5,
               'pricing_reference':'price proof','input_count_basis':'count proof'}
        for change in ({'input_tokens':-1},{'input_tokens':True},{'input_tokens':10.5},
                       {'max_output_tokens':0},{'max_output_tokens':'20'},
                       {'input_usd_per_million':float('nan')},{'output_usd_per_million':0},
                       {'pricing_reference':''},{'input_count_basis':None}):
            with self.assertRaises(ValueError): ledger.reserve_tokens('api',**(quote|change))
        self.assertEqual(ledger.snapshot()['calls'],{})

    def test_pre_spawn_failure_releases_but_ambiguous_or_cancelled_calls_stay_reserved(self):
        ledger = BudgetLedger(self.path, cap_usd=20); self.reserve(ledger)
        ledger.release_unspawned('batch', reason='local payload validation failed before spawn')
        self.assertEqual(ledger.snapshot()['totals']['outstanding_reserved_cents'], 0)
        self.reserve(ledger); ledger.mark_spawn_intent('batch')
        with self.assertRaises(ValueError): ledger.mark_spawn_intent('batch')
        with self.assertRaises(ValueError): ledger.release_unspawned('batch', reason='spawn raised, status unknown')
        ledger.mark_uncertain('batch', reason='controller lost spawn response')
        self.assertEqual(ledger.snapshot()['totals']['outstanding_reserved_cents'], 1092)
        ledger.attach_call('batch', 'fc-found'); ledger.mark_uncertain('batch', reason='cancellation requested')
        with self.assertRaises(ValueError): ledger.release_unspawned('batch', reason='cancelled')
        with self.assertRaises(ValueError): ledger.reconcile('batch', estimated_cost_usd=0, evidence={})
        self.assertEqual(ledger.snapshot()['totals']['outstanding_reserved_cents'], 1092)

    def test_reconciliation_preserves_completed_cost_and_honestly_records_overrun(self):
        ledger = BudgetLedger(self.path, cap_usd=20); self.reserve(ledger)
        ledger.mark_spawn_intent('batch'); ledger.attach_call('batch', 'fc-1')
        evidence = {'basis': 'worker elapsed estimate, not invoice', 'reference': 'execution.json#sha256'}
        ledger.reconcile('batch', estimated_cost_usd=3.001, evidence=evidence)
        self.assertEqual(ledger.snapshot()['totals']['completed_estimate_cents'], 301)
        self.assertEqual(ledger.snapshot()['totals']['outstanding_reserved_cents'], 0)
        count = len(ledger.snapshot()['events'])
        ledger.reconcile('batch', estimated_cost_usd=3.001, evidence=evidence)
        self.reserve(ledger)  # Existing completed attempt is not a new reservation.
        self.assertEqual(len(ledger.snapshot()['events']), count)
        with self.assertRaises(ValueError): ledger.mark_spawn_intent('batch')
        ledger.reconcile('batch', estimated_cost_usd=22, evidence={'basis': 'corrected provider usage', 'reference': 'usage-row-1'})
        self.assertTrue(ledger.snapshot()['totals']['over_admission_limit'])
        with self.assertRaises(BudgetExceeded): self.reserve(ledger, 'next', rate=1, seconds=1)

    def test_different_remote_call_needs_new_attempt_key_and_reservation(self):
        ledger = BudgetLedger(self.path, cap_usd=100); self.reserve(ledger)
        ledger.mark_spawn_intent('batch'); ledger.attach_call('batch', 'fc-1')
        with self.assertRaises(ValueError): ledger.attach_call('batch', 'fc-2')
        self.reserve(ledger, 'retry-1'); ledger.mark_spawn_intent('retry-1')
        with self.assertRaises(ValueError): ledger.attach_call('retry-1', 'fc-1')
        ledger.attach_call('retry-1', 'fc-2')
        self.assertEqual(ledger.snapshot()['totals']['outstanding_reserved_cents'], 2184)

    def test_thread_transactions_cannot_overbook(self):
        ledger = BudgetLedger(self.path, cap_usd=10)
        def attempt(index):
            other = BudgetLedger(self.path, cap_usd=10)
            try: self.reserve(other, str(index), rate=1, seconds=3600); return True
            except BudgetExceeded: return False
        with ThreadPoolExecutor(max_workers=4) as pool:
            accepted = list(pool.map(attempt, range(12)))
        self.assertEqual(sum(accepted), 4)
        self.assertEqual(ledger.snapshot()['totals']['outstanding_reserved_cents'], 460)

    def test_process_transactions_cannot_overbook(self):
        ledger = BudgetLedger(self.path, cap_usd=10)
        script = '''import sys
from bam_budget import BudgetLedger, BudgetExceeded
ledger=BudgetLedger(sys.argv[1],cap_usd=10)
try:
 ledger.reserve(sys.argv[2],identity={"key":sys.argv[2]},timeout_seconds=3600,hourly_rate_usd=2)
 print("accepted")
except BudgetExceeded: print("denied")
'''
        scripts = Path(__file__).resolve().parents[1] / 'scripts'
        processes = [subprocess.Popen([sys.executable, '-c', script, str(self.path), str(i)],
                                      cwd=scripts, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(4)]
        results = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 0, stderr); results.append(stdout.strip())
        self.assertEqual(results.count('accepted'), 2)
        self.assertEqual(results.count('denied'), 2)
        self.assertEqual(ledger.snapshot()['totals']['outstanding_reserved_cents'], 460)

    def test_failed_atomic_replace_preserves_previous_complete_ledger(self):
        ledger = BudgetLedger(self.path, cap_usd=20); before = self.path.read_bytes()
        with patch('bam_budget.os.replace', side_effect=OSError('simulated write interruption')):
            with self.assertRaises(OSError): self.reserve(ledger)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(ledger.snapshot()['calls'], {})
        self.assertFalse(list(self.path.parent.glob('.budget-*')))

    def test_corrupted_ledger_fails_closed(self):
        ledger = BudgetLedger(self.path, cap_usd=20); self.reserve(ledger)
        self.path.write_text('{unfinished')
        with self.assertRaises(json.JSONDecodeError): ledger.snapshot()


if __name__ == '__main__': unittest.main()
