"""Local admission ledger for explicitly authorized remaining compute spend.

Estimates are not provider invoices or guarantees. No provider/API interaction.
POSIX flock plus a per-path thread lock serialize local JSON transactions. Each
remote attempt needs its own durable key; reconnects reuse that attempt's key.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading


_LOCKS = {}
_LOCKS_GUARD = threading.Lock()
BUFFER_CENTS = 500
MARGIN = Decimal('1.15')
STATES = {'reserved', 'spawn_intent', 'active', 'uncertain', 'completed', 'released'}


class BudgetExceeded(RuntimeError):
    pass


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(name + ' must be an explicit numeric value')
    number = Decimal(str(value))
    if not number.is_finite() or number < 0 or (positive and number == 0):
        raise ValueError('Invalid ' + name)
    return number


def _cents(value, *, cap=False):
    return int((value * 100).to_integral_value(rounding=ROUND_FLOOR if cap else ROUND_CEILING))


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _totals(ledger):
    records = ledger['calls'].values()
    completed = sum(r['estimated_cost_cents'] for r in records if r['state'] == 'completed')
    outstanding = sum(r['reservation_cents'] for r in ledger['calls'].values()
                      if r['state'] not in ('completed', 'released'))
    available = ledger['cap_cents'] - ledger['buffer_cents'] - completed - outstanding
    return {'completed_estimate_cents': completed, 'outstanding_reserved_cents': outstanding,
            'available_for_new_calls_cents': available, 'over_admission_limit': available < 0}


class BudgetLedger:
    def __init__(self, path, *, cap_usd):
        self.path = Path(path).resolve()
        self.cap_cents = _cents(_number(cap_usd, 'cap_usd', positive=True), cap=True)
        if self.cap_cents <= BUFFER_CENTS:
            raise ValueError('Explicit cap must exceed the retained $5 buffer')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCKS_GUARD:
            self.thread_lock = _LOCKS.setdefault(str(self.path), threading.RLock())
        with self._transaction(): pass

    @contextmanager
    def _transaction(self):
        with self.thread_lock, open(str(self.path) + '.lock', 'a+b') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if self.path.exists():
                    ledger = json.loads(self.path.read_text())
                    if (ledger.get('schema_version') != 1 or ledger.get('cap_cents') != self.cap_cents
                            or ledger.get('buffer_cents') != BUFFER_CENTS or ledger.get('reservation_multiplier') != '1.15'):
                        raise ValueError('Existing ledger cap/policy differs; cap changes require explicit separate review')
                    for key, record in ledger['calls'].items():
                        if record.get('state') not in STATES or record.get('call_key') != key:
                            raise ValueError('Invalid ledger state')
                        for field in ('reservation_cents', 'estimated_cost_cents'):
                            if field in record and (type(record[field]) is not int or record[field] < 0):
                                raise ValueError('Invalid ledger cost')
                else:
                    ledger = {'schema_version': 1, 'cap_cents': self.cap_cents,
                              'buffer_cents': BUFFER_CENTS, 'reservation_multiplier': '1.15',
                              'accounting': 'local estimates; not provider billing', 'calls': {}, 'events': []}
                before = _hash(ledger)
                yield ledger
                if not self.path.exists() or before != _hash(ledger): self._write(ledger)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _write(self, ledger):
        fd, name = tempfile.mkstemp(prefix='.budget-', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(ledger, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
            os.replace(name, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try: os.fsync(directory)
            finally: os.close(directory)
        finally:
            if os.path.exists(name): os.unlink(name)

    @staticmethod
    def _event(ledger, key, action, **details):
        ledger['events'].append({'utc': datetime.now(timezone.utc).isoformat(),
                                 'call_key': key, 'action': action, **details})

    def reserve(self, call_key, *, identity, timeout_seconds, hourly_rate_usd):
        """Reserve timeout × allocated hourly rate × 1.15, rounded up to cents.

        Existing attempt identity/rate/timeout must match. A completed/active call
        returns its record without new cost; this is NOT permission to respawn.
        A released pre-spawn attempt can be reserved again if room still exists.
        """
        timeout = _number(timeout_seconds, 'timeout_seconds', positive=True)
        rate = _number(hourly_rate_usd, 'hourly_rate_usd', positive=True)
        return self._reserve_binding(call_key, identity, {
            'timeout_seconds': str(timeout.normalize()), 'hourly_rate_usd': str(rate.normalize()),
            'reservation_cents': _cents(timeout * rate / 3600 * MARGIN)})

    def reserve_tokens(self, call_key, *, identity, input_tokens, max_output_tokens,
                       input_usd_per_million, output_usd_per_million,
                       pricing_reference, input_count_basis):
        """Reserve API input plus bounded output/reasoning at explicit quoted rates.

        Token counts must include the whole submitted batch. This is admission,
        not submission: normal intent/ID/uncertainty rules still apply. The caller
        must establish provider token counts or a conservative bound, include
        reasoning in the output allowance, and price any additional features.
        Credits/cache discounts are not silently deducted from the shared cap.
        """
        if type(input_tokens) is not int or input_tokens < 0:
            raise ValueError('Input token count must be a nonnegative integer')
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError('Maximum output token count must be a positive integer')
        for label, value in (('pricing_reference', pricing_reference), ('input_count_basis', input_count_basis)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(label + ' is required')
        input_rate = _number(input_usd_per_million, 'input_usd_per_million', positive=True)
        output_rate = _number(output_usd_per_million, 'output_usd_per_million', positive=True)
        quote = (input_tokens * input_rate + max_output_tokens * output_rate) / 1000000
        return self._reserve_binding(call_key, identity, {
            'cost_basis': 'api_token_quote', 'input_tokens': input_tokens,
            'max_output_tokens': max_output_tokens,
            'input_usd_per_million': str(input_rate.normalize()),
            'output_usd_per_million': str(output_rate.normalize()),
            'pricing_reference': pricing_reference, 'input_count_basis': input_count_basis,
            'reservation_cents': _cents(quote * MARGIN)})

    def _reserve_binding(self, call_key, identity, resources):
        if not isinstance(call_key, str) or not call_key or len(call_key) > 256:
            raise ValueError('A stable nonempty durable call key is required')
        if not isinstance(identity, dict) or not identity: raise ValueError('Call identity is required')
        binding = {'identity_sha256': _hash(identity), **resources}
        with self._transaction() as ledger:
            old = ledger['calls'].get(call_key)
            if old:
                if any(old.get(k) != v for k, v in binding.items()):
                    raise ValueError('Durable call identity/resources changed')
                if old['state'] != 'released': return json.loads(json.dumps(old))
            if binding['reservation_cents'] > _totals(ledger)['available_for_new_calls_cents']:
                raise BudgetExceeded('New call would consume outstanding/completed spend or the $5 buffer')
            record = {'call_key': call_key, **binding, 'state': 'reserved', 'call_id': None}
            ledger['calls'][call_key] = record
            self._event(ledger, call_key, 'reserve', reservation_cents=record['reservation_cents'])
            return dict(record)

    def mark_spawn_intent(self, call_key):
        """Persist immediately BEFORE spawn. An existing intent must be inspected,
        never interpreted as permission to spawn again after a crash.
        """
        with self._transaction() as ledger:
            record = ledger['calls'][call_key]
            if record['state'] != 'reserved': raise ValueError('Attempt already started; reconnect or reconcile')
            record['state'] = 'spawn_intent'; self._event(ledger, call_key, 'spawn_intent')

    def attach_call(self, call_key, call_id):
        """Bind/reconnect a persisted provider call ID without a second charge."""
        if not isinstance(call_id, str) or not call_id: raise ValueError('Call ID required')
        with self._transaction() as ledger:
            record = ledger['calls'][call_key]
            if record['state'] in ('completed', 'released'): raise ValueError('Attempt is already closed')
            if record['call_id'] not in (None, call_id): raise ValueError('Different call ID for the same attempt')
            if any(r['call_key'] != call_key and r['call_id'] == call_id for r in ledger['calls'].values()):
                raise ValueError('Provider call ID already accounted under another key')
            if record['state'] == 'active' and record['call_id'] == call_id: return
            record.update(state='active', call_id=call_id)
            self._event(ledger, call_key, 'attach_call', call_id=call_id)

    def release_unspawned(self, call_key, *, reason):
        """Release only a failure known to precede any remote spawn attempt."""
        if not isinstance(reason, str) or not reason.strip(): raise ValueError('Release reason required')
        with self._transaction() as ledger:
            record = ledger['calls'][call_key]
            if record['state'] == 'released': return
            if record['state'] != 'reserved' or record['call_id'] is not None:
                raise ValueError('Spawned, cancelled or uncertain billing must remain reserved until reconciled')
            record['state'] = 'released'; self._event(ledger, call_key, 'release_unspawned', reason=reason)

    def mark_uncertain(self, call_key, *, reason):
        """Cancellation, spawn ambiguity, transport failure: retain full reserve."""
        if not isinstance(reason, str) or not reason.strip(): raise ValueError('Uncertainty reason required')
        with self._transaction() as ledger:
            record = ledger['calls'][call_key]
            if record['state'] in ('released', 'completed'): raise ValueError('Attempt is already closed')
            record['state'] = 'uncertain'; self._event(ledger, call_key, 'uncertain', reason=reason)

    def reconcile(self, call_key, *, estimated_cost_usd, evidence):
        """Explicitly replace a reservation with an evidenced completed estimate.

        Cancellation alone is not evidence of zero billing. Caller must establish
        completion/termination or provider usage. Overruns are recorded honestly,
        even beyond the cap; they block further reservations rather than vanish.
        Corrections are allowed and retain prior estimates in the event history.
        """
        cents = _cents(_number(estimated_cost_usd, 'estimated_cost_usd'))
        if not isinstance(evidence, dict) or not evidence.get('basis') or not evidence.get('reference'):
            raise ValueError('Reconciliation requires evidence basis and reference')
        _hash(evidence)
        with self._transaction() as ledger:
            record = ledger['calls'][call_key]
            if record['state'] in ('reserved', 'released'): raise ValueError('No remote attempt to reconcile')
            if record['state'] == 'completed' and record['estimated_cost_cents'] == cents and record['evidence'] == evidence:
                return
            self._event(ledger, call_key, 'reconcile', previous_state=record['state'],
                        previous_estimated_cost_cents=record.get('estimated_cost_cents'),
                        estimated_cost_cents=cents, evidence=evidence)
            record.update(state='completed', estimated_cost_cents=cents, evidence=evidence)

    def snapshot(self):
        with self._transaction() as ledger:
            return json.loads(json.dumps({**ledger, 'totals': _totals(ledger)}))
