"""Budget-reserved single-call synthetic LoRA inference probe; reconnect never respawn."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import modal
from bam_budget import BudgetLedger
from bam_student_vllm_probe_cloud import app, measure, digest, write_json, ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='config/bam_pipeline_v4_reference.json')
    p.add_argument('--budget-config',default='config/bam_budget_150_20261102.json')
    p.add_argument('--run-id',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    budget=json.loads(Path(a.budget_config).read_text());ledger=BudgetLedger(budget['ledger_path'],cap_usd=budget['cap_usd'])
    config=json.loads(Path(a.config).read_text())
    config['student_vllm']={'max_num_seqs':4,'durable_chunk_size':4,'max_num_batched_tokens':8192}
    request={'run_id':a.run_id,'config':config,'expected_worker':digest(ROOT/'scripts/bam_training.py'),
             'expected_probe':digest(ROOT/'scripts/bam_student_vllm_probe_worker.py'),
             'expected_backend':digest(ROOT/'scripts/bam_student_vllm.py')}
    if (out/'request.json').exists() and json.loads((out/'request.json').read_text())!=request:
        raise ValueError('Immutable probe request changed')
    write_json(out/'request.json',request)
    key='student-vllm-probe-'+a.run_id
    record=ledger.reserve(key,identity=request,timeout_seconds=600,hourly_rate_usd=4.743648)
    def collect(call):
        while True:
            try:response=call.get(timeout=45);break
            except Exception as error:
                if isinstance(error,TimeoutError) or type(error).__name__=='TimeoutError':
                    print('Waiting for single reserved vLLM LoRA probe',flush=True);continue
                ledger.mark_uncertain(key,reason=str(error));raise
        write_json(out/'response.json',response);(out/'execution.log').write_text(response['raw_log'])
        if response['result'] is not None:write_json(out/'result.json',response['result'])
        reconcile(response)
        print(json.dumps({'run_id':a.run_id,'error':response['error'],'output':str(out)}),flush=True)
    def reconcile(response):
        if ledger.snapshot()['calls'][key]['state']=='completed':return
        callmeta=json.loads((out/'call.json').read_text())
        seconds=(datetime.now(timezone.utc)-datetime.fromisoformat(callmeta['started_utc'])).total_seconds()+30
        # If local retrieval resumes much later this deliberately overestimates; never undercounts idle ambiguity.
        cost=seconds/3600*4.743648*1.15
        ledger.reconcile(key,estimated_cost_usd=cost,evidence={'basis':'completed remote response; local spawn-to-retrieval elapsed plus30s and15%; not invoice',
                                                            'reference':str((out/'response.json').resolve())})
    if (out/'response.json').exists():reconcile(json.loads((out/'response.json').read_text()));return
    if (out/'call.json').exists():
        saved=json.loads((out/'call.json').read_text());ledger.attach_call(key,saved['call_id'])
        collect(modal.FunctionCall.from_id(saved['call_id']));return
    if record['state']!='reserved' or (out/'spawn_intent.json').exists():
        raise RuntimeError('Unknown prior spawn intent; inspect instead of creating another call')
    with modal.enable_output(),app.run(detach=True):
        ledger.mark_spawn_intent(key)
        write_json(out/'spawn_intent.json',{'utc':datetime.now(timezone.utc).isoformat()})
        try:call=measure.spawn(**request)
        except Exception as error:ledger.mark_uncertain(key,reason='Spawn outcome unknown: '+str(error));raise
        write_json(out/'call.json',{'call_id':call.object_id,'started_utc':datetime.now(timezone.utc).isoformat()})
        ledger.attach_call(key,call.object_id)
        print(json.dumps({'call_id':call.object_id,'reservation_cents':record['reservation_cents']}),flush=True)
        collect(call)


if __name__=='__main__':main()
