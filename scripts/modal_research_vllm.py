"""Deploy once with modal deploy; submit jobs with this script's Python CLI."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import threading

import modal
from modal_research import image as mlx_image, models, runs, digest, unpack, run_logged

mlx_image = mlx_image.add_local_python_source('modal_research')

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = 'cs230-nba-research-vllm'
app = modal.App(APP_NAME)
encoder_image = mlx_image.add_local_file(ROOT/'scripts/encode_research_mlx.py',
                                        '/opt/encode_research_mlx.py')
image = (modal.Image.debian_slim(python_version='3.14')
         .apt_install('libnuma1')
         .add_local_file(ROOT/'requirements-research-vllm.txt','/tmp/requirements.txt',copy=True)
         .run_commands('python -m venv /opt/nba-venv',
                       '/opt/nba-venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt')
         .env({'HF_HUB_DISABLE_TELEMETRY':'1','TOKENIZERS_PARALLELISM':'false','VLLM_NO_USAGE_STATS':'1'})
         .add_local_file(ROOT/'scripts/research_vllm_worker.py','/opt/research_vllm_worker.py',copy=True)
         .add_local_file(ROOT/'scripts/convert_research_weights.py','/opt/convert_research_weights.py',copy=True)
         .add_local_python_source('modal_research'))


@app.function(image=image, volumes={'/models':models}, cpu=4, memory=32768,
              timeout=1800, min_containers=0, max_containers=1, single_use_containers=True, retries=0,
              nonpreemptible=True)
def convert_model(config, expected, reference):
    source = Path('/models')/Path(config['model_path']).name
    target = source.with_name(source.name+'-dequantized-fp16-direct')
    for name, wanted in expected.items():
        if digest(source/Path(name).name) != wanted:
            raise ValueError('Original pinned checkpoint hash changed')
    if not (target/'conversion.json').exists():
        if target.exists():
            raise ValueError('Incomplete earlier conversion exists; preserve and inspect it')
        Path('/tmp/weight-reference.json').write_text(json.dumps(reference))
        subprocess.run(['/opt/nba-venv/bin/python','-u','/opt/convert_research_weights.py',str(source),str(target),
                        '/tmp/weight-reference.json'],check=True,timeout=1500)
        metadata = {'source_model_id':config['model_id'],'source_revision':config['model_revision'],
                    'source_files':expected,'method':'Direct affine four-bit unpacking in float32 then FP16; sampled rows match original MLX GPU exactly; no fitting or new checkpoint',
                    'converter_sha256':digest('/opt/convert_research_weights.py'),'gpu_reference_first_row_sha256':reference,
                    'output_files':{p.name:digest(p) for p in sorted(target.iterdir()) if p.is_file()},
                    'scope':'Restores the quantized weight levels, not the original unquantized Qwen weights.'}
        (target/'conversion.json').write_text(json.dumps(metadata,indent=2)+'\n')
    models.commit()
    return {'model_path':str(target),'conversion':json.loads((target/'conversion.json').read_text()),
            'deployed_source_sha256':{name:digest(Path('/opt')/name) for name in
                                     ('research_vllm_worker.py','convert_research_weights.py')}}


@app.function(image=image,gpu='H100',volumes={'/models':models,'/runs':runs},cpu=4,memory=49152,
              timeout=7200,min_containers=0,max_containers=1,single_use_containers=True,retries=0)
def execute(payload, expected_sha, run_id, token_reference, model_path, benchmark, resume_run='', invocation_budget=3):
    if invocation_budget not in (3,4) or invocation_budget==4 and not resume_run:
        raise ValueError('A fourth invocation requires an explicitly inspected resume')
    prefix = 'research_compute_bundle' if benchmark else 'nba_research'
    destination = Path('/runs/vllm')/(resume_run or run_id)
    root = destination/prefix
    if destination.exists() and benchmark:
        # A preempted benchmark must restart uncached to keep timing meaningful.
        destination = destination.with_name(run_id+'-retry-'+datetime.now(timezone.utc).strftime('%H%M%S%f'))
        root = destination/prefix
    if root.exists():
        if (root/'input_archive.sha256').read_text().strip()!=expected_sha:
            raise ValueError('Resume archive differs from the original cloud transfer')
    else:
        root = unpack(payload,expected_sha,destination,prefix)
        (root/'input_archive.sha256').write_text(expected_sha+'\n')
    attempts_path=root/'invocations.json'
    attempts=json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    if len(attempts)>=invocation_budget:
        raise ValueError('Cloud invocation budget exhausted; inspect preserved results before resuming')
    attempts.append(datetime.now(timezone.utc).isoformat())
    attempts_path.write_text(json.dumps(attempts)+'\n')
    manifest = json.loads((root/('probe/manifest.json' if benchmark else 'transfer_manifest.json')).read_text())
    for name, wanted in manifest['files'].items():
        if digest(root/name) != wanted:
            raise ValueError('Transferred source differs: '+name)
    (root/'token_reference.json').write_text(json.dumps(token_reference)+'\n')
    (root/'scripts/research_vllm_worker.py').write_bytes(Path('/opt/research_vllm_worker.py').read_bytes())
    hardware = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
    execution = {'run_id':run_id,'gpu':hardware,'input_archive_sha256':expected_sha,
                 'benchmark':benchmark,'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running',
                 'remote_directory':str(root),'invocations':attempts}
    command = [str(root/'.venv/bin/python'),str(root/'scripts/research_vllm_worker.py'),'--model-path',model_path]
    if benchmark:
        command += ['--benchmark']
    # vLLM's environment is separate from MLX. Do not prepend MLX CUDA libraries.
    with (root/'execution.log').open('w') as log:
        process = subprocess.Popen(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        timer = threading.Timer(1100 if benchmark else 6900,process.kill)
        timer.start()
        try:
            for line in process.stdout:
                log.write(line); log.flush()
                print(line,end='',flush=True)
            code = process.wait()
            execution.update(status='complete' if code==0 else 'failed',returncode=code)
        finally:
            timer.cancel()
            if process.poll() is None:
                process.kill(); process.wait()
    execution['finished_utc'] = datetime.now(timezone.utc).isoformat()
    paths = sorted(p for folder in ('data/processed/game_note_research','reports/game_note_research')
                   for p in (root/folder).rglob('*') if p.is_file())
    execution['output_files'] = {str(p.relative_to(root)):digest(p) for p in paths}
    (root/'modal_execution.json').write_text(json.dumps(execution,indent=2)+'\n')
    content = io.BytesIO()
    with tarfile.open(fileobj=content,mode='w:gz') as archive:
        for path in paths+[root/'execution.log',root/'modal_execution.json']:
            archive.add(path,arcname=str(path.relative_to(root)),recursive=False)
    runs.commit()
    return {'execution':execution,'archive':content.getvalue()}


@app.function(image=encoder_image,gpu='H100',volumes={'/models':models,'/runs':runs},
              cpu=4,memory=32768,timeout=7200,min_containers=0,max_containers=2,
              single_use_containers=True,retries=0)
def encode_reports(request, expected_worker_sha, run_id):
    import re
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',run_id):
        raise ValueError('Invalid encoder run ID')
    if digest('/opt/encode_research_mlx.py')!=expected_worker_sha:
        raise ValueError('Deployed encoder differs; redeploy before encoding')
    root=Path('/runs/encodings')/run_id
    root.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(request,sort_keys=True,ensure_ascii=False)
    request_path=root/'request.json'
    if request_path.exists() and request_path.read_text()!=payload:
        raise ValueError('Encoder resume request differs')
    request_path.write_text(payload)
    try:
        run_logged(['/opt/nba-venv/bin/python','-u','/opt/encode_research_mlx.py',
                    str(request_path),str(root/'results')],root,root/'execution.log',6900)
        return json.loads((root/'results/result.json').read_text())
    finally:
        runs.commit()


def deployed_functions(environment='main'):
    """Resolve existing functions; never create an ephemeral app or start compute."""
    functions = {}
    for name in ('convert_model', 'execute'):
        function = modal.Function.from_name(APP_NAME, name, environment_name=environment)
        try:
            function.hydrate()
        except modal.exception.NotFoundError as error:
            raise RuntimeError(
                f'Deploy {APP_NAME} first: .venv/bin/modal deploy --env {environment} '
                'scripts/modal_research_vllm.py'
            ) from error
        functions[name] = function
    return functions


def verify_deployed_sources(conversion):
    expected = {name:digest(ROOT/'scripts'/name) for name in
                ('research_vllm_worker.py','convert_research_weights.py')}
    if conversion.get('deployed_source_sha256') != expected:
        raise ValueError('Deployed worker/converter differs from local code; redeploy before starting GPU work')


def main(snapshot: str = '', finish_local: bool = False, resume_run: str = '',
         environment: str = 'main', check_deployment: bool = False, invocation_budget: int = 3):
    if invocation_budget not in (3,4) or invocation_budget==4 and not resume_run:
        raise ValueError('A fourth invocation requires an explicitly inspected resume')
    if check_deployment:
        if snapshot or finish_local or resume_run:
            raise ValueError('--check-deployment cannot be combined with job options')
        functions = deployed_functions(environment)
        print(json.dumps({'deployment':APP_NAME,'environment':environment,
                          'functions':{name:function.object_id for name,function in functions.items()},
                          'compute_started':False},indent=2))
        return
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    import run_game_note_research as r
    from transformers import AutoTokenizer
    benchmark = not snapshot
    bundle = ROOT/'data/probes/research_compute_bundle'
    path = ROOT/'data/probes/research_compute_bundle.tar.gz' if benchmark else Path(snapshot).resolve()
    if not benchmark and not path.is_relative_to(ROOT/'data/probes/cloud_research_transfer'):
        raise ValueError('Use the explicit research-only snapshot')
    if not benchmark:
        plan = r.read_json(r.REPORT/'cloud_execution_plan.json')
        if plan['worker_sha256'] != r.sha(ROOT/'scripts/research_vllm_worker.py'):
            raise ValueError('Research worker differs from the reviewed benchmark')
        if plan['snapshot_sha256'] != r.sha(path):
            raise ValueError('Full run requires the reviewed current transfer snapshot')
        checked = r.read_json(ROOT/plan['benchmark_result'])
        if r.sha(ROOT/plan['benchmark_result']) != plan['benchmark_result_sha256']:
            raise ValueError('Benchmark result changed')
        if checked['schema_and_source_checks']!='passed' or sum(not c['passed'] for c in checked['factual_challenges'])>2:
            raise ValueError('Training-only benchmark failed the declared quality guard')
    if resume_run:
        import re
        if benchmark or not re.fullmatch(r'\d{8}T\d{6}Z', resume_run):
            raise ValueError('Resume requires a full-research run ID')
        previous = r.read_json(ROOT/'data/probes/modal_vllm'/resume_run/'execution.json')
        if previous['benchmark'] or previous['status'] != 'failed' or previous['input_archive_sha256'] != r.sha(path):
            raise ValueError('Only the same failed, preserved workload can be resumed')
    config = r.read_json(r.CONFIG)
    manifest = r.read_json(bundle/'probe/manifest.json')
    units = r.read_rows(bundle/'probe/units.jsonl' if benchmark else r.DATA/'units.jsonl')
    tokenizer = AutoTokenizer.from_pretrained(ROOT/config['model_path'],local_files_only=True)
    token_reference = {u['id']:r.value_hash(r.encode_messages(tokenizer,r.extraction_messages(u))) for u in units}
    functions = deployed_functions(environment)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = ROOT/'data/probes/modal_vllm'/run_id
    output.mkdir(parents=True,exist_ok=False)
    status = {'run_id':run_id,'stage':'conversion','benchmark':benchmark,'snapshot':str(path),'resume_run':resume_run or None,
              'deployment':APP_NAME,'environment':environment}
    r.write_json(output/'status.json',status)
    try:
        reference=r.read_json(ROOT/'data/probes/dequantization_gpu_check.json')['first_row_sha256']
        conversion = functions['convert_model'].remote(config,manifest['model_files'],reference)
        r.write_json(output/'conversion.json',conversion)
        verify_deployed_sources(conversion)
        status['stage']='gpu_execution'; r.write_json(output/'status.json',status)
        payload = path.read_bytes()
        call = functions['execute'].spawn(payload,hashlib.sha256(payload).hexdigest(),run_id,token_reference,
                                          conversion['model_path'],benchmark,resume_run,invocation_budget)
        status['function_call_id']=call.object_id
        r.write_json(output/'status.json',status)
        print(json.dumps({'run_id':run_id,'function_call_id':call.object_id}),flush=True)
        result = call.get()
        (output/'results.tar.gz').write_bytes(result.pop('archive'))
        r.write_json(output/'execution.json',result['execution'])
        status['stage']=result['execution']['status']
    except Exception as error:
        status.update(stage='failed',error=str(error)); r.write_json(output/'status.json',status)
        raise
    r.write_json(output/'status.json',status)
    print(json.dumps({'output':str(output),**status}),flush=True)
    if status['stage']!='complete':
        raise RuntimeError('Remote run failed; partial artifacts returned for inspection')
    if finish_local:
        if benchmark:
            raise ValueError('A benchmark cannot feed the forecasting experiment')
        status['stage']='local_source_validation_and_forecasting'
        r.write_json(output/'status.json',status)
        commands = [
            [str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/import_modal_research.py'),str(output)],
            [str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/run_research_experiment.py'),'--stage','all'],
        ]
        try:
            for command in commands:
                subprocess.run(command,cwd=ROOT,check=True)
        except Exception as error:
            status.update(stage='local_followup_failed',error=str(error))
            r.write_json(output/'status.json',status)
            raise
        status['stage']='experiment_complete'
        r.write_json(output/'status.json',status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', default='', help='Reviewed research snapshot; omit for the benchmark')
    parser.add_argument('--finish-local', action='store_true', help='Import and run local follow-up after full research')
    parser.add_argument('--resume-run', default='', help='Resume a preserved failed full-research run')
    parser.add_argument('--environment', default='main', help='Modal environment containing the deployment')
    parser.add_argument('--check-deployment', action='store_true', help='Resolve deployed functions without starting compute')
    parser.add_argument('--invocation-budget',type=int,default=3,choices=(3,4),
                        help='Bounded retry allowance; fourth call only after inspecting a failed resume')
    main(**vars(parser.parse_args()))
