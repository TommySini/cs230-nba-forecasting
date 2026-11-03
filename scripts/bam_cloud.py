"""Stable, zero-idle Modal app for BAM model fetch, inference, and training.

Deploy explicitly with `.venv/bin/modal deploy --env main scripts/bam_cloud.py`.
Importing or checking deployment never launches compute.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import threading
import time

import modal

ROOT=Path(__file__).resolve().parents[1]
APP_NAME='cs230-nba-bam'
app=modal.App(APP_NAME)
models=modal.Volume.from_name('cs230-nba-research-models',create_if_missing=True,version=2)
runs=modal.Volume.from_name('cs230-nba-research-runs',create_if_missing=True,version=2)
MODEL_CACHE='/models/bam/hf_cache'
common_env={'HF_HUB_DISABLE_TELEMETRY':'1','TOKENIZERS_PARALLELISM':'false','VLLM_NO_USAGE_STATS':'1'}
fetch_image=modal.Image.debian_slim(python_version='3.12').pip_install('huggingface-hub==0.30.2').env(common_env)
inference_image=(modal.Image.debian_slim(python_version='3.14').apt_install('libnuma1')
    .add_local_file(ROOT/'requirements-research-vllm.txt','/tmp/requirements.txt',copy=True)
    .run_commands('python -m venv /opt/nba-venv','/opt/nba-venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt')
    .env(common_env)
    .add_local_file(ROOT/'scripts/bam_inference.py','/opt/bam_inference.py',copy=True)
    .add_local_file(ROOT/'scripts/bam_checkpoint.py','/opt/bam_checkpoint.py',copy=True)
    .add_local_file(ROOT/'scripts/bam_training.py','/opt/bam_training.py',copy=True))
training_image=(modal.Image.debian_slim(python_version='3.12')
    .add_local_file(ROOT/'requirements-bam-training.txt','/tmp/bam-requirements.txt',copy=True)
    .run_commands('python -m venv /opt/bam-venv','/opt/bam-venv/bin/pip install --no-cache-dir -r /tmp/bam-requirements.txt')
    .env(common_env | {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
    .add_local_file(ROOT/'scripts/bam_checkpoint.py','/opt/bam_checkpoint.py',copy=True)
    .add_local_file(ROOT/'scripts/bam_training.py','/opt/bam_training.py',copy=True))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n'); temporary.replace(path)


def checked_id(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,119}',value):
        raise ValueError('Invalid run ID')
    return value


def checked_model(spec):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',spec.get('model_id','')):
        raise ValueError('Expected explicit Hugging Face owner/repository')
    if not re.fullmatch(r'[a-f0-9]{40}',spec.get('model_revision','')):
        raise ValueError('Immutable 40-character model revision required')
    if spec.get('quantization') not in (None,'awq'):
        raise ValueError('Unsupported model quantization')
    return {k:spec[k] for k in ('model_id','model_revision','quantization') if k in spec}


def model_key(spec):
    spec=checked_model(spec)
    return spec['model_id'].replace('/','--')+'--'+spec['model_revision']


def safe_extract(payload,expected_sha,destination,max_bytes=8*1024**3):
    """Flat relative regular-file archives only; existing files can never change."""
    if hashlib.sha256(payload).hexdigest()!=expected_sha:
        raise ValueError('Archive transfer checksum mismatch')
    if len(payload)>1024**3:
        raise ValueError('Transfer archive exceeds 1GiB')
    root=Path(destination); root.mkdir(parents=True,exist_ok=True)
    members=[]; seen=set(); total=0
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
        for member in archive.getmembers():
            path=PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or not path.parts or '..' in path.parts
                    or '\\' in member.name or str(path) in seen):
                raise ValueError('Unsafe or duplicate archive entry: '+member.name)
            if path.parts[0] in ('.git','.venv'):
                raise ValueError('Environment/repository internals are not training artifacts')
            seen.add(str(path)); total+=member.size
            if total>max_bytes:
                raise ValueError('Expanded archive exceeds size limit')
            target=root.joinpath(*path.parts)
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError('Archive destination escapes run root')
            content=archive.extractfile(member).read()
            if len(content)!=member.size:
                raise ValueError('Incomplete archive member')
            if target.exists() and (not target.is_file() or digest(target)!=hashlib.sha256(content).hexdigest()):
                raise ValueError('Existing run artifact differs: '+str(path))
            members.append((target,content))
        # Validate every member before mutating any destination file.
        for target,content in members:
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(content)
    return {str(path.relative_to(root)):hashlib.sha256(content).hexdigest() for path,content in members}


def lock_request(path,request):
    path=Path(path)
    if path.exists() and json.loads(path.read_text())!=request:
        raise ValueError('Run ID already belongs to a different immutable request')
    write_json(path,request)


def _completed_artifact_marker(line):
    try:
        value=json.loads(line)
    except (ValueError,TypeError):
        return False
    if not isinstance(value,dict): return False
    return (any(key in value for key in ('checkpoint_complete','batch_complete','generation_attempt_complete','research_fit_complete'))
            or {'completed','total','batch_seconds'} <= value.keys())


def _run_logged(command,cwd,path,seconds=6900,commit=None,commit_interval=20):
    if not 0 < commit_interval <= 30:
        raise ValueError('Log commit interval must be within 0..30 seconds')
    with Path(path).open('a') as stream:
        process=subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        timer=threading.Timer(seconds,process.kill); timer.start()
        last_commit=time.monotonic()
        try:
            for line in process.stdout:
                stream.write(line);stream.flush();print(line,end='',flush=True)
                if commit and (_completed_artifact_marker(line) or time.monotonic()-last_commit>=commit_interval):
                    commit()
                    last_commit=time.monotonic()
            code=process.wait()
            if code:
                raise RuntimeError(f'Worker exited {code}; preserved log: {path}')
        finally:
            timer.cancel()
            if process.poll() is None:
                process.kill();process.wait()
            if process.stdout is not None:
                process.stdout.close()
            if commit:
                commit()


@app.function(image=fetch_image,volumes={'/models':models},cpu=4,memory=16384,
              timeout=7200,min_containers=0,max_containers=1,single_use_containers=True,retries=0)
def fetch_models(model_specs):
    """CPU-only download; immutable HF revisions and all retained file hashes."""
    from huggingface_hub import snapshot_download
    models.reload()
    manifests=[]
    try:
        for supplied in model_specs:
            spec=checked_model(supplied)
            target=Path(snapshot_download(spec['model_id'],revision=spec['model_revision'],cache_dir=MODEL_CACHE,
                token=False,allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.tiktoken','LICENSE','README.md']))
            if target.name!=spec['model_revision']:
                raise ValueError('Hub did not resolve requested immutable revision')
            files={str(p.relative_to(target)):digest(p) for p in sorted(target.rglob('*')) if p.is_file()}
            if not any(name.endswith('.safetensors') for name in files) or 'config.json' not in files:
                raise ValueError('Incomplete checkpoint download')
            manifest={'model_spec':spec,'snapshot_path':str(target),'files':files,
                      'total_bytes':sum((target/name).stat().st_size for name in files),
                      'manifest_sha256':object_hash({'model_spec':spec,'files':files})}
            path=Path('/models/bam/manifests')/(model_key(spec)+'.json')
            lock_request(path,manifest)
            manifests.append(manifest)
            models.commit()
        return {'models':manifests,'cache_dir':MODEL_CACHE,'gpu_started':False}
    finally:
        models.commit()


def generation_function(model_spec,config):
    options=dict(config.get('generation',{}))
    options.update(config.get('generation_by_model',{}).get(model_spec['model_id'],{}))
    parallelism=options.get('tensor_parallel_size',1)
    if type(parallelism) is not int or parallelism not in (1,2):
        raise ValueError('Generation requires one or two GPUs explicitly')
    return 'generate_heavy' if parallelism==2 else 'generate'


def _generate(requests,model_spec,run_id,config,expected_worker_hash,*,function_name):
    if generation_function(model_spec,config)!=function_name:
        raise ValueError('Model parallelism differs from allocated GPU worker')
    models.reload();runs.reload()
    checked_id(run_id)
    if digest('/opt/bam_inference.py')!=expected_worker_hash:
        raise ValueError('Deployed inference worker differs; redeploy first')
    spec=checked_model(model_spec)
    manifest=json.loads((Path('/models/bam/manifests')/(model_key(spec)+'.json')).read_text())
    if manifest['model_spec']!=spec:
        raise ValueError('Checkpoint spec differs from fetched model')
    root=Path('/runs/bam/generation')/run_id;root.mkdir(parents=True,exist_ok=True)
    request={'requests':requests,'model_manifest':manifest,'config':config,'worker_sha256':expected_worker_hash}
    lock_request(root/'request.json',request)
    runs.commit()
    try:
        _run_logged(['/opt/nba-venv/bin/python','-u','/opt/bam_inference.py','--root',str(root),
                     '--request',str(root/'request.json')],root,root/'execution.log',commit=runs.commit)
        return {'run_id':run_id,'summary':json.loads((root/'summary.json').read_text()),
                'identity':json.loads((root/'identity.json').read_text()),
                'responses':[json.loads(line) for line in (root/'responses.jsonl').read_text().splitlines()]}
    finally:
        runs.commit()


@app.function(image=inference_image,gpu='H100',volumes={'/models':models,'/runs':runs},cpu=6,memory=65536,
              timeout=7200,min_containers=0,max_containers=4,single_use_containers=True,retries=0)
def generate(requests,model_spec,run_id,config,expected_worker_hash):
    return _generate(requests,model_spec,run_id,config,expected_worker_hash,function_name='generate')


@app.function(image=inference_image,gpu='H200:2',volumes={'/models':models,'/runs':runs},cpu=12,memory=131072,
              timeout=7200,min_containers=0,max_containers=1,single_use_containers=True,retries=0)
def generate_heavy(requests,model_spec,run_id,config,expected_worker_hash):
    return _generate(requests,model_spec,run_id,config,expected_worker_hash,function_name='generate_heavy')


@app.function(image=training_image,gpu='H100',volumes={'/models':models,'/runs':runs},cpu=6,memory=65536,
              timeout=21600,min_containers=0,max_containers=1,single_use_containers=True,retries=0)
def train(payload_archive,expected_sha,run_id,stage,config,expected_worker_hash,expected_checkpoint_hash=None):
    models.reload();runs.reload();checked_id(run_id)
    if stage not in ('research-train','research-generate','research-repair','encode','forecast-train','evaluate','smoke','smoke-resume'):
        raise ValueError('Unknown worker stage')
    if digest('/opt/bam_training.py')!=expected_worker_hash:
        raise ValueError('Deployed training worker differs; redeploy first')
    if digest('/opt/bam_checkpoint.py')!=expected_checkpoint_hash:
        raise ValueError('Deployed checkpoint helper differs; supply its exact hash and redeploy')
    if config.get('cache_dir')!=MODEL_CACHE:
        raise ValueError('Training must use the already fetched BAM HF cache')
    spec=checked_model(config)
    model_manifest=json.loads((Path('/models/bam/manifests')/(model_key(spec)+'.json')).read_text())
    if model_manifest['model_spec']!=spec:
        raise ValueError('Fetched model does not match training config')
    for name,expected in model_manifest['files'].items():
        if digest(Path(model_manifest['snapshot_path'])/name)!=expected:
            raise ValueError('Fetched model changed: '+name)
    root=Path('/runs/bam/training')/run_id;root.mkdir(parents=True,exist_ok=True)
    scientific_config=dict(config)
    scientific_config['research']=dict(config.get('research',{}))
    fold_ids=scientific_config['research'].pop('fold_ids',None)
    repair_batch_id=scientific_config['research'].pop('repair_batch_id',None)
    lock_request(root/'cloud_config.json',scientific_config)
    stage_id=stage
    if fold_ids is not None:
        if stage!='research-generate' or not isinstance(fold_ids,list) or not fold_ids:
            raise ValueError('fold_ids is only allowed for research generation')
        stage_id+='-'+'-'.join(checked_id(str(fold)) for fold in fold_ids)
    if repair_batch_id is not None:
        if stage!='research-repair' or fold_ids is not None:
            raise ValueError('repair_batch_id is only allowed for a separate research repair stage')
        stage_id+='-'+checked_id(repair_batch_id)
    stage_dir=root/'cloud_stages'/stage_id
    files=safe_extract(payload_archive,expected_sha,root)
    stage_request={'archive_sha256':expected_sha,'files':files,'stage':stage,'config_sha256':object_hash(config),
                   'worker_sha256':expected_worker_hash,'checkpoint_helper_sha256':expected_checkpoint_hash,
                   'model_manifest_sha256':model_manifest['manifest_sha256']}
    lock_request(stage_dir/'request.json',stage_request)
    lock_request(stage_dir/'config.json',config)
    execution={'run_id':run_id,'stage':stage,'started_utc':datetime.now(timezone.utc).isoformat(),
               'status':'running','worker_sha256':expected_worker_hash,'checkpoint_helper_sha256':expected_checkpoint_hash,
               'model_manifest_sha256':model_manifest['manifest_sha256']}
    write_json(stage_dir/'execution.json',execution)
    runs.commit()
    try:
        watchdog = 1080 if stage in ('smoke','smoke-resume') else (21300 if stage in ('research-train','research-generate','forecast-train') else 6900)
        _run_logged(['/opt/bam-venv/bin/python','-u','/opt/bam_training.py','--run-dir',str(root),
                     '--stage',stage,'--config',str(stage_dir/'config.json'),'--budget-seconds',str(watchdog-120)],
                    root,stage_dir/'execution.log',seconds=watchdog,commit=runs.commit)
        execution['status']='complete'
    except Exception as error:
        execution.update(status='failed',error=str(error))
    finally:
        execution['finished_utc']=datetime.now(timezone.utc).isoformat()
        write_json(stage_dir/'execution.json',execution)
        runs.commit()
    return execution


def artifact_manifest(root,include_adapters=False):
    root=Path(root)
    files={}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Unexpected symlink in run artifacts')
        if not path.is_file() or '.cache' in path.parts:
            continue
        relative=path.relative_to(root)
        if not include_adapters and ('adapter' in relative.parts or path.suffix=='.safetensors'):
            continue
        if not include_adapters and 'checkpoints' in relative.parts and path.suffix=='.pt':
            continue
        files[str(relative)]={'sha256':digest(path),'bytes':path.stat().st_size}
    return files


@app.function(image=fetch_image,volumes={'/runs':runs},cpu=2,memory=8192,
              timeout=1800,min_containers=0,max_containers=1,single_use_containers=True,retries=0)
def fetch_result(run_id,kind='training',include_adapters=False):
    runs.reload();checked_id(run_id)
    if kind not in ('training','generation'):
        raise ValueError('Unknown run kind')
    root=Path('/runs/bam')/kind/run_id
    if not root.is_dir():
        raise ValueError('Run does not exist')
    files=artifact_manifest(root,include_adapters)
    archive_root=Path('/runs/bam/exports')/kind/run_id;archive_root.mkdir(parents=True,exist_ok=True)
    archive=archive_root/(object_hash(files)+'.tar.gz')
    if not archive.exists():
        with tarfile.open(archive,'w:gz') as output:
            for relative in files:
                output.add(root/relative,arcname=relative,recursive=False)
        runs.commit()
    size=archive.stat().st_size
    return {'run_id':run_id,'kind':kind,'files':files,'archive_id':archive.name,
            'archive_sha256':digest(archive),'archive_bytes':size,
            'payload':archive.read_bytes() if size<=16*1024**2 else None,'chunk_bytes':16*1024**2}


@app.function(image=fetch_image,volumes={'/runs':runs},cpu=1,memory=1024,
              timeout=300,min_containers=0,max_containers=2,single_use_containers=True,retries=0)
def read_artifact_chunk(run_id,kind,archive_id,offset,limit=16*1024**2):
    runs.reload();checked_id(run_id)
    if kind not in ('training','generation') or not re.fullmatch(r'[a-f0-9]{64}\.tar\.gz',archive_id):
        raise ValueError('Invalid archive identifier')
    if not isinstance(offset,int) or offset<0 or not isinstance(limit,int) or not 1<=limit<=16*1024**2:
        raise ValueError('Invalid bounded byte range')
    path=Path('/runs/bam/exports')/kind/run_id/archive_id
    with path.open('rb') as stream:
        stream.seek(offset);content=stream.read(limit)
    return {'offset':offset,'payload':content,'sha256':hashlib.sha256(content).hexdigest(),'total_bytes':path.stat().st_size}


def deployed_functions(environment='main'):
    result={}
    for name in ('fetch_models','generate','generate_heavy','train','fetch_result','read_artifact_chunk'):
        function=modal.Function.from_name(APP_NAME,name,environment_name=environment)
        function.hydrate();result[name]=function
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-deployment',action='store_true')
    parser.add_argument('--environment',default='main')
    parser.add_argument('--fetch-models',type=Path)
    parser.add_argument('--local-output',type=Path)
    args=parser.parse_args()
    functions=deployed_functions(args.environment)
    if args.check_deployment:
        print(json.dumps({'app':APP_NAME,'functions':{k:v.object_id for k,v in functions.items()},'compute_started':False},indent=2))
    elif args.fetch_models:
        config=json.loads(args.fetch_models.read_text())
        specs=[checked_model(config),checked_model(config['teacher'])]
        result=functions['fetch_models'].remote(specs)
        if args.local_output:
            write_json(args.local_output,result)
        print(json.dumps(result,indent=2))
    else:
        parser.error('Choose --check-deployment or --fetch-models CONFIG')


if __name__=='__main__':
    main()
