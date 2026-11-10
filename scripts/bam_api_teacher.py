"""CPU-only TRAIN target boundary and immutable API payload preparation.

No provider SDK, credentials, network calls or driver integration. Backend
envelopes are transport-neutral contracts, not advertised provider API syntax.
"""
from copy import deepcopy
from datetime import date
import hashlib
import json
from pathlib import Path

import bam_research as research


PROVIDERS = {"openai", "anthropic"}
TARGET_STAGES = {"postmortem", "gold", "semantic_review", "concept_labels"}
FORBIDDEN_REVIEW_KEYS = {"training_only_postmortem", "normalized_postgame_sources", "home_win"}
ORIGINAL_COMPARISON_MANIFEST_SHA256 = 'f39ec6410ad9281ca7235d21d8a384ce8ffe090e5f7152c5786295f5cec8c133'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_immutable(path, content):
    path = Path(path)
    if path.exists():
        _require(path.read_bytes() == content, "Frozen API preparation changed: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode()


def load_train_registry(snapshot):
    """Read hash-verified game inputs only; never open dev/test outcome stores."""
    snapshot = Path(snapshot)
    if (snapshot / 'bam_manifest.json').exists():
        from bam_data import _verify_shared_files, read_snapshot_inputs, snapshot_manifest_path
        _verify_shared_files(snapshot)
        rows = read_snapshot_inputs(snapshot, splits=("train",))
        _require(len({row['game_id'] for row in rows}) == len(rows), "Duplicate shared game IDs")
        return {'snapshot_manifest_sha256': file_hash(snapshot_manifest_path(snapshot)),
                'games': {row['game_id']: {key: row[key] for key in ('game_id', 'game_date_et', 'cutoff_utc', 'split')}
                          for row in rows}}
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    expected = manifest['artifact_sha256']['inputs.jsonl']
    _require(file_hash(snapshot / 'inputs.jsonl') == expected, "Snapshot game inputs changed")
    rows = [json.loads(line) for line in (snapshot / 'inputs.jsonl').read_text().splitlines() if line.strip()]
    _require(len({row['game_id'] for row in rows}) == len(rows), "Duplicate snapshot game IDs")
    return {'snapshot_manifest_sha256':file_hash(snapshot / 'manifest.json'),
            'games':{row['game_id']:{key:row[key] for key in ('game_id','game_date_et','cutoff_utc','split')} for row in rows}}


def load_request_authorizations(path, expected_sha256):
    """Load an independently pinned manifest written by trusted request builders.

    Transport must not construct authorizations from incoming candidate bodies.
    The expected manifest digest comes from the preparation/protocol artifact.
    """
    _require(file_hash(path) == expected_sha256, "Trusted request authorization manifest changed")
    value = json.loads(Path(path).read_text())
    _require(value.get('schema_version') == 1 and isinstance(value.get('requests'),list), "Invalid request authorization manifest")
    result = {}
    fields = {'request_id','request_sha256','game_id','stage','source_hash'}
    for entry in value['requests']:
        _require(isinstance(entry,dict) and set(entry) == fields and entry['request_id'] not in result, "Malformed/duplicate request authorization")
        result[entry['request_id']] = deepcopy(entry)
    return result


def bind_trusted_training_context(registry, trusted_games, request_authorizations, *, coverage_audit_sha256=None):
    """Bind source identity from canonical loader outputs, including annotations.

    ``trusted_games`` must come from the caller's hash-validated snapshot/sidecar
    loader (e.g. Pipeline.game), never from API requests. This function validates
    their metadata and binds the actual source objects' hashes. The independently
    pinned authorization manifest binds the entire expected stage payload, so a
    caller cannot authorize a forged context merely by recomputing its own ID.
    """
    result = deepcopy(registry)
    source_hashes = result.setdefault('trusted_source_hashes',{})
    for game in trusted_games:
        research.validate_game(game)
        known = result['games'].get(game['game_id'])
        _require(known is not None and all(game[k] == known[k] for k in ('game_id','game_date_et','cutoff_utc','split')), "Trusted game metadata differs from snapshot registry")
        annotated = any('coverage_audit' in page for source in game['sources'] for page in source['pages'])
        _require(not annotated or (isinstance(coverage_audit_sha256,str) and len(coverage_audit_sha256) == 64), "Annotated source binding requires its validated sidecar hash")
        source_hashes[game['game_id']] = research.value_hash(game['sources'])
    result['coverage_audit_sha256'] = coverage_audit_sha256
    result['authorized_training_requests'] = deepcopy(request_authorizations)
    return result


def authorize_comparison_registry(registry, source, *, expected_source_manifest_sha256=ORIGINAL_COMPARISON_MANIFEST_SHA256):
    """Authorize ONLY byte-verified cases in an independently pinned manifest."""
    source = Path(source)
    _require(file_hash(source/'manifest.json') == expected_source_manifest_sha256, "Frozen comparison manifest identity differs")
    manifest = json.loads((source/'manifest.json').read_text())
    for name,digest in manifest['artifacts'].items():
        _require(file_hash(source/name) == digest, "Original fixed comparison changed")
    requests = [json.loads(line) for line in (source/'requests.jsonl').read_text().splitlines() if line.strip()]
    _require(len(requests) == 16 and len({r['request_id'] for r in requests}) == 16, "Exactly sixteen frozen comparison cases required")
    result = deepcopy(registry)
    result['authorized_comparison_requests'] = {r['request_id']:research.value_hash(r) for r in requests}
    result['comparison_manifest_sha256'] = expected_source_manifest_sha256
    return result


def api_spec(spec):
    allowed = {'provider','model','model_version','model_release_date','context_window_tokens','max_output_tokens'}
    _require(isinstance(spec,dict) and set(spec) <= allowed, "Unexpected API configuration fields; credentials are forbidden")
    _require(spec.get('provider') in PROVIDERS and isinstance(spec.get('model'),str) and spec['model'].strip(), "Explicit API provider/model required")
    if 'model_version' in spec:
        _require(isinstance(spec['model_version'],str) and spec['model_version'].strip(), "API model version must be explicit text")
    if spec.get('model_release_date'):
        date.fromisoformat(spec['model_release_date'])
    for key in ('context_window_tokens','max_output_tokens'):
        _require(type(spec.get(key)) is int and spec[key] > 0, "Explicit positive API token budget required")
    _require(spec['max_output_tokens'] < spec['context_window_tokens'], "API output budget must leave input space")
    return deepcopy(spec)


def api_identity(spec):
    spec = api_spec(spec)
    return spec['provider'] + ':' + spec['model'] + ('@' + spec['model_version'] if spec.get('model_version') else '')


def build_role_plan(student, prediction_critic, teacher, reviewer, *, minimum_eval_date):
    """Modern API targets do not change prediction-time checkpoint eligibility."""
    for model in (student,prediction_critic):
        _require(model.get('model_id') and model.get('model_revision') and model.get('model_revision_date'), "Pinned historical prediction model required")
        _require(date.fromisoformat(model['model_revision_date']) < date.fromisoformat(minimum_eval_date), "Prediction-time checkpoint is not earlier than evaluation")
    _require(api_identity(teacher) != api_identity(reviewer), "API generator and reviewer must be independent identities")
    return {'prediction_models':{'student':deepcopy(student),'critic':deepcopy(prediction_critic)},
            'training_targets':{'teacher':api_spec(teacher),'reviewer':api_spec(reviewer),'allowed_splits':['train'],
                                'allowed_stages':sorted(TARGET_STAGES),'semantic_review_target_stages':['gold']},
            'minimum_eval_date':minimum_eval_date,
            'temporal_scope':'Retrospective TRAIN-only teacher distillation; heldout student and critic retain strict historical checkpoints. This does not certify that modern teacher pretraining has no influence on TRAIN explanations.'}


def _contains_key(value, forbidden):
    if isinstance(value,dict):
        return bool(set(value) & forbidden) or any(_contains_key(v,forbidden) for v in value.values())
    if isinstance(value,list):
        return any(_contains_key(v,forbidden) for v in value)
    return False


def validate_training_request(request, registry, *, targets=None, purpose='training_targets'):
    """The trusted manifest, not a self-declared split, authorizes each game."""
    _require(request.get('request_id') == research.value_hash({k:v for k,v in request.items() if k != 'request_id'}), "Scientific request hash changed")
    known = registry['games'].get(request.get('game_id'))
    _require(known is not None and known['split'] == 'train' and request.get('split') == 'train', "API teacher is restricted to manifest-verified TRAIN games")
    messages = request.get('messages')
    _require(isinstance(messages,list) and len(messages) in (2,3) and messages[0].get('role') == 'system'
             and all(m.get('role') == 'user' for m in messages[1:]), "Only original task plus bounded repair message is permitted")
    payload = json.loads(messages[1]['content'])
    _require(payload['game']['game_id'] == known['game_id'] and payload['game']['game_date_et'] == known['game_date_et']
             and payload['game']['cutoff_utc'] == known['cutoff_utc'], "API game/cutoff differs from trusted snapshot")
    stage = request.get('stage')
    if purpose == 'actual_claim_comparison':
        _require(stage == 'actual_claim_review', "Comparison preparation accepts only frozen actual-claim requests")
        _require(registry.get('authorized_comparison_requests',{}).get(request['request_id']) == research.value_hash(request), "Request is not authorized by the pinned comparison manifest")
        _require(not _contains_key(payload,FORBIDDEN_REVIEW_KEYS), "Actual-claim comparison contains hindsight data")
        return
    _require(purpose == 'training_targets' and stage in TARGET_STAGES, "Prediction-input stages cannot use the training API backend")
    _require(request.get('source_hash') and request.get('parent_hashes'), "Training target source/parent lineage required")
    _require(registry.get('trusted_source_hashes',{}).get(request['game_id']) == request['source_hash'], "Request source identity differs from trusted canonical loader binding")
    authorization = registry.get('authorized_training_requests',{}).get(request['request_id'])
    _require(authorization is not None and authorization.get('request_sha256') == research.value_hash(request)
             and all(authorization.get(k) == request.get(k) for k in ('request_id','game_id','stage','source_hash')), "Stage payload is not authorized by the trusted request manifest")
    if stage in ('postmortem','gold','concept_labels'):
        _require(request.get('model_role') == 'gold_teacher', "API target generation requires its explicit training teacher role")
    if stage == 'postmortem':
        _require(set(payload) <= {'game','instruction','normalized_postgame_sources','output_schema'} and payload.get('normalized_postgame_sources'), "Only normalized postgame evidence is allowed")
        for source in payload['normalized_postgame_sources']:
            _require(set(source) <= {'source_id','publisher','url','sha256','archive_time','source_kind','pagination','pages','source_warnings','source_warnings_scope'}, "Unexpected raw postgame source fields")
            for page in source['pages']:
                _require(set(page) <= {'page','page_kind','section','label','flags','method','lines'}, "Unexpected normalized postgame page fields")
                _require(all(isinstance(line,dict) and set(line) == {'line','text'} for line in page['lines']), "Postgame model input must contain normalized line/text only")
    if stage == 'gold':
        modules = payload.get('forecasting_context',{}).get('researcher_outputs',[])
        _require(len(modules) == 3 and {m.get('module') for m in modules} == set(research.MODULES), "Gold must combine all three real researcher outputs")
        _require('training_only_postmortem' in payload, "Combined gold requires its separate TRAIN postmortem")
    if stage == 'concept_labels':
        modules = payload.get('forecasting_context',{}).get('researcher_outputs',[])
        _require(len(modules) == 3 and {m.get('module') for m in modules} == set(research.MODULES), "Concept labels must see all three real researcher outputs")
        _require('training_only_postmortem' in payload and payload.get('concept_definitions'), "Concept labels require the TRAIN postmortem and concept definitions")
        _require(isinstance(request.get('concept_pool_sha256'),str) and len(request['concept_pool_sha256']) == 64, "Concept labels must bind their concept pool")
    if stage == 'semantic_review':
        target_hash = request.get('target_artifact_hash')
        target = (targets or {}).get(target_hash)
        _require(target is not None and target.get('artifact_hash') == research.value_hash({k:v for k,v in target.items() if k != 'artifact_hash'}), "Gold review requires its intact target artifact")
        _require(target.get('status') in ('mechanically_valid','approved') and target.get('stage') == 'gold' and target.get('game_id') == known['game_id'] and target.get('split') == 'train'
                 and target.get('source_hash') == request['source_hash'], "API reviewer target must be matching TRAIN gold")
        _require('original_pregame_research_context' in payload, "API gold review requires the original pregame research context")
        _require(request.get('generator_model') == target.get('model') and request.get('reviewer_model') != target.get('model'), "Gold reviewer is not independent of its generator")
        _require(request['parent_hashes'][0] == target_hash, "Gold review target parent changed")
        for message in messages[1:]:
            _require(not _contains_key(json.loads(message['content']),FORBIDDEN_REVIEW_KEYS), "Gold review must remain outcome-blind")


def api_payload(request, spec, registry, *, targets=None, purpose='training_targets', input_token_estimate=None):
    """Prepare an envelope; sending requires a future credential/budget adapter."""
    spec = api_spec(spec)
    validate_training_request(request,registry,targets=targets,purpose=purpose)
    _require(request['max_tokens'] <= spec['max_output_tokens'], "API output budget would truncate the scientific request")
    if input_token_estimate is not None:
        _require(type(input_token_estimate) is int and input_token_estimate >= 0 and input_token_estimate + request['max_tokens'] <= spec['context_window_tokens'], "Complete API request exceeds configured context")
    if request['stage'] == 'semantic_review':
        _require(request['reviewer_model'] == api_identity(spec), "Configured API reviewer differs from request identity")
    return {'scientific_request_id':request['request_id'],'scientific_request_sha256':research.value_hash(request),
            'backend':spec,'messages':deepcopy(request['messages']),'max_output_tokens':request['max_tokens'],
            'response_contract':deepcopy(request.get('output_schema')),'purpose':purpose,
            'submission_status':'offline_prepared_no_provider_call',
            'context_check':'Estimate only; provider token accounting/reasoning reservation required before paid submission.'}


def freeze_comparison_bundle(source, destination, registry, *, expected_source_manifest_sha256=ORIGINAL_COMPARISON_MANIFEST_SHA256):
    """Copy the original16 requests byte-for-byte; labels stay outside payloads."""
    source,destination = Path(source),Path(destination)
    registry = authorize_comparison_registry(registry,source,expected_source_manifest_sha256=expected_source_manifest_sha256)
    manifest = json.loads((source/'manifest.json').read_text())
    for name,digest in manifest['artifacts'].items():
        _require(file_hash(source/name) == digest, "Original fixed comparison changed")
    raw = (source/'requests.jsonl').read_bytes()
    requests = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    _require(len(requests) == 16 and len({r['request_id'] for r in requests}) == 16, "Exactly the original16 unique requests required")
    for request in requests:
        validate_training_request(request,registry,purpose='actual_claim_comparison')
    output = {'schema_version':1,'purpose':'TRAIN-only provider comparison; no selected provider, no credentials or network calls',
        'request_count':16,'requests_sha256':hashlib.sha256(raw).hexdigest(),
        'source_manifest_sha256':file_hash(source/'manifest.json'),'snapshot_manifest_sha256':registry['snapshot_manifest_sha256'],
        'scientific_request_ids':[r['request_id'] for r in requests],
        'scoring_expected_sha256':file_hash(source/'expected.json'),
        'scoring_limit':'Frozen16 labels retained; independently identified header-control defect requires separate15-case sensitivity and reasoning audit. Do not silently relabel.',
        'provider_independent':True,'submission_status':'not_submitted'}
    _write_immutable(destination/'requests.jsonl',raw)
    _write_immutable(destination/'scoring'/'expected.json',(source/'expected.json').read_bytes())
    _write_immutable(destination/'manifest.json',_encoded(output))
    return output


def prepare_reference_gold_bundle(games, module_artifacts, postmortems, destination, registry):
    """Create requests only when all actual reference modules qualify.

    Missing/partial/rejected modules produce a pending preparation manifest,
    never fabricated gold tasks or approval. This helper does not generate.
    """
    _require(len(games) == 4 and len({g['game_id'] for g in games}) == 4, "Exactly four distinct reference games required")
    pending,requests = [],[]
    for game in games:
        known = registry['games'].get(game['game_id'])
        _require(known is not None and known['split'] == 'train' and game['split'] == 'train', "Gold reference games must be TRAIN")
        modules = module_artifacts.get(game['game_id'],[]); postmortem = postmortems.get(game['game_id'])
        if len(modules) != 3 or any(a.get('stage') != 'module' or a.get('status') != 'approved' for a in modules) or postmortem is None:
            pending.append(game['game_id']);continue
        _require(registry.get('trusted_source_hashes',{}).get(game['game_id']) == research.value_hash(game['sources']), "Reference source lacks canonical loader binding")
        for artifact in modules + [postmortem]:
            _require(artifact.get('artifact_hash') == research.value_hash({k:v for k,v in artifact.items() if k != 'artifact_hash'}), "Reference parent artifact changed")
        request = research.build_gold_request(game,modules,postmortem)
        requests.append(request)
    status = {'status':'awaiting_qualified_modules_or_postmortems' if pending else 'ready_for_offline_API_preparation',
              'game_ids':[g['game_id'] for g in games],'pending_game_ids':pending,
              'qualified_request_count':len(requests),'snapshot_manifest_sha256':registry['snapshot_manifest_sha256'],
              'submission_status':'not_submitted'}
    destination=Path(destination)
    if not pending:
        _write_immutable(destination/'requests.jsonl',('\n'.join(json.dumps(r,ensure_ascii=False) for r in requests)+'\n').encode())
        authorizations = {'schema_version':1,'requests':[{key:r[key] for key in ('request_id','game_id','stage','source_hash')} | {'request_sha256':research.value_hash(r)} for r in requests]}
        _write_immutable(destination/'request_authorizations.json',_encoded(authorizations))
        status['request_authorizations_sha256'] = file_hash(destination/'request_authorizations.json')
    _write_immutable(destination/('pending.json' if pending else 'manifest.json'),_encoded(status))
    return status
