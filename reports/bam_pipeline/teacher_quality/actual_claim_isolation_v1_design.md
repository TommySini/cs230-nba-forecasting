# Actual claim isolation diagnostic, frozen before inference

Cases: `data/processed/bam_quality/actual_claim_isolation_v1`. Manifest SHA256: `f39ec6410ad9281ca7235d21d8a384ce8ffe090e5f7152c5786295f5cec8c133`.

Sixteen original training-game findings are copied without editing their IDs, dates, facts, interpretations, caveats or citations. Eight invalid cases are the four Detroit totals-as-averages claims, Cunningham's thresholds-as-exact-averages claim, two wrong Cleveland milestone years and Cleveland's wrong defensive rank. Eight positive controls comprise four canonical table findings and four actual prose findings: Miami's current back-to-back, Herro's correctly separated 52/57 streaks, Mobley's explicitly published season averages, and the historical Denver nine-of-ten record against Detroit with an explicit nonprediction caveat.

Positive source checks: Miami named player headings and column rows identify Adebayo and Herro; Philadelphia's current-season table gives its published team rates; Cunningham's current-season card gives the exact rates. The prose controls retain complete sentence lines and distinguish a historical record from a forecast. These were selected for source diagnosis, not at random; all favorable and unfavorable labels were saved in `expected.json` before any inference request was launched.

Requests contain one actual finding, its original artifact provenance, complete cited source pages, and source metadata. The wrong-year cases additionally include Cleveland's full schedule page 37, and the Cunningham threshold case includes page one with the explicit current averages. Other unverified generated findings and expected labels are absent. The production compact-review rubric is retained with the whole-report task removed. This diagnostic cannot certify omissions or whole-report coverage.

The exact reference-v4 Qwen3 model revision and generation configuration are copied into `launch_payload.json`. Actual local tokenization gives a maximum input of 11,489 tokens, plus 4,500 answer and 4,096 reasoning tokens, below the 32,768 context limit. No context is truncated. Because claim isolation also supplies full relevant pages, any improvement cannot be attributed solely to fewer simultaneous decisions.

The separate script `scripts/bam_actual_claim_probe.py` freezes artifacts, checks returned original IDs and canonical visible source references, materializes exact source text, and reports both model and citation-containment verdicts. Unsupported findings cannot be promoted by the containment check. It retains every raw response and reports malformed or missing coverage. Seven focused tests pass. Frozen production and benchmark files are unchanged.

The directory `actual_claim_isolation_unlaunched_preflight_draft` is a superseded, never-launched preparation artifact. It is not the frozen inference set. Only `actual_claim_isolation_v1` should be launched or scored.

Parent launch, when authorized and scheduled in the available generation slot:

```python
import json, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
import bam_cloud
p = json.loads(Path('data/processed/bam_quality/actual_claim_isolation_v1/launch_payload.json').read_text())
call = bam_cloud.deployed_functions()['generate'].spawn(
    p['requests'], p['model_spec'], p['run_id'], p['config'], p['expected_worker_hash'])
print(call.object_id)
```

Scoring after retrieval:

```sh
.venv/bin/python scripts/bam_actual_claim_probe.py score --directory data/processed/bam_quality/actual_claim_isolation_v1 --result RESULT.json --output SCORE.json
```
