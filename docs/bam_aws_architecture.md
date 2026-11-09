# AWS research and forecasting architecture

Design for review, 2026-11-07. The user requested the complete architecture
before execution and specified that all experiment compute must run on AWS.
No instances, buckets, model requests, uploads or training jobs have been
started by this design. The cap is $200 within the existing accounting scope.
The immediate milestone is a TRAIN-only 16–32-game proof of learning; DEV is
deferred and TEST is excluded. This supersedes Modal/direct-provider execution
plans for new work and complements `docs/bam_training_first.md`.

**2026-11-08 continuation:** the user subsequently authorized resolving account
prerequisites and implementing this plan autonomously, permitted any suitable
AWS region, and prioritized efficiency. An IAM-only stack is now
deployed; GPU quota and Bedrock authorization still block paid
execution. A standalone TRAIN worker, reviewed-gold input contract and read-only
funding preflight are implemented with synthetic tests. No instance, artifact
bucket, model invocation or real learning run has been launched. See
[`aws_toolkit_setup.md`](aws_toolkit_setup.md) for live evidence and remaining
orchestration work. Earlier unimplemented/unverified statements below describe
the design's original state, not the latest account check.

## Components and ownership

| Component | Proposed implementation | What is trained |
| --- | --- | --- |
| Statistical baseline | 252-feature gradient boosting (`run_gradient_boosting.py`): TRAIN games forecast by models fitted on earlier TRAIN seasons, DEV/TEST by the TRAIN fit | Booster fitted on earlier games; fixed during LLM optimization |
| Pregame researcher | Frozen, pinned Qwen2.5-14B-Instruct on EC2; three analytical roles sharing one checkpoint | No task-specific weight updates |
| Postmortem and gold teacher | Strong reasoning-capable Claude model through Amazon Bedrock; Sonnet 4.6 is an initial qualification candidate | No weight updates |
| Gold reviewer | A different Bedrock model, with Opus 4.6 an initial candidate; blinded to outcome/postmortem | No weight updates |
| Forecaster | Separate Qwen2.5-14B-Instruct instance with rank-8 LoRA and a learned scalar residual head on EC2 | LoRA adapters plus probability head |

Teacher/reviewer names are candidates, not claims of qualified quality, account
access, regional availability or lowest cost. Configure reasoning explicitly
for the selected model's API and count reasoning in the output-token budget.
Bedrock calls use AWS authorization; this design does not require direct OpenAI
or Anthropic API credentials. Researcher and forecaster share base checkpoint
identity, but research generation must never load forecasting adapters.

**Efficiency update:** qualify AWS-sold `openai.gpt-oss-120b-1:0` and
`deepseek.v3.2` first, with one generating and the other independently reviewing.
Choose the assignment by cost per accepted, grounded TRAIN report, counting
postmortem, review, reasoning and bounded repairs. Start with the cheaper
gpt-oss teacher candidate; neither model has passed project quality checks or
account authorization yet. Use a small common TRAIN qualification subset before
generating the full pilot. Keep all pregame evidence, explicit output bounds and
one GPU worker; do not save cost through silent source truncation. Avoid setting
up batch infrastructure for the tiny pilot until measured savings justify it.
Current on-demand N. Virginia prices and access evidence are recorded in
`reports/aws_setup/20261108/teacher_us_east_1_prices.json` and the setup notes.

## Data and baseline

Use a small selection from the already collected 2023–24 TRAIN games with
approved pregame notes and richer box-score history. Reconstruct the boosting
probabilities before using these examples: the completed final statistical
model was fitted with later games and is not a valid source of TRAIN forecasts.
Select a representative 16–32-game subset spanning the available publishers,
teams, packet lengths and pregame probabilities. Produce reviewed gold for
every selected pilot game. Do not choose games for easy-to-explain outcomes.

For each game, preserve full source bytes, source/version timestamps, hashes,
page/section identities, extracted text and separately decoded tables. The
prediction cutoff is scheduled tipoff minus 60 minutes. Every pregame source
must satisfy that cutoff. Prior-game features and each booster fitting pool
must strictly precede the target; imputation is fitted only on that pool.
Normalize only allowed numeric historical fields from box scores. Raw ESPN
responses, current news and target-game results never become prediction inputs.

Keep the current selected baseline settings: 100 depth-1 trees, learning rate
0.05, minimum leaf 20, log loss, full subsampling, seed 230, no early stopping
and no probability calibration. No further baseline tuning on DEV in this phase.
Build expanding chronological folds within TRAIN; no gradients from the LLM
update the booster. Later DEV uses a booster fitted on TRAIN and held fixed.

Beyond the pilot, the frozen snapshot `bam_notes_v2` has 1,456 forecast-eligible
TRAIN games with notes and the full 252-feature history (2013–14 to 2023–24).

## Prediction-time path

1. Earlier-game statistics produce a boosting probability `p_boost`.
2. The frozen researcher reads every page/section of the pregame packet. Chunk
   large sources and account for every chunk; never silently truncate a PDF or
   a training sequence. Code decodes supported table formats and computes
   derived quantities. Flag unreadable pages and unsupported table layouts.
3. One structured research brief collects three analytical roles:
   availability/rotation; matchup strengths and weaknesses; recent changes,
   rest and workload. Each finding includes a factual basis, citation, basketball
   interpretation, counterevidence, time scope and uncertainty. Record whether
   the signal appears already represented by the booster. Avoid inventing
   percentage-point probability effects or treating past absence as current status.
4. Validate citations and arithmetic and qualify semantic quality on TRAIN.
   Unresolved assertions remain in the audit, not as asserted forecasting facts.
   Pregame research is produced and stored before postmortem/gold generation;
   hindsight never edits the ordinary brief for that game. Improvements to a
   frozen research prompt must come from TRAIN and create versioned regeneration.
5. The forecaster receives matchup metadata, `p_boost`, the same pregame
   statistical features and the compact cited research brief. It does not receive
   a postmortem, gold text, target outcome or raw source response. Its transformer
   forms a numerical representation `h` of that pregame input.
6. A zero-initialized linear head computes `delta = w^T h + b` and returns
   `sigmoid(logit(p_boost) + delta)`. Thus the initial prediction exactly matches
   boosting. Changes are learned from outcomes, not manually assigned to claims.

The forecaster can also generate a readable explanation through its normal
language-model output. The probability head reads the pregame representation;
it does not wait for this generated explanation. An explanation is not proof
of which internal features caused an adjustment. Head-only fitting and a
separately fixed embedding are not the intended joint model.

## TRAIN-only gold construction

There are three distinct documents:

| Document | Inputs | Use |
| --- | --- | --- |
| Ordinary research brief | Pregame notes, tables and known statistical context | Actual forecaster input |
| Postmortem | Normalized TRAIN outcome, box score and sourced postgame accounts | Private teacher context only |
| Reviewed gold report | Ordinary brief, pregame evidence and teacher's postmortem context | Forecaster explanation target only |

The teacher examines how the game unfolded, distinguishing foreseeable signals
from realized luck and unsupported causal stories. It then improves the analysis
of the original researcher's output as if still at the pregame cutoff. Hindsight
may guide attention, but every resulting factual claim must be supported by
pregame evidence, with qualifications and opposing evidence retained.

A different model reviews the gold against the pregame sources without the
postmortem or outcome. Check entity, season, date, sample window, table headers,
arithmetic, current availability and unsupported causal claims. Keep every
rejection and bounded repair attempt. A literal citation match alone is not
semantic verification, and model review is not external fact-checking.

Modern Bedrock teacher/reviewer models are restricted to TRAIN. Before future
DEV use, the frozen research and semantic-review route must independently meet
the agreed pre-DEV knowledge-cutoff criterion. Do not reuse the modern teacher
as a hidden DEV researcher, reviewer or repair service. No TEST work is included.

## Training objective and proof of learning

Keep pretrained base weights frozen in 4-bit NF4 form and train small LoRA
adapters plus the scalar head. This is parameter-efficient fine-tuning, not
training a 14B model from scratch. The existing pinned implementation is the
starting point; AWS hardware compatibility and actual sequence memory are unproven.

For N outcome examples and M accepted gold examples:

    loss = mean_N(Brier(predicted_probability, outcome))
         + lambda * mean_M(per-example gold completion cross-entropy)

The two means are normalized independently. In the pilot, N = M = the selected
16–32 games. For a larger run, all eligible outcomes contribute even if only a
subset has accepted gold. No fixed 200-report limit is adopted; expand accepted
gold coverage according to measured quality and cost after the pilot.

Use separate forward/backward passes for (a) probability from the pregame prefix
and (b) completion supervision against gold, then one optimizer update. The
first pass cannot see teacher-forced gold tokens. Brier updates the head and
adapters; explanation cross-entropy updates the adapters through the language
model. Log both objectives independently and keep lambda fixed for comparisons.

First check data/label alignment, zero-head equivalence to boosting, finite
gradients and updates in the appropriate parameters. Then deliberately fit the
small sample closely. A useful proposed diagnostic is at least a 50% relative
reduction in inference-mode TRAIN Brier, alongside a clear decline in gold
completion loss; pin actual criteria and a maximum spend/runtime before fitting.
This threshold is an engineering diagnostic, not a statistical finding about
generalization or the value of text. Checkpoint/resume must preserve optimizer
and random state. Failure to learn calls for diagnosis before more data or DEV.

The production trainer presently loads DEV automatically and requires gold for
every training example. Before execution, implement the TRAIN-only entry point,
sparse-gold normalization, training metrics and isolation checks. No current
`--stage all` invocation is compatible with this phase.

## AWS execution layout

- **S3:** immutable raw sources and manifests; separate pregame inputs,
  training-only outcomes/postmortems/gold, model checkpoints and reports.
  The pilot job's role is limited to its TRAIN artifacts. Do not upload/mount
  DEV or TEST data into the pilot workspace.
- **Small EC2 CPU controller:** source processing, numerical table decoding,
  chronological boosting, Bedrock requests, run manifests and the single-writer
  admission ledger. Serialize reservations and save durable job/request IDs;
  do not turn a copied S3 JSON ledger into an unsafe multi-writer ledger.
- **Amazon Bedrock:** bounded teacher and independent review calls. Cache exact
  source/prompt/model-bound outputs and retain uncertain charges. This is the
  AWS route for managed model inference; no direct external-provider calls.
- **Ephemeral EC2 GPU worker:** frozen-research generation and QLoRA training in
  separate jobs sharing cached base weights. Initial hardware candidate is
  `g6e.2xlarge` (one L40S, nominal 48 GB GPU memory, 64 GiB host memory). Verify
  actual memory and throughput with complete representative sequences before
  accepting the full fit. Modal H100 timings do not establish AWS L40S timings.
- **S3 checkpoints and CloudWatch logs:** save gradients/loss summaries, model
  and data hashes, checkpoint identities, token usage and resource-runtime records.
  Terminate GPU instances at job completion or their bounded maximum runtime;
  account for remaining storage and CPU costs after compute stops.

AWS region, identity/role, Bedrock model access, EC2 GPU quota, exact model IDs
and current unit prices have not been checked in the user's account. There is
no live AWS orchestration implementation yet. All upcoming experiment processing,
including source preparation and booster fitting, is intended to run on AWS;
local repository edits and architecture review are not experiment runs.

## Budget allocation for review

These are proposed allocations, not cost estimates or guarantees of completion:

| Allocation | USD |
| --- | ---: |
| Existing tracked estimates and uncertain reservation | 8.59 |
| Initial TRAIN proof: research, reviewed gold and GPU learning diagnostic | 40.00 |
| Expanded TRAIN research and accepted gold, conditional on pilot | 70.00 |
| Full TRAIN fitting and later DEV work, conditional on pilot | 55.00 |
| CPU, storage, transfer, logging and contingency, including existing $5 buffer | 26.41 |
| Total | 200.00 |

Reallocate within the cap after measuring the pilot. Reserve bounded input,
reasoning/output tokens, retries and maximum instance runtimes using refreshed
AWS prices and margins before any admission. No credits, cached-weight transfer,
Spot discounts or tax treatment are assumed. Historical project costs outside
the existing ledger remain unknown. AWS Budgets alerts are supplementary: they
are delayed and do not establish a real-time hard spending cap. Use application
admission controls plus timed resource termination, preserving a reserve for
billing uncertainty and infrastructure charges. A $200 allowance does not prove
that every expanded note can be researched and the model fully trained.

## Pilot deliverables and next decision

Provide the selected TRAIN manifest; complete ordinary research briefs; accepted
gold, citations and review decisions; exact forecaster inputs; training Brier,
explanation and total-loss curves; model/adapter/head checkpoints; and an AWS
usage/cost reconciliation with measured larger-run projections. Report failed
cases and how much work completed, rather than treating spending as success.

After a working pilot, expand TRAIN data and gold coverage within the measured
budget. Only after full-TRAIN learning is understood, compare boosting and the
hybrid on identical DEV games and choose by DEV Brier. TEST remains excluded.

## Verified service references

- [Bedrock extended thinking and billed token accounting](https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-extended-thinking.html)
- [EC2 accelerated-instance specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)
- [AWS G6e instance specifications](https://aws.amazon.com/ec2/instance-types/g6e/)
- [AWS Budgets update delays](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html)

Service capabilities checked 2026-11-07; no account-specific access or price
quote is implied by these public documents.
