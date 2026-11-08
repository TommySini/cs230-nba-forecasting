# AWS Agent Toolkit setup

## New email: EC2 three-region limit and CPU-first preparation

The user supplied an AWS new-customer email saying EC2 launches are temporarily
limited to **three regions**. The email does not identify this account's allowed
three. Earlier checks of 17 `opt-in-not-required` regions were quota reads;
they do not prove EC2 launch authorization in those regions. Do not propose
more region scanning as a way around this account restriction. The user also
pasted a denial without a case number; do not silently assign it to the new
N. Virginia request.

Useful CPU-first work is TRAIN source normalization, timestamp/leakage audits,
prior-only statistical features and the chronological boosting baseline. This
does not require Bedrock or a GPU. A bounded CPU job can be prepared once the
actual Standard-instance quota, permitted region, scoped runtime, funding and
external termination are verified. CPU use has no documented guaranteed spend
threshold or timetable for unlocking G-family quota. Do not run idle machines
to build usage history. The intended 14B CUDA joint-learning worker still needs
GPU access; a CPU smoke test would not establish the intended learning result.

AWS's G6e specification gives an alternative for the existing quota discussion:
`g6e.xlarge` needs **4 G-family vCPUs**, with one 48 GB GPU and 32 GiB system RAM;
`g6e.2xlarge` needs 8, with the same GPU and 64 GiB system RAM. Ask about the
smaller initial allocation if 8 cannot be approved; do not claim its 32 GiB
loading/throughput is qualified. The current gate remains pinned to the original
8-vCPU instance until sizing, prices and runtime are validated.

Fresh live checks failed: AWS MCP search/retrieve tools returned `Unknown tool`
even after retry. The CLI fallback (2.39.6) returned `Your session has expired.
Please reauthenticate using 'aws login'.` for profile cs230. No current CPU quota,
GPU request update or allowed-region list was obtained. Preserve the toolkit
installation and Codex-only profile binding; sign-in renewal is needed before
account work resumes. No new cloud request or external message was sent.
Evidence: `reports/aws_setup/20261108/cpu_first_region_email.json`.

References: [EC2 family-specific vCPU quotas](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-on-demand-instances.html),
[G6e specifications](https://aws.amazon.com/ec2/instance-types/g6e/).

## Concrete inference check: 2026-11-08 05:11 UTC

After the user changed payment method, a single minimal gpt-oss-120b Converse
access probe in us-east-1 was admitted with **one cent** reserved in the original
ledger, 4096 conservative input tokens and a hard 16-token output maximum at
standard-tier prices. The current earned credit check remained $100 and
GetModelInvocationLoggingConfiguration returned `{}`. No NBA data was used.
AWS rejected the attempt with `ValidationException: Operation not allowed`.
The MCP script's exception handler also raised `NameError: ClientError is not
defined`; its stderr and api_calls nevertheless retain the original provider
error. No repeat invocation was attempted and no usage was returned.

The one-cent reservation is retained as `uncertain` pending billing evidence:
completed estimates $6.77, outstanding $1.83, buffer $5, admission availability
$186.40. This is not a claim of a one-cent charge. Existing accounting was
preserved and one attempt appended. See `probe_admission.json`,
`bedrock_access_probe.json` and `probe_budget_state.json` in the evidence folder.
The unsent [authorization inquiry](aws_bedrock_access_followup.md) now includes
the concrete error. Account authorization/internal verification is not a
confirmed root cause; AWS must clarify if it persists after activation settles.
No training, source processing, gold generation or GPU launch has run.

## Region flexibility, payment fix and efficiency: 2026-11-08 05:07 UTC

The user now permits **any suitable AWS region** and prioritizes efficiency.
The existing installation/profile default and Oregon IAM stack were preserved;
no worker was deployed elsewhere. Preflight and worker receipts now bind an
explicit commercial region instead of requiring Oregon. Every regional quota,
role-policy verification, model availability and quote must match that plan.
32 focused synthetic tests passed (17 preflight, 15 worker); instance size,
8-vCPU threshold, funding, deadlines and TRAIN-only boundaries are unchanged.
Old preflight reports without `binding.region` must be regenerated.

### GPU search and new request

Direct GetServiceQuota checks covered all **17 enabled commercial regions**
returned by DescribeRegions. On-Demand G/VT and G/VT Spot quotas are **0 in every
one**. The other 17 listed regions were not opted in; none was enabled by us.
The initial broad EC2 ListServiceQuotas scan timed out and is not evidence of
complete quota coverage; the successful direct checks establish the G results.

The SageMaker `ml.g6e.2xlarge for training job usage` quota was also checked in
all 17 enabled regions: nine returned 0, and eight returned
`NoSuchResourceException: The request failed because the specified quota and
service do not exist.` That is not a claim about every SageMaker GPU type in
those regions. A fully paginated us-east-1 scan found all 206 distinct returned
G/P training-or-Spot quota records at 0. No available training route was found.

N. Virginia lists `g6e.2xlarge` offerings in four availability zones; catalog
offering does not establish current capacity. A new us-east-1 request for
**8 On-Demand G/VT vCPUs** was submitted, ID
`8405145fff6c4426a52d9e373c4fac98BKDvHupa`. It is `CASE_OPENED`, support case
**179411421300610**. Applied quota is not increased by a pending request.
The Oregon denial and its unsent appeal remain separate. Do not submit repeated
requests across more regions as a substitute for this review.

### Payment change and teacher access

The user changed the payment method. We retried the same previously reviewed
Sonnet/Opus public agreements. Both now report agreement `AVAILABLE`: the
`INVALID_PAYMENT_INSTRUMENT` error has cleared. Both still report
`authorizationStatus=NOT_AUTHORIZED`, with entitlement and region `AVAILABLE`.
AWS-sold DeepSeek V3.2, gpt-oss-120b and Nova Pro also reported NOT_AUTHORIZED in
us-east-1; this is not a Marketplace payment error for those models. No
inference was invoked. Additional account-level access is a possibility in
AWS's documentation, not a confirmed internal reason for this account's state.
Allow activation to settle and recheck; if it persists, use the unsent
[Bedrock authorization inquiry](aws_bedrock_access_followup.md). Do not ask the
user to change the payment method again based on the old error.

Marketplace can be avoided by AWS-sold Bedrock models, including DeepSeek,
Qwen and OpenAI gpt-oss. Sonnet 4.6 is Marketplace-billed and its model card
does not support `bedrock-mantle`; the general Anthropic first-use-form
exception is not a payment-validation workaround. Its model card also says
charges appear under the provider rather than Amazon Bedrock: the supplied
generic Bedrock credit entries alone do not prove Anthropic-charge coverage.

For initial quality qualification, prefer gpt-oss-120b and DeepSeek V3.2 as a
distinct teacher/reviewer pair. Live Price List quotes in us-east-1, standard
on-demand `bedrock-runtime`, USD per million tokens:

| Model | Input | Output |
| --- | ---: | ---: |
| gpt-oss-120b | 0.15 | 0.60 |
| DeepSeek V3.2 | 0.62 | 1.85 |

These prices are not a full pilot cost reservation. Optimize cost per accepted,
grounded report including postmortem, reasoning, review and bounded repair.
The cheaper model is not yet quality-qualified. Its Apache 2.0 and DeepSeek's
MIT model licenses do not contain Anthropic's competing-model restriction;
AWS service terms still apply. Preserve full evidence and the outcome-blind
review. No live IAM policy was expanded to these candidates yet.

Evidence is under `reports/aws_setup/20261108/`: `g_on_demand_all_enabled_regions.json`,
`gpu_alternatives_all_enabled_regions.json`, `sagemaker_us_east_1_training_quotas.json`,
`us_east_1_quota_request.json`, `bedrock_agreement_retry.json`,
`access_followup_0507.json`, and `teacher_us_east_1_prices.json`.

Sources: [AWS model access](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html),
[Sonnet card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-4-6.html),
[gpt-oss card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-oss-120b.html),
[DeepSeek card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-deepseek-deepseek-v3-2.html),
[AWS service terms](https://aws.amazon.com/service-terms/).

## Credit details and access recheck: 2026-11-08 04:55 UTC

The user supplied the credit's Applicable products list and reported expiration
in **late 2027**. Relevant entries include Amazon Elastic Compute Cloud,
Amazon Bedrock, Amazon Bedrock Service, AmazonBedrockFoundationModels,
Amazon Simple Storage Service, AmazonCloudWatch, AWS Data Transfer and
Amazon SageMaker. This supplies the requested service-eligibility evidence;
expiry is beyond the upcoming pilot. No browser access is needed to repeat
this check. The exact date was not supplied and has not been invented.
See `reports/aws_setup/20261108/user_credit_details.json` for provenance.

The live recheck in `followup_0455.json` still reports **$100** remaining
account-plan credits, Oregon On-Demand G/VT `Quota.Value=0`, and both Claude
agreements `ERROR` with `INVALID_PAYMENT_INSTRUMENT` and `NOT_AUTHORIZED`.
The product list does not establish another $100 of earned credit.

Marketplace handles the third-party Claude agreement used by Bedrock. It is
not a separate GPU requirement. Credit eligibility and model activation are
separate: AWS requires an accepted payment method for Marketplace activation.
The user believes a valid method is already present; the error does not prove
that no card was entered. Payment settings have not been inspected, so the
specific reason for rejection is unresolved. Check Billing > Payment
preferences for the method used for Marketplace, then account/billing support
if it is already correctly configured. Do not alter payment information.

Quota zero blocks the planned Oregon On-Demand G-family worker. This is not
an all-region/all-service GPU finding. Other regions, Spot and SageMaker have
not been established as usable alternatives. Reopening the existing case with
the prepared appeal is the next route for this worker; the draft remains unsent.
Teacher hindsight design is approved as described below.

## Follow-up: 2026-11-08 04:46 UTC

This update supersedes pending statuses below. The user supplied an AWS email
declining GPU case `179411216100407` and inviting a detailed-use-case appeal.
The API still reports `Status=CASE_OPENED` with effective quota 0; that summary
does not overturn the denial email. An [appeal draft](aws_gpu_quota_appeal.md)
is prepared but **not sent**. The Support API remains unavailable under the
account's support subscription; use the console to reopen/respond.

Both Claude models now return `agreementAvailability.status=ERROR`,
`errorMessage=INVALID_PAYMENT_INSTRUMENT:A valid payment instrument must be provided.`,
and `authorizationStatus=NOT_AUTHORIZED`. Entitlement and regional availability
remain AVAILABLE. The FTU form submission succeeded; AWS Marketplace payment
validation is the next confirmed blocker. The user must inspect/verify the
payment method in Billing; no payment information was accessed or changed.

Sonnet/Opus were candidates, not scientific requirements. A read-only check of
DeepSeek V3.2 and Qwen3-235B found `agreementAvailability.status=AVAILABLE` but
`authorizationStatus=NOT_AUTHORIZED`; catalog presence does not establish an
immediately usable alternative. Their authorization cause has not been verified.
No models were switched, no extra agreements accepted, and no inference called.

Anthropic's Bedrock terms restrict building competing products, including
training competing AI models. It is not established that this narrow academic
predictor is covered by that restriction, but model activation is not explicit
permission for output-based training. Qualify a teacher with clear applicable
rights or obtain clarification before producing a training corpus. See
[the applicable Anthropic terms](https://aws.amazon.com/legal/bedrock/third-party-models/).

The user's [credit documentation](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/useconsolidatedbilling-credits.html)
confirms that selecting the credit in Billing > Credits reveals **Expiration date**
and **Applicable products**. It does not expose this account's values. Browser
access is unnecessary if the user copies those fields; no additional browser
permissions were requested. Earned account-plan balance remains $100.

Teacher design reconfirmed: TRAIN-only outcomes, normalized box scores and
sourced postgame context belong in the private postmortem used to guide gold.
Gold may emphasize foreseeable mechanisms but must retain pregame support and
uncertainty. A reviewer sees pregame evidence and gold without the postmortem;
the actual forecaster input remains the frozen ordinary pregame brief. Outcomes
supervise Brier separately; gold supervises explanation completion separately.

## Continuation: 2026-11-08 04:38 UTC

The integrated AWS MCP tools are loaded and were used successfully in this
session. STS verified the same root-backed `cs230` account. The existing CLI,
MCP installation and Codex-only binding were preserved. Parallel coding agents
were explicitly prohibited from using AWS credentials or account tools.

Live evidence is saved in [`reports/aws_setup/20261108/`](../reports/aws_setup/20261108/).
This section supersedes the installation-time prerequisites below.

### Credits

`GetAccountPlanState` still reports `PAID`, `ACTIVE`, and **$100** remaining.
`ListAccountActivities` returns five activities, each `NOT_STARTED` with a $20
reward: EC2, Bedrock playground, cost budget, Lambda web app, and RDS/Aurora.
Their additional $100 is potential credit, not earned balance. The budget
activity's `GetAccountActivity` record gives an activity deadline of
`2027-05-08T03:52:16Z`; this is **not** the credit expiration date.

Other promotional credits, credit expiration, and account-specific EC2/Anthropic
Bedrock eligibility remain unverified. The browser host was unavailable and
native Chrome inspection timed out. The user was asked to copy the credit
expiration and eligible-service details from Billing > Credits. No paid
inference, instance, storage, or training job was launched. Do not count an
unearned reward or the $200 project allowance as authorized cash spending.

The original ledger is unchanged: cap $200, completed estimates $6.77,
outstanding reservation $1.82, buffer $5, admission availability $186.41.
These figures are project estimates, not AWS credits or invoice totals.

### Scoped runtime access

CloudFormation stack **`cs230-train-pilot-20261108`** reached `CREATE_COMPLETE`
in Oregon. [`infra/aws/train-pilot-iam.json`](../infra/aws/train-pilot-iam.json)
creates only three EC2 roles and three instance profiles, under `/cs230/`:

- `cs230-train-pilot-20261108-controller`: TRAIN preparation/targets and exact
  Sonnet/Opus US inference profiles; no Marketplace subscription permission.
- `cs230-train-pilot-20261108-researcher`: code, pregame sources and base weights;
  writes ordinary research; cannot read outcomes or gold.
- `cs230-train-pilot-20261108-forecaster`: TRAIN pregame/research/targets/weights;
  writes checkpoints and metrics.

All artifact access is restricted to the future bucket
`cs230-nba-<aws-account-id>-us-west-2`, prefix `train/pilot-20261108` and per-role
subpaths. The bucket and log groups have not been created by this work. Roles
trust EC2, not the local root session, and cannot create IAM or compute resources.
No static credentials were created or placed on an instance. Local account
setup still uses the approved root-backed profile; no operator role or paid
launcher has been established.

Read-back of the deployed template and every inline/attached role policy
matched the reviewed template. All three roles have one inline policy and no
attached managed policies. IAM simulation checked researcher pregame access,
denial of target access, forecaster target access and explicit TEST-path denial,
and denied controller IAM/compute creation. These are policy simulations, not
an EC2 runtime test. Instance-profile outputs and policies are in the evidence.

Validation: `cfn-lint==1.60.0`, project-specific cfn-guard rules (Guard 3.2.1),
AWS `ValidateTemplate`, and deployment validation passed. The validator was
installed in the repository `.venv`; Guard's official release SHA256 was checked.
The Guard rules cover this IAM-only stack, not a general compliance audit.
`DescribeTrails` in Oregon, including shadow trails, returned `trailList=[]`;
durable CloudTrail delivery was not verified or configured. Event history and
MCP audit records provide less retention than a durable trail.

### Oregon GPU quota

Requested **8 vCPUs** for EC2 `L-DB2E81BA` (On-Demand G/VT) in `us-west-2`.
Request: `294524f6ec704dccb50fb60667bf33b7RqeLYsJa`.
Current request status: **`CASE_OPENED`**, support case **`179411216100407`**.
Applied quota remains **0**; a submitted request does not authorize a launch.

`support:DescribeCases` failed with `SubscriptionRequiredException`:
"Amazon Web Services Premium Support Subscription is required to use this service."
This prevents reading the case conversation through the Support API. Do not
upgrade support; inspect the existing case in the Support console if AWS asks
for more information. No other-region GPU capacity claim is made.

### Claude access: identified and submitted missing steps

The original failure was diagnosed with `GetUseCaseForModelAccess`, which
returned `ResourceNotFoundException`: "You have not filled out the request form.
Fill out the form before getting access."

Submitted the Anthropic first-time-use form with the project URL, internal
academic use, and an explicit description of reviewed reports supervising
Qwen forecasting adapters. Then created the public usage-based agreements for
Sonnet 4.6 and Opus 4.6. No provisioned/reserved throughput was purchased and no
model invocation was made.

Latest state for both: `agreementAvailability.status=PENDING`,
`authorizationStatus=NOT_AUTHORIZED`, `entitlementAvailability=AVAILABLE`,
`regionAvailability=AVAILABLE`. The previous `AUTHORIZED` status changed during
processing. Do not describe access as ready until availability is rechecked and
an admitted inference probe succeeds. No additional cause is inferred from the
pending state. [AWS model-access documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html)
describes FTU submission, agreement creation, and pending propagation.

Both `us.anthropic.claude-sonnet-4-6` and
`us.anthropic.claude-opus-4-6-v1` profiles are active and route to us-east-1,
us-east-2 and us-west-2; the controller policy includes these exact destinations.
Public quotes captured now: g6e.2xlarge Linux On-Demand $2.24208/hour;
c7i.large $0.08925/hour; Sonnet US profile $3.30/$16.50 per million input/output
tokens; Opus US profile $5.50/$27.50. These are not complete job estimates:
storage, transfer, logs, retries, cleanup and tax treatment remain to be bounded.

### Implementation delivered; actual pilot still blocked

- `scripts/bam_train_contract.py`: fixed TRAIN-only artifacts, immutable hashes,
  pregame cutoff/source/chunk checks, frozen ordinary brief, and independent
  outcome-blind accepted-gold validation.
- `scripts/bam_train_only.py`: separate probability/completion passes, exact
  outcome/gold denominators, zero residual head, joint adapter/head updates,
  TRAIN-only inference metrics and optimizer/RNG checkpoint resume.
- `scripts/bam_aws_preflight.py`: read-only prerequisite and funding checks.
  Credits and the original ledger are checked separately. Its output is not
  an atomic reservation or a launch receipt.
- `config/bam_train_only_pilot.json`: bounded initial diagnostic settings; GPU
  memory/throughput and real-data learning remain unqualified.

Still required: credit eligibility evidence and AWS quota/model readiness;
AWS source preparation and honest prior-only booster; frozen full-note research;
qualified Bedrock teacher/reviewer transport and accepted gold; storage and a
single-writer admitted launcher with external timed termination. The worker
requires matching preflight and launcher receipts; do not fabricate them.
No real NBA fit has run. DEV and TEST were not opened or processed.
Final integrated validation passed 169 targeted synthetic/boundary tests with
zero failures, errors or skips; results are saved as `tests.json` in the evidence
directory. Local checks are not a substitute for the admitted AWS learning pilot.

## Original installation record

2026-11-08 UTC (November 7 evening America/New_York). User requested installation
from the official AWS setup guide, selected profile `cs230` and region
`us-west-2`, completed browser sign-in, and explicitly approved connecting the
root-backed profile to Codex only.

## Installed and verified

- AWS CLI **2.39.6**, installed user-locally with AWS's shell installer. The
  macOS package signature verified against AWS team ID `94KV3E626L`.
- Binary: `~/.local/bin/aws`; installation: `~/.local/share/aws-cli`.
  The existing `~/.zshrc` already includes `~/.local/bin`; no duplicate PATH
  modification was needed. `uv` and `curl` were already available.
- Browser authentication with `aws login --region us-west-2 --profile cs230`
  completed. STS verified the root identity in the account ending `7290`.
  No access keys, passwords or cached sign-in tokens were printed or committed.
- `aws configure agent-toolkit --yes --region us-east-1 --profile cs230`
  installed 25 default AWS skills and generated MCP entries for detected agents.
  The remote skill catalog was successfully queried afterward.
- Codex's generated `aws-mcp` entry in `~/.codex/config.toml` now has
  `AWS_MCP_PROXY_PROFILES = "cs230"` in its `env` table. Generated command and
  arguments, and all other Codex settings, were preserved. A private pre-edit
  backup is in `~/.aws/agent-toolkit-backups/codex-only-20261108/`.
- An isolated MCP handshake using the exact Codex configuration succeeded:
  `MCP Proxy for AWS` version `1.8.0`, protocol `2025-03-26`, eight tools returned
  with all catalog pages read. No tool invocation was performed during that
  check; the temporary proxy process was terminated afterward.
- Other detected agents received the installer's default entries, but **no
  `cs230` profile binding was added to them**. A proposed multi-agent binding
  was blocked by automatic approval review; the user then approved Codex only.
- Official standard-account AWS guidance was appended to this worktree's
  `AGENTS.md` under stable markers, preserving all original project instructions.
  The rule source is Apache-2.0 licensed. Profile/root-identity evidence supports
  the standard-account path; the user did not identify a new-experience project.

The running chat does not automatically reload newly configured MCP servers.
Start a new Codex session to expose the AWS tools and installed skills there.
The CLI remains available through explicit `--profile cs230` in this session.
For another account later, sign in with `aws login --profile <name>`, add its
profile name to Codex's space-separated `AWS_MCP_PROXY_PROFILES` list, and
restart Codex. Do not copy the current binding to other agents without approval.

## Read-only AWS checks

The Free Tier service returned:

```json
{
  "accountPlanType": "PAID",
  "accountPlanStatus": "ACTIVE",
  "accountPlanRemainingCredits": {"amount": 100.0, "unit": "USD"}
}
```

This confirms $100 of remaining account-plan credits, not the user's stated
$200 total or the absence of other promotional credits. No billing-console
credit detail was available through the current UI tools. Do not treat account
credits as a hard spending limit or assume coverage of every Bedrock model.

Oregon (`us-west-2`) EC2 service quotas currently report:

| Quota | Code | Value |
| --- | --- | ---: |
| Running On-Demand G and VT instances | L-DB2E81BA | 0 |
| Running On-Demand P instances | L-417A185B | 0 |
| All G and VT Spot Instance Requests | L-3819A6DF | 0 |

The proposed `g6e.2xlarge` requires 8 G-family vCPUs. No increase has been
requested and no GPU instance launched by this setup. These observations cover
Oregon only; they are not an all-region quota or resource inventory.

The Oregon Bedrock catalog lists these active candidates, both requiring
an inference profile:

- `anthropic.claude-sonnet-4-6`
- `anthropic.claude-opus-4-6-v1`

The account-level availability check for Sonnet 4.6 returned
`authorizationStatus=AUTHORIZED`, `entitlementAvailability=AVAILABLE`,
`regionAvailability=AVAILABLE`, but `agreementAvailability.status=NOT_AVAILABLE`.
Catalog presence therefore does not establish invocation readiness. No model
agreement/subscription was accepted and no inference request was made.

## Before paid experiment work

1. Reconcile actual available credits, expiry and service eligibility with the
   user's billing view. Retain the existing $200 ledger cap and uncertain costs;
   do not assume the unverified difference is authorized out-of-pocket spending.
2. Set up project-scoped AWS execution access before provisioning compute.
   The current local root sign-in was approved only for the Codex connection.
3. Obtain sufficient Oregon G-family quota for one intended GPU instance.
4. Resolve the teacher model's agreement/access prerequisites and pin current
   prices, reasoning/output limits and an inference profile before calling it.
5. Implement the TRAIN-only worker, AWS orchestration and separately normalized
   gold objective described in the architecture. No DEV or TEST processing.

There are no new EC2, S3, IAM or training resources from this setup. No claim is
made about pre-existing resources elsewhere in the account.

## Upstream documentation issue

The supplied AWS rules reference a skill named `aws-secrets-manager`, but both
`get-skill-file` and `get-skill-metadata` returned HTTP 404 for that name. A
catalog search instead returned `creating-secrets-using-best-practices` among
other relevant skills. No secret values were retrieved. Use available official
sign-in/IAM guidance for those workflows; resolve the appropriate skill before
any future Secrets Manager task.

## References

- [Official setup guide](https://raw.githubusercontent.com/aws/agent-toolkit-for-aws/refs/heads/main/setup-instructions/setup.md)
- [Setup troubleshooting](https://raw.githubusercontent.com/aws/agent-toolkit-for-aws/refs/heads/main/setup-instructions/setup-troubleshooting.md)
- [Installed AWS rule source](https://raw.githubusercontent.com/aws/agent-toolkit-for-aws/refs/heads/main/rules/aws-agent-rules.md)
- [Toolkit license](https://github.com/aws/agent-toolkit-for-aws/blob/main/LICENSE)
