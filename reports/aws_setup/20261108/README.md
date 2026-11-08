# AWS prerequisite continuation

**Latest email/access update:** `cpu_first_region_email.json` records the user's
three-region EC2 launch restriction and fresh-check failures (MCP Unknown tool;
CLI cs230 session expired). Opt-in-enabled regions are not the new-account
EC2 launch allowlist. No current CPU quota or new request status was obtained.

**05:11 UTC update:** one minimal gpt-oss-120b access probe was rejected with
`ValidationException: Operation not allowed`. One cent remains reserved as
uncertain; no successful inference or usage returned. See `bedrock_access_probe.json`
for the full tool result (including its subsequent handler NameError) and
`probe_budget_state.json`. Current admission availability is $186.40, with
$1.83 outstanding and prior estimates/buffer preserved. No GPU launch or real
dataset processing occurred.

**05:07 UTC update:** payment-method change and retry cleared Claude agreement
errors (now AVAILABLE); authorization remains NOT_AUTHORIZED. Direct scans found
G/VT On-Demand and Spot quotas zero in all 17 enabled regions. A new us-east-1
8-vCPU request is CASE_OPENED, case 179411421300610. SageMaker alternatives did
not establish a usable route. See `access_followup_0507.json` and the all-region
quota evidence. Broad ListServiceQuotas initially timed out; its error is saved
and no completeness claim relies on it. User permits any region and wants
efficiency; `teacher_us_east_1_prices.json` records cheap AWS-sold candidates.
The explicit-region preflight/worker change passed 32 focused synthetic tests.

**04:55 UTC update:** `followup_0455.json` confirms the $100 balance, zero
Oregon G/VT quota and unchanged Claude payment-validation failure.
`user_credit_details.json` records the user's applicable-products evidence
and late-2027 expiry statement. Browser access is no longer needed for those
fields; no additional earned credit was established.

**04:46 UTC update:** `followup_0446.json` supersedes pending model statuses:
both Anthropic agreements failed with `INVALID_PAYMENT_INSTRUMENT`. The user
also supplied the GPU-denial email, despite quota API status still CASE_OPENED.
The unsent appeal is in `docs/aws_gpu_quota_appeal.md`. No paid request was made.

Account checks at 2026-11-08 04:34 UTC; model/quota readiness rechecked at 04:38 UTC.
These checks cover the account-plan credit APIs, global IAM resources created
for this project, and Oregon compute/Bedrock prerequisites. They are not an
all-region resource inventory or complete promotional-credit audit.

| Prerequisite | Observed result |
| --- | --- |
| Integrated tooling | AWS MCP calls succeed with the approved Codex-only identity |
| Earned account-plan credit | $100, PAID / ACTIVE |
| Additional reward opportunity | Five $20 activities, all NOT_STARTED |
| Credit expiry/service eligibility | User reports late 2027; supplied list includes EC2, Bedrock foundation models, S3, CloudWatch and transfer |
| Runtime IAM | IAM-only stack CREATE_COMPLETE; three scoped EC2 roles/profiles |
| GPU quota | G On-Demand and Spot 0 in all enabled regions; new N. Virginia case 179411421300610 |
| Anthropic FTU | Missing form confirmed by full API error, then submitted |
| Sonnet/Opus model agreements | AVAILABLE after payment fix and retry; still NOT_AUTHORIZED |
| Experiment execution | No launch; one minimal model-access probe rejected, one-cent reserve retained |

`account_checks.json` preserves the relevant raw read responses. `iam_effective.json`
preserves role trust/permissions and profiles read from IAM; they matched the
deployed template. `iam_simulations.json` contains policy simulations, not live
instance tests. `price_quotes.json` contains public EC2 and relevant Oregon
Bedrock offer rates; no presigned offer URLs or credentials are stored.

The Support API cannot read the opened case on this account:
`SubscriptionRequiredException: Amazon Web Services Premium Support Subscription is required to use this service.`
Use the Support console if correspondence is required; no paid support upgrade
was made. Browser inventory reported `The browser access host is unavailable`,
and native Chrome inspection timed out, preventing credit-detail inspection.

Validation: cfn-lint 1.60.0, project IAM Guard rules with checksum-verified
cfn-guard 3.2.1, AWS ValidateTemplate, deployed-template/policy read-back and
six IAM simulations. No deployment validation errors were returned.
The local Guard rules are limited to this IAM stack. Durable CloudTrail delivery
was not established; Oregon DescribeTrails (including shadow trails) returned
an empty trailList.

Final integrated validation: **169 tests passed**, zero failures/errors/skips
(`tests.json`). These cover TRAIN contracts, joint fitting on tiny synthetic
tensors, checkpoint resume, funding checks, the existing budget ledger, and
research/API boundaries. They do not establish Qwen/L40S throughput or NBA learning.

The existing experiment ledger remains unchanged at $200 cap, $6.77 completed
estimates, $1.82 outstanding, $5 buffer and $186.41 available for admission.
Ledger availability is not AWS credit availability. No receipt or reservation
was fabricated to run the new worker.

See [setup and remaining implementation](../../../docs/aws_toolkit_setup.md)
for exact roles, inference profiles, next steps and execution limits. Real-data
learning, source preparation, research and gold generation are still unperformed.
