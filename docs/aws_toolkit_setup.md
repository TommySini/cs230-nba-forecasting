# AWS setup and remaining prerequisites

This is a consolidation of recorded setup work, as recorded on **2026-11-08 UTC**. Report cleanup made no AWS calls or changes. The observations below
are dated evidence, not a fresh account-readiness check. Experiment design lives
in [TRAIN-first training](bam_training_first.md) and
[the AWS architecture](bam_aws_architecture.md).

## Recorded access and blockers

The local `cs230` profile is bound to **Codex only**. The user approved that
root-backed connection for account setup; it must not be propagated to other
agents or placed on instances. Paid execution requires scoped roles and the
existing admission controls.

Recorded G/VT On-Demand and Spot quotas were zero in all 17 enabled regions.
Two 8-vCPU requests were open: Oregon case `179411216100407` and N. Virginia
case `179411421300610`. The Oregon denial email and the user's November 8
appeal remain relevant; the API status did not overturn the email. The Support
API returned `SubscriptionRequiredException`, so it could not read correspondence.
Later account checks failed because the MCP tools were unavailable and the
`cs230` CLI session had expired. No fresh CPU quota or launch-region list was
obtained in those checks.

The user supplied an email limiting EC2 launches to three regions, but their
identities were not established. Earlier quota reads across 17 enabled regions
did not establish launch permission there. The user permits any suitable AWS
region; each proposed launch must bind matching quota, permissions and prices
to one region. The toolkit service itself uses `us-east-1`.

Recorded Bedrock agreements became AVAILABLE, while Sonnet, gpt-oss-120b and
DeepSeek V3.2 remained `NOT_AUTHORIZED`. One admitted gpt-oss-120b access probe on November 8 returned
`ValidationException: Operation not allowed`. Its handler also raised a
`NameError`, but the original provider error was retained. No successful usage
was returned and no retry was made. The account-specific cause remained
unconfirmed. The user sent the Bedrock inquiry.
Agreement changes, broader IAM and paid CPU usage are not established remedies.

The original probe did not use NBA data. It reserved one cent, which remains
uncertain pending billing evidence rather than being reported as a known charge.
A new probe requires its own admission; historical agreement or catalog records
are not invocation authorization.

## Funding and execution boundaries

The recorded project ledger had a $200 cap, $6.77 completed estimates, $1.83
outstanding reservations, a $5 buffer and $186.40 available. The authoritative
configuration is [`bam_budget_200_20261107.json`](../config/bam_budget_200_20261107.json);
its ledger remains in `data/processed/bam_budget/`. Do not reset it or treat these
figures as an AWS credit balance.

Account-plan checks reported $100 earned credit. An additional $100 depended on
five activities that were then `NOT_STARTED`. The user supplied eligible-product
information and reported expiry in late 2027; an exact date was not supplied.
These records do not establish another $100 of earned credit. Reconcile actual
remaining credits and applicable services before spending. A paid account can
charge beyond its credits; the project cap is not permission to assume cash
funding for an unverified difference.

The intended first experiment uses 16–32 TRAIN games. DEV is deferred and TEST
excluded. No real NBA pilot fit was recorded. A synthetic CPU smoke test would
not establish 14B GPU learning or throughput. Do not launch idle machines to
build usage history. Teacher choice still requires appropriate output-training
rights and quality qualification; the earlier review left the applicability of
Anthropic's competing-model terms unresolved.

## Existing scoped infrastructure

The IAM-only stack `cs230-train-pilot-20261108` was recorded `CREATE_COMPLETE`
in Oregon. Its [CloudFormation template](../infra/aws/train-pilot-iam.json)
creates controller, researcher and forecaster roles/profiles under `/cs230/`.
The researcher cannot read outcomes/gold; the controller cannot create IAM or
compute resources. The recorded template and role-policy readbacks matched.
Policy simulations passed the intended separation checks, but were not runtime
instance tests.

The template restricts artifacts to the planned Oregon bucket and pilot prefix;
that setup work did not create the bucket or log groups. Moving compute regions
requires checking those constraints. No static credentials were placed on an
instance. Durable CloudTrail delivery was not established by the recorded check.

The TRAIN contract, worker, budget checks and teacher payload builders exist.
Remaining work includes teacher transport, driver integration, storage and admitted preparation of audited
research/gold, and a single-writer launcher with external timed termination.
The worker requires matching preflight and launcher receipts; do not fabricate
them. See [HANDOFF](../HANDOFF.md) for the current code-work sequence. Old price
quotes and test counts are historical evidence, not a current launch admission.

## Installation record and historical evidence

The November 8 installation recorded AWS CLI 2.39.6 at `~/.local/bin/aws`,
installed under `~/.local/share/aws-cli`. Codex's AWS MCP configuration binds
`AWS_MCP_PROXY_PROFILES` to `cs230`; other agents received no such binding.
The existing shell PATH was preserved. Skill discovery and a proxy handshake
succeeded. These records do not imply that a later session is authenticated.
The repository's AWS tool/skill rules remain in AGENTS.md.

The original setup document and all 30 evidence files remain in Git at
`67596558a188014803724bd60a158ce6f205ec3e`. Their former working-tree directory
was `reports/aws_setup/20261108/`. Read a record without restoring the folder:

```bash
git show 67596558a188014803724bd60a158ce6f205ec3e:reports/aws_setup/20261108/bedrock_access_probe.json
```

| Question | Historical record in that directory |
| --- | --- |
| Earned credits and later user-supplied expiry/eligibility | `account_checks.json`, `followup_0455.json`, `user_credit_details.json` |
| Probe admission, provider error and uncertain reservation | `probe_admission.json`, `bedrock_access_probe.json`, `probe_budget_state.json` |
| Three-region restriction and later access continuation | `cpu_first_region_email.json` |
| GPU quota observations and request | `g_on_demand_all_enabled_regions.json`, `us_east_1_quota_request.json`, `access_followup_0507.json` |
| Role readback and simulations | `iam_effective.json`, `iam_simulations.json` |
| Regional price quotes | `price_quotes.json`, `teacher_us_east_1_prices.json` |

The setup chronology is recoverable with `git show` at the same revision and
path `docs/aws_toolkit_setup.md`. Preserve the original observations when using
these records; old error states and earlier budget totals were superseded by
later entries. No missing credentials, secret values or account details need to
be copied into reports.
