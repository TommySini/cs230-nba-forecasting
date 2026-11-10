# Project working instructions

- This is Tommaso Sini and Anthony Viverito's Stanford CS230 NBA forecasting project.
- Read `HANDOFF.md` and `README.md` before continuing.
- **Rewrite, don't accumulate.** When you change how a process works (new
  features, model, split, data source), edit the file that implements it. Keep one
  file per job: one `run_gradient_boosting.py`, never a `run_rich_gradient_boosting.py`
  beside it. Create new files only for genuinely new jobs, and delete whatever the
  change makes obsolete in the same change. No `_v2`/dated copies, and no keeping
  finished experiments' code. Git history keeps everything (the tag `pre-cleanup`
  holds the state before the November 2026 cleanup).
- **Reports stay lean:** `reports/` contains three maintained research reports and
  a reading guide, plus the run records the dataset release pins by path
  (`reports/gradient_boosting/`, `reports/polymarket/`, `reports/tabpfn/`; never move
  them without a new data release). Archived metrics belong in `results/`; dataset manifests and
  collector diagnostics stay with their data. Keep supporting files only when
  active code needs them or they preserve essential unique results. Update the
  owning report, delete superseded prose/copies, and cite Git history for retired
  investigations. Collectors and evaluators must not write report narratives.
- **Fixed splits:** TRAIN = seasons 2012–13 to 2023–24, DEV = 2024–25,
  TEST = 2025–26, defined once in `config/gradient_boosting.json`. Fit on TRAIN,
  choose on DEV, score TEST once. Do not change the splits.
- **Frozen datasets:** once a dataset is built and validated, leave it alone.
  Don't recollect, re-normalize or extend it unless the user asks.
- Use the repository's `.venv` for all Python work: `.venv/bin/python` and
  `uv pip install --python .venv/bin/python --group local` (from `pyproject.toml`).
  Do not install packages into system Python or the global Conda environment.
- The canonical repository is `/Users/tommasosini/Desktop/Repos/cs230-nba-forecasting`.
  All datasets live in its git-ignored `data/`. In any other worktree, make
  `data` a link to it instead of copying:
  `ln -s /Users/tommasosini/Desktop/Repos/cs230-nba-forecasting/data data`.
- Preserve raw sources and provenance. Never feed raw ESPN responses to the model:
  they contain postgame results, recaps, and current news.
- Keep outcomes separate from pregame inputs. Enforce source timestamps at or
  before each game's prediction cutoff (tipoff minus 60 minutes).
- The game notes are a selected corpus (some teams and seasons), not league-wide
  coverage. Do not claim otherwise.
- After collector or dataset changes, run the tests and the matching validator
  (`validate_notes_v2.py`, `validate_polymarket.py`, `validate_gradient_boosting.py`,
  or `bam_data.py validate <snapshot>`).

## AWS project context (2026-11-08 setup)

- Use AWS for new experiment compute. The user authorized any suitable AWS
  region on 2026-11-08; Oregon was the original selection. Pin each launch plan
  to one region with matching live quota, permissions and pricing evidence.
  The Agent Toolkit/MCP service itself uses `us-east-1`.
  Exception: on 2026-11-09 the user explicitly authorized all TabPFN baseline
  compute locally, like gradient boosting. Use this worktree's `.venv` and the
  frozen boosting inputs; this does not change BAM's AWS execution policy.
- The user approved binding profile `cs230` to **Codex only**. Do not propagate
  this root-backed sign-in to other agents. Use a scoped execution role before
  launching training infrastructure; do not put sign-in credentials on instances.
- Follow `docs/bam_training_first.md` and `docs/bam_aws_architecture.md`:
  first prove learning on 16–32 TRAIN games; defer DEV and exclude TEST.
- The experiment cap is $200, including existing tracked estimates/reservations.
  The AWS Free Tier API reported only $100 in account-plan credits on 2026-11-08;
  credit eligibility and the complete credit balance remain to be reconciled
  before spending. A paid AWS account can charge beyond its credits.
- Setup details and remaining prerequisites are in `docs/aws_toolkit_setup.md`.

<!-- BEGIN AWS Agent Toolkit rules -->

# AWS Guidance

- Where these AWS rules conflict with the project's own instructions, the
  project's instructions take precedence.
- Prefer the AWS MCP Server for AWS interactions — it provides sandboxed
  execution, observability, and audit logging. If unavailable, use the
  AWS CLI directly.
- For any AWS task or question — building, configuring, migrating, debugging,
  or operating a service, writing SDK code, authoring CDK/CloudFormation, or
  answering how a service works — first call `aws___search_documentation` with
  topic `agent_skills` and load the matching skill with `aws___retrieve_skill`.
  Do this even when you believe you know the answer: skills carry current
  service behavior that your training may not. Prefer the skill's guidance over
  general knowledge. Skip it only for work that does not involve AWS.
- When uncertain about specific AWS details (API parameters, permissions,
  limits, error codes), verify against documentation rather than guessing.
  State uncertainty explicitly if you cannot confirm.
- Before reporting a count, a total, or that something does not exist, read
  every page of the results and search every Region and scope where it could
  be. If you could not, say the result is partial and name what you checked.
- To say what a system currently does, read what the runtime uses: the
  completed deployment, the alias or version that is invoked, the template the
  stack is running. A newer stored version, a draft, or a description field is
  not evidence of live behavior.
- Present a cause as confirmed only if data, configuration, or logs you read
  show it. Label anything else as unverified, and say what you would check
  next.
- Before you interpret the records that show a failure, such as error log
  lines, failed executions, or table items, read one of them whole, with
  every field. Don't truncate messages or select fields in your own code
  until you have.
- When an error says a value was undefined, null, empty, or missing, find
  the input, query result, or item that supplied that value. If the account
  shows why it was empty, such as no item matching the key, read that
  before you call the cause unverified.
- When a field in a record you read explains the result, such as an error
  counter, an empty array, a null, or a state value, name the field and its
  exact value in the answer, and say which record it came from.
- When the user asks a specific question or names a suspected cause, confirm
  or rule out each cause they raised, and do not offer alternatives your evidence
  does not support.
- When creating infrastructure, prefer infrastructure-as-code (AWS CDK or
  CloudFormation) over direct CLI commands.
- When working with infrastructure, follow AWS Well-Architected Framework
  principles.
- Do not use em dashes in AWS resource names or descriptions. Use
  hyphens instead.

## Before you conclude

- Before deciding the AWS account can't hold what the request asks for,
  look. List the places it could live, such as S3 buckets, databases, and
  log groups, and read what is there.
- Don't conclude that records are missing from metadata, such as a
  creation date or a stored-bytes count, or from one query that failed or
  came back empty. Read the records another way, across the full time
  range, before reporting them absent.
- A tool's own error is evidence. If a call the answer depends on fails
  with expired or missing credentials or AccessDenied, report that error
  and what it blocks, and don't overrule it with another check. Searching
  other Regions is for finding resources, not for retrying an auth error.
  Retry throttling with backoff before you report it.

## Secret Safety

- MUST load the `aws-secrets-manager` skill first for any secret,
  credential, API key, token, or password task. MUST NOT call
  `secretsmanager get-secret-value` or `batch-get-secret-value`, and MUST
  NOT hit the Secrets Manager Agent daemon directly. MUST use
  `{{resolve:secretsmanager:secret-id:SecretString:json-key}}` with
  `asm-exec` so the secret resolves at runtime without entering context.

<!-- END AWS Agent Toolkit rules -->

