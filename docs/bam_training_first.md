# BAM training-first phase

Updated 2026-11-07 following the user's instruction to establish that the model
learns on TRAIN before using DEV, and to leave TEST out of the current work.
This document supersedes the execution order in the November 3 budget proposal.
It records a protocol; it does not certify a prepared dataset or a completed fit.

**AWS-only update, approximately 20:40 America/New_York:** the user requested the
complete architecture before execution and specified all compute on AWS. Follow
[`bam_aws_architecture.md`](bam_aws_architecture.md): Bedrock teacher/reviewer,
EC2 processing/research/training and S3 artifacts. No launch or paid request is
authorized by this architecture-review step; prior Modal/direct-provider launch
instructions are superseded for new work.

## Current budget and priorities

The user set the current experiment budget to **$200** on November 7 at about
20:34 America/New_York. This supersedes the earlier $100 feasibility proposal
and $150 compute authorization. Conservatively include the existing tracked
completed estimates and uncertain reservations inside $200; do not reset them
or present this as a verified total of all historical project bills.

The current budget configuration is `config/bam_budget_200_20261107.json`.
It uses the same ledger under its original filename, with an audited cap change
and a byte-preserved backup. At revision, $6.77 completed estimates and $1.82
uncertain costs remain accounted for, plus the existing $5 buffer, leaving
$186.41 admissible for new calls. These are estimates, not invoices. Refresh
provider quotes before admitting paid work; the new configuration has no GPU
cost policy yet, so it cannot silently reuse old rates to launch jobs.

First fund the representative 16–32-game TRAIN proof of learning, including
reviewed research/gold, actual joint loss reduction and measured throughput.
The pilot must be admitted against bounded API-token and GPU-time reservations.
Use its quality and cost measurements to allocate the remaining budget between
broader accepted gold coverage and full-TRAIN fitting. No larger gold count,
extra model family, researcher fine-tuning or completed full-corpus run is
promised merely because the cap increased. DEV remains deferred and TEST excluded.

The planned forecaster remains the already integrated Qwen2.5-14B-Instruct with
LoRA and a residual probability head. It is an instruction model, not a model
with a separate extended-thinking mode. A stronger reasoning-capable API teacher
is a candidate for TRAIN gold and checking; the revised teacher has not been
selected or qualified. No reasoning-token budget is assumed to be free.

## Gold reports are auxiliary supervision

The earlier proposal of about 200 accepted gold reports was a budget allocation,
not a scientific optimum, a limit on outcome-training examples, or an approved
fixed sample size. First measure research quality, accepted-target cost and actual
training throughput on TRAIN. Then size gold coverage within the existing budget;
do not silently increase spending or promise gold for every training game.

Every eligible TRAIN game contributes its actual home-win outcome to Brier loss.
Games with an accepted gold report additionally contribute explanation loss. For
the provisional 1,506-game training cohort and 200 accepted gold examples, this
would mean 1,506 outcome targets and 200 explanation targets, not 200 training
games. The same forecaster and probability head are shared across all games.

Use the objective

    mean_TRAIN((p_home - home_win)^2)
      + explanation_weight * mean_GOLD(per-example completion cross-entropy)

Normalize the two terms separately. Missing gold means no explanation target;
it does not mean dropping that game's outcome, assigning an empty target, or
silently diluting the explanation objective by the fraction of games with gold.
Track Brier, explanation cross-entropy and the weighted total separately. Keep
the weight fixed when comparing total-loss curves; changing the weight alone
must not count as improved learning.

Gold is created from the researcher's pregame output and a separate TRAIN-only
postmortem. It must retain uncertainty and contain only pregame-supported claims.
The forecaster's probability pass receives the ordinary pregame research and
prior-only boosting probability. Gold is a target in a separate completion pass,
never an input to the probability head. Researcher and teacher remain fixed in
the lean design; LoRA adapters in the forecaster and its residual head are trained.

## Execution order

1. Prepare and audit TRAIN inputs, outcomes and honest prior-only booster
   forecasts. Freeze a representative 16–32-game TRAIN diagnostic sample using
   source/team/season/packet-length coverage. Its notes must be fully processed,
   and its gold reports independently reviewed. No DEV or TEST research,
   inference, labels or scoring belong to this phase.
2. Run a bounded small-sample fitting diagnostic. Record the zero-adjustment
   booster baseline, input/label alignment, finite losses and parameter updates.
   Verify learning in both the head and LoRA adapters over multiple steps; with
   a zero-initialized head, an initial zero Brier gradient to the adapters is
   expected. Intentionally fit this small sample closely to diagnose whether
   the optimization and supervision work. A successful tiny-sample fit is not
   evidence of generalization.
3. Expand to the full eligible TRAIN set. Save checkpoints and comparable
   training curves, including inference-mode Brier on the fixed training cohort
   and mean per-example gold completion loss. Diagnose stagnant loss, unstable
   gradients, data problems and conflicts between objectives using TRAIN only.
   Do not demand zero full-dataset loss or spend indefinitely: outcomes are noisy
   and memorizing them is not the eventual forecasting objective. Bound runs
   by the current budget and measured throughput.
4. Only after the training diagnostic works and full-TRAIN learning is understood,
   assess generalization on DEV. Use DEV Brier to select among properly trained
   checkpoints and the zero-adjustment baseline; train loss is a diagnostic, not
   a requirement to choose an overfit checkpoint. No gradients or gold targets
   from DEV. A low train loss with worse DEV performance calls for less fitting
   or better regularization, not further minimization of training loss alone.

TEST is excluded from this phase: no generation, prediction, scoring, inspection
or automatic evaluation stage. Previously exposed historical results cannot be
made unseen; that does not authorize additional TEST work now.

## Implementation and launch status

**2026-11-08 implementation update:** the new standalone
`scripts/bam_train_only.py` path loads only a hash-pinned TRAIN bundle through
`bam_train_contract.py`. It supports separately normalized sparse-gold training
internally; the first public pilot entry requires accepted gold for all 16–32
games. It records inference-mode TRAIN metrics, verifies joint head/adapter
updates, and preserves optimizer and random state on resume. The example config
pins a maximum of 32 full-cohort optimizer updates / 1,800 seconds, with success
defined as at least 50% Brier and 10% gold CE reduction and verified updates.
These initial diagnostic settings have only synthetic validation; no throughput,
memory fit, NBA learning or generalization result is established.

`bam_aws_preflight.py` checks earned eligible funding separately from the existing
$200 ledger and reports missing prerequisites. Its result creates no reservation.
The worker additionally requires a launcher receipt bound to the job, source
manifest, exact config and model, with an external termination assertion.
The launcher and its live checks/atomic reservations are still to be implemented;
do not fabricate receipts to bypass them. Latest account state is in
[`aws_toolkit_setup.md`](aws_toolkit_setup.md).

The current production `train_forecaster` loads DEV labels automatically and
selects candidates on DEV each epoch. It is not the entry point for this first
TRAIN-only phase without an explicit implementation change. The old pipeline's
`--stage all` also reaches evaluation and must not be used for this phase.

Before a paid launch, integrate the new worker with qualified frozen research,
AWS Bedrock teacher/reviewer transport, source-frozen inputs and accepted gold.
Research/teacher API readiness, representative target quality and
actual joint-training cost remain unqualified. The frozen snapshot `bam_notes_v2`
has 1,456 forecast-eligible TRAIN games with the full 252-feature history. No paid
job is launched by this protocol update.
