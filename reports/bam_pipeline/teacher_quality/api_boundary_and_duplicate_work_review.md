# Offline API boundary and repeated research work

The API helper is preparation only: it does not load credentials, call a provider,
or change the production driver. The scientific comparison requests remain the
original 16 byte-for-byte. Their frozen labels remain separate, including the
known defective header control; report the 15-case sensitivity separately.

## Trusted preparation boundary

`load_train_registry(snapshot)` verifies the input file against its snapshot
manifest without opening labels. The caller must pin that manifest in its run
protocol. `bind_trusted_training_context` takes canonical games from the trusted
snapshot loader **after** source and coverage-sidecar validation. It binds the
complete source objects, including attached coverage annotations. This is a
caller trust contract, not a replacement for canonical file/image verification.

The transport also loads an independently pinned request authorization manifest
using `load_request_authorizations(path, expected_sha256)`. Each entry binds the
entire scientific request hash, ID, game, stage and source hash. Rehashing a
modified request does not authorize it. Transport must never create this manifest
from the candidate requests it is trying to submit. Trusted stage builders write
the manifest; the preparation/protocol artifact pins its digest.

Only TRAIN postmortem, combined gold and gold review are permitted. Gold still
requires all three researcher outputs. Gold review requires its intact matching
target, a distinct reviewer identity and the original pregame context; postgame
content is excluded from original and repair review messages. Modern API teachers
remain retrospective TRAIN supervision, not historical prediction models. The
prediction student and critic retain their pinned historical eligibility.

The separate comparison path accepts only exact requests from the independently
pinned original actual-claim manifest. Its original requests do not gain a new
invented source hash. Provider envelopes preserve scientific message content and
request hashes. Returned provider model/version identity, budget admission,
reasoning/context accounting and durable response provenance remain transport
responsibilities.

Distinct configured model strings are only an identity guard, not proof of
independent models: two aliases may resolve to the same deployed model, and a
mutable alias can change. Before submission, transport must bind resolved pinned
model/version identities; after response it must check returned identity against
that binding. If the provider cannot establish distinct resolved identities,
generator/reviewer independence remains unverified and must block an acceptance
claim. Different providers or different request IDs alone are not that proof.

No four-game gold bundle is ready: actual qualified modules/postmortems are still
missing. The pending manifest records this; no gold tasks or approvals were
fabricated.

## Avoidable work versus required hierarchy

The current driver generates and reviews every initial partial/final module,
then generates and reviews every final student report again. The 4,982 initial
generation count is a structural extrapolation, not a measured full-cohort count.
There are exactly 1,272 final module inputs for 424 games: 1,098 TRAIN and 174
DEV/TEST.

1. **Omit initial DEV/TEST final outputs.** Continue all needed partial hierarchy,
   but stop when the three final requests exist. Export those requests directly.
   This removes 174 initial final generations and 174 corresponding reviews;
   later adapted student generation and its final acceptance gates are unchanged.
   It does not remove partial synthesis: their outputs are needed to construct
   bounded final prompts. The driver must distinguish a prepared request from an
   approved output and must not create `prompted_modules.json` placeholders.

2. **Cold-fold exact reuse is conditional.** The zero-training fold has 411 final
   rows. Its initial final reports may replace later base-model generation only
   when checkpoint, adapter absence, precision, tokenizer, messages, sampling,
   completion and context contracts all match. Reviews also require identical
   target/context/reviewer provenance. Current BF16 versus NF4 paths do not meet
   this automatically. Never relabel a BF16 base output as an NF4 generation or an
   adapted student output. This can save up to 411 generation/review pairs after
   an explicitly unified runtime contract, not in the current run by assertion.

3. **TRAIN drafts are a new protocol, not skipped approval.** BAM can use explicitly
   unverified researcher drafts as input for a correcting gold teacher. However,
   the current gold builder takes only sources cited by final drafts. Likewise,
   synthesis only retains partial findings. Dropping initial reviews under the
   present context construction can silently remove useful or contradictory
   evidence before the final gold gate sees it. A safe draft protocol must retain
   full original selected evidence and omission/coverage audit through synthesis,
   gold and independent review, with measured context budgets. Drafts must remain
   mechanically valid but unapproved; every final gold and actual forecast-input
   report still needs its existing mechanical and independent semantic gate.
   Current gold requires approved parents, so this is an explicit builder/schema
   change, not a driver flag. Do not count its savings before four-game quality
   and source-context sizing validate the changed protocol.

Recommendation: implement the first omission when architecture is released;
consider exact cold-fold reuse with the new runtime contract. Keep draft-review
removal a separately reviewed option. A cheaper selector can reduce output waste,
but retaining formerly rejected source evidence may increase legitimate module
hierarchy work. Neither repeated calls nor fewer calls establish report quality.
