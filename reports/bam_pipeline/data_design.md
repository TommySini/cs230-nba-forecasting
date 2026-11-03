# BAM immutable data preparation

`scripts/bam_data.py prepare --config config/bam_pipeline_v1.json --workers 8`
creates a new, content-addressed snapshot under
`data/processed/bam_pipeline/<snapshot_id>/`. A caller can supply `--snapshot-id`;
an existing ID is validated and reused only when the complete input identity
matches. It is never overwritten. Partial extraction is staged in a hidden
directory and the completed snapshot is renamed atomically. Each new arrival of
approved notes needs another snapshot; running jobs retain their frozen inputs.

The manifest binds note manifests, raw bytes, numerical history, frozen booster,
source-verification and extraction code, reviewed transcription references,
configuration, and relevant Python package versions. Raw sources are copied
verbatim inside the snapshot rather than depending on mutable external paths.
Every output file has a SHA-256 entry in `manifest.json`.

`inputs.jsonl` contains one row per game, with game/date/team metadata, the
scheduled tipoff and its minus-60-minute cutoff, chronological split, prior-game
features, an earlier-only booster probability or null, eligibility/reason, and
all source references. `labels.jsonl` contains the separate outcome labels.
`boosting_lineage.json` lists fitting IDs for every monthly or frozen fold.
No recap, target score or outcome label is copied into prediction inputs.

Sources are identified by `source_id = "sha256:" + raw_byte_hash`. Each reference
contains publisher, publisher URL, exact archive replay URL/time, archive metadata,
original raw path, snapshot raw path, extraction path and extraction hash. All
PDF pages and the plain/layout extractions are retained. Selected page text is
split into one-indexed lines preserving line endings: concatenating the line
texts reconstructs the page exactly. Citations use source ID, page, line and
verbatim quote. Figures, sparse pages, undecoded tables and suspicious glyphs
remain explicitly flagged; first-page approval does not certify full-packet
research quality.

Multiple packets for one game merge deterministically. Identical raw hashes
deduplicate, retaining provenance aliases. Distinct archived versions of the
same publisher URL remain available for audit; the latest verified pre-cutoff
capture is active for research. Conflicting byte hashes claiming the same URL
and capture time fail. Distinct publisher URLs remain active, allowing both
teams' packets. `load_snapshot_game(snapshot_directory, game_id)` materializes
active page/line sources, checks their hashes and never opens the labels file.

All packets from a game and all games on an ET date share a split. Historical
games at or before the prediction model's knowledge cutoff are training-only,
as are all games used to fit/select the original booster. Games lacking five
prior games per team or 200 eligible examples before the target calendar month
remain research-only training supervision, with null booster probability. The
older eligible games use the original frozen booster settings fitted only on
months preceding each target month. Games after the original booster's final
fitting date use that immutable original model. No hyperparameter selection is
performed by this preparer.

The initial split preserves the existing pilot's date boundaries: development
begins December 4, 2025; test begins February 1, 2026. Existing inspected test IDs
are explicitly listed and their status is exploratory. The script does not
certify historical games as fresh merely because an ID is absent from the pilot
manifests: model-knowledge and prior-analysis exposure still require review.

Current discovery found this worktree's 256 approved collection records, the
older one-record approved source sample, and 107 complete extraction caches.
The 149 approved 2023–24 records are already included in the 256, not additive.
No additional approved note manifests were found in the canonical checkout or
sibling repository worktrees at the initial read-only inventory. Claude's live
collector is independent; add its atomically published, validated approved
manifest through `notes_paths` in a new configuration/snapshot. A timestamp
assertion alone is not accepted as archive proof.

Run `scripts/bam_data.py validate <snapshot_directory>` to check every artifact,
source hash, source cutoff, page/line reconstruction, split/date boundary and
prediction eligibility. Focused unit tests exercise duplicate/version handling,
cutoff rejection, booster training membership, old-game quarantine to training,
whole-date splits, outcome separation and tampering failures.

## Expanded PDF/HTML snapshots

`scripts/bam_data_html.py` is the separate v2.1 builder. It leaves the active v1
builder and all prior snapshots unchanged. Pass the frozen original and expansion
PDF manifests plus `approved_html.jsonl` as repeatable `--notes-path` arguments,
with a chosen `--snapshot-id` and `--cache-snapshot` pointing to a validated prior
snapshot. Frozen copies of mutable collection manifests are required before a
final training-corpus preparation; the builder rechecks all input hashes before
atomic publication.

HTML sources carry `source_kind: html` and `pagination: article_sections`.
Numbered citation pages are explicitly labeled `page_kind: article_section`:
they are original article blocks, not physical PDF pages. Canonical line text
and section hashes are preserved. Title and nearest-heading context reference
their original section/line IDs; they do not create synthetic citation text.
The source audit retains excluded nontext block dispositions. Unknown textual
blocks, future modifications, wrong game identities and missing archive proof
remain quarantined. CMS-only clean timestamps remain a separate weaker tier.

Every admitted HTML source has a full-source temporal/content review bound to
an immutable per-source review JSON, raw-body hash, exact article-section hash
and parser version. Approval establishes source identity and pre-cutoff
availability, not universal truth of the publisher's basketball assertions.
Known internal contradictions and template errors stay visible in quarantine.

The new collector validates official NBA prior records with three separately
pinned Cup championship exclusions. The original booster keeps its frozen
feature history, including those actual prior Cup finals. Eight other special
All-Star event rows in the raw history do not survive the numerical warmup and
are absent from NBA feature ancestry. See `baseline_scope_audit_v1.json` and the
new collector's plan for exact event IDs and provenance.

The final expanded input snapshot is
`data/processed/bam_pipeline/bam_v2_20261102_expanded`, with 424 research games
(366 train, 21 dev, 37 test), including 319 forecast-eligible games
(261 train, 21 dev, 37 test). The 105 research-only games remain TRAIN-only.
All original 256 games, their split memberships and source hashes are retained;
their boosting probabilities match exactly. The snapshot contains 444 PDF
packets and 25 HTML articles, with 13,262 physical PDF pages and 450 article
sections. All 788 planned PDF capture leads have been assessed; no new search
scope is implied by this freeze.

Portland uses the separate `bam_html_por.py` adapter. It requires an explicit
full weekday/date/year opening, both independently validated official prior
records, source home/away routing and a verified Pacific wall-clock tipoff.
The authoritative cutoff remains the verified UTC schedule minus 60 minutes.
Literal PST/PDT discrepancies remain warnings rather than changing the cutoff.
Five of nine archived Portland articles passed; four remain quarantined.
Original canonical article sections are unchanged. Derived `context_sections`
refer only to actual title, dated opening, topical marker and dated prior-game
header sections, so historical recap paragraphs retain their own date context.
Trusted review warnings are stored at `game.sources[].source_warnings` and in
hashed per-source review leaves. They do not certify every source assertion.

Full-page representation is not a claim that every image was transcribed.
The expanded visual review records two image-only numeric tables (Brooklyn
2024-03-27 physical page 16 and Detroit 2024-12-03 physical page 24) that remain
raw-preserved but unavailable as citation-supported numeric text. No OCR values
were admitted. Other new sparse flags are blank pages or a headshot roster.

Normalized postgames are separate at
`data/processed/bam_postgame/bam_v2_20261102_full`: 366 TRAIN records, 366
boxscores and 386 recap sources. `bam_postgame_audit.py` checks publication and
last-modification metadata at the actual three expanding research folds and the
final dev boundary. The final audit found no missing or late recap timestamps.
This is metadata evidence, not an archival proof: retrieval in 2026 does not
establish historical byte availability, and score/boxscore correction times
remain unproven. Raw ESPN payloads never become prediction inputs.
