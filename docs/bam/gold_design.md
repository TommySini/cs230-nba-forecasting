# BAM research and gold target implementation

Implemented by `scripts/bam_research.py`; this document describes the pipeline,
not completed training or evidence that research improves forecasts.

## Faithfulness and NBA adaptations

The BAM paper combines researcher outputs into forecasting context, then constructs
training gold reports from that pre-forecast context and a separate postmortem.
The teacher writes as of the forecast date: hindsight directs attention without
licensing future facts. Its postmortem includes general, timeline and market-view
analyses. Its researchers are prompted agents; directly training one shared NBA
research student across three roles is our adaptation. We preserve the same
researcher-context → separate postmortem → gold-report dependency, and project the
combined gold report into module targets. We do not reproduce its market-smoothed
quantitative supervision or claim exact replication of its finance system.
Source: [BAM §§4.1–4.2](https://arxiv.org/html/2607.09921v1).

## Interfaces and artifacts

`build_selection_requests(game, max_chunk_chars=18000)` visits every page and every
line, including explicit empty-page dispositions. Each request includes page
context, provenance and conservative code-decoded table facts. No fixed page or
claim-count cap discards the tail of a packet. Unrecognized or invalid tables
remain visible with a decoder failure; they do not become invented statistics.
The decoders support specific recent-player tables and explicitly headed
player-season and team-season average layouts. Split player names, sample counts,
season scope, column alignment and units are checked; other layouts remain
unknown. This is not blanket coverage of every publisher table. `bam_data.load_snapshot_game` expands hash-checked page references.

`build_module_requests(game, selection_artifacts, max_chunk_chars=18000)` requires
exact, nonduplicated chunk coverage and emits calls to the same research
student for three roles: availability/rotation, matchup, and recent changes/rest/workload. Each
finding separates its fact, date/scope, interpretation, counterevidence,
uncertainty, booster overlap and source/page/line citations. Module reports also
state unknowns and coverage limits. Original source context accompanies selected
findings, including nearby paragraphs and the nearest preceding player, season
and table headings; selection text is not treated as certification.

Large modules use deterministic `module_partial` groups.
`build_module_synthesis_requests` requires every sibling digest, then emits a final
`module` request or another bounded digest level. No finding is silently dropped
by a context-size operation; the model consolidates relevance and repetition,
with full raw candidates retained in the audit. At most four reduction levels
are allowed before an explicit failure. Callers supply an exact tokenizer through
`token_counter` and a 10,000-token input budget. The independent 48,000-character
guard is additional protection, not a token-count substitute. Without a tokenizer
only the character guard applies, and the backend must still enforce its token
budget. Final training inputs are these same deploy-time `module` prompts.

`build_postmortem_request(game, postgame)` accepts only a training game and
normalized, separately stored postgame sources. The safe record schema is
`{game_id, split:"train", normalized:true, sources:[...]}`. Every source has
`source_id,publisher,url,sha256,archive_time,pages`; every page has numbered text
lines. Raw API payloads are excluded by the caller's normalization and by the
builder's field allowlist. The postmortem is never an inference input.

`build_gold_request(game, reviewed_modules, postmortem)` requires all three
independently reviewed research outputs. The teacher receives those outputs,
original pregame citation context, code-decoded table facts, the optional booster
probability, and the separate postmortem. It produces one combined report containing
all three modules. Research-only training games may have no booster probability.
Gold targets are prohibited for development and test games.

Requests contain stage, model role, messages, maximum output tokens, schema,
source/input hashes and parent artifact hashes. Models and transport are selected
by the orchestrator. These builders make no remote calls and launch no jobs.

## Acceptance and repair

`ingest_response(request,response,game,model,finish_reason="stop")` retains the
raw response even on JSON/schema failure, truncation or invalid citations. Passing
mechanical checks yields `mechanically_valid`, never `approved`. For source
selection, individual invalid findings are retained in `rejected_findings` with
reasons while valid findings and explicit full-page coverage remain usable.
`build_selection_repair_request` can repair those citations without losing the
page. Module and gold reports remain strict: any invalid finding rejects the
whole candidate. Citation text is
copied from canonical sources, timestamps are checked, and fabricated numerical
facts, dense-table interpretations without an exact decoded fact, missing table
context, obvious future dates and outcome language are rejected. This is a
conservative guardrail, not a proof of factual entailment or complete leakage
removal. The independent semantic check remains necessary.

`build_review_request` gives a different named model the candidate and pregame
source context, with no postmortem or outcome. Original outcome-blind module/partial context is
also supplied, including selected evidence omitted by the report. Gold reviews
receive only an allowlisted pregame subset of the original gold request; the
training postmortem is excluded. This improves omission checks within supplied
research context, not completeness of the whole corpus. Each finding requires source evidence and six explicit attribute comparisons
before an explicit supported/unsupported/unclear verdict; self-contradictory
support decisions are rejected. Version 3 reviewers provide source references
and a source reading; code materializes the quoted lines. Literal model-copy
format is not used as a proxy for factual quality. the whole report also needs a verdict for
uncited prose, coverage and competing evidence. `apply_semantic_review` approves
only complete, unanimous support, and binds the review to the exact candidate
hash. A semantic model can still err; human reference-set evaluation remains a
separate research-quality requirement.

`build_repair_request` retains the original context, rejected raw output, reasons,
and semantic decisions. Missing citations may be repaired from the supplied
sources; unsupported claims must be removed or made unknown. Repairs have an
explicit retry budget. A repaired response must pass all checks again, including
a new independent review. No missing verdict is synthesized as approval.

## Training and forecasting boundaries

`export_student_examples` creates training-only chat examples from outcome-blind
module request messages plus the corresponding reviewed combined-gold module.
Rows preserve game/date/fold/module, source hashes, teacher hash and acceptance
status. Duplicate canonical quotations and URL metadata are stripped from target
text; source/page/line references remain. `export_forecast_explanation` emits the
combined approved report as a compact, training-only explanation target for the
optional joint forecaster loss. `export_student_inputs` creates the same pregame input without a target.
The forecasting worker must use actual generated student reports; a teacher-gold
report cannot substitute for one. `assemble_student_brief` requires approved
student-role module artifacts and refuses a game in that researcher's training
ID set. The training orchestrator additionally enforces chronological folds and
checkpoint eligibility; this helper cannot independently verify a model's entire
training history. The briefing interface requires an eligible prior-only booster
probability. Neither label nor postmortem is serialized into the brief.

There is no silent token truncation in this module. The backend must reject or
explicitly repair over-budget requests. Retaining complete context may be costly;
measured context sizes and rejection counts should accompany any experiment.
Missing opponent notes remain a documented coverage limit, not inferred evidence.

## Context-size check

A local Qwen2.5-14B tokenizer audit used all 107 existing packets and their saved
research findings as a sizing proxy. These are not results from the new teacher
or evidence of new report quality. Full raw packets had a median of **88,884**
tokens, 95th percentile **129,584**, and maximum **204,341**. Repeating whole cited
pages with existing findings was also unsuitable (median **83,995**, maximum
**254,892** tokens).

After retaining citation neighborhoods and nearest season/player/table headings,
the hierarchy emitted 711 initial requests for the same proxy (604 partial
requests). With the exact 10,000-token input guard, their median input length was
**9,226**, 95th percentile **9,942**, and maximum **10,000** tokens. The source
reports contained 30–150 findings per game (median 73). Actual counts will depend
on new selection and digest outputs; final target tokens must also fit the
training budget. The backend must reserve completion space and measure the
complete rendered training example rather than truncating it.

Validation: 39 focused BAM tests and 4 existing conservative table-decoder tests
pass. Tests cover train-only hindsight, source timing and drift, every-page and
every-digest coverage, citation context, numerical facts, missing semantic
verdicts, independent model identity, retained malformed outputs, repair budgets,
compact exports, and refusal of same-game-trained student reports.

## Compact production review after the training-only source audit

Production review schema version 4 replaces the repetitive six-attribute matrix with one `source_reading`, `required_citations`, `reason` and `verdict` per finding ID. The complete report still requires `report_reason` and `report_verdict`. Builders and driver interfaces are unchanged; this is a new immutable protocol, not another benchmark variant. Frozen V1/V2/V3 benchmark artifacts remain unchanged.

All original outcome-blind module context remains visible, deduplicated with candidate citation context. Gold-review context excludes the training-only postmortem. Required citations must reference source lines actually supplied to the reviewer. Code attaches canonical text and computes missing required lines using `(source_id, page, line)` identity. Extra original header/context citations are allowed. A supported model verdict with missing required original citations becomes an explicit effective `unclear` verdict; `model_verdict`, `citation_check`, and the raw response retain that decision and why it was withheld. No unsupported/unclear verdict is promoted. Evidence necessity, interpretation and entailment remain semantic judgments, not certified by containment.

HTML evidence now preserves `source_kind`, `pagination`, `page_kind`, `section` and available section `label` throughout selection, module evidence, review unions and canonical citations. Its `page` coordinate is a stable article-section identifier; PDF sources retain physical-page labeling. Normalized postgame records are labeled separately.

HTML body sections also carry the original article title and nearest preceding heading as `context_sections`. Each copied line must match its original section and line identifier exactly. Selection and review may cite those original sections, and module evidence includes them beside the body. A dated “Last Game” heading therefore remains visible without changing the paragraph or turning a historical absence into current availability. Sources without this optional metadata remain compatible; PDF behavior is unchanged.

An incomplete generation receives a bounded output expansion (selection 3,000 → 6,000 → 8,192 tokens) while preserving the partial response, original evidence, rejection parent and finish reason. An exhausted cap fails explicitly. The driver must still check the exact expanded prompt plus completion and reasoning reserve against model context; no evidence is silently truncated.

The four complete-report manual audit games are the original CLE, DEN and DET selections plus TRAIN MIA `espn:401704882`. `teacher_quality/manual_source_expectations_v2.json` contains 40 source-only expectations, with exact citations and review results still unset. The MIA packet provides an independent prose disambiguation of Herro's 52 regular-season versus 57 combined regular/postseason made-three games on page 4, lines 22–24. These expectations are targeted review coverage, not certified full-packet gold or a forecasting evaluation.

Selection findings now receive code-owned chunk ordinals; module and gold findings receive readable `<module>-f1`, `<module>-f2` IDs before review. Raw model IDs remain in raw-response artifacts. Review decisions are never renamed: missing, duplicate or foreign IDs fail validation. This removes bookkeeping-related evidence loss without accepting factual errors.

Selection also materializes every validated canonical table fact exactly once in its anchor chunk. Player season baselines route to availability/rotation, team baselines to matchup, and prior/recent-game facts to workload. Exact decoder text, scope, uncertainty and full citations are preserved; inference remains neutral until a model synthesizes it with other evidence. Original model table proposals and decoder provenance remain in `table_evidence_audit`, and paraphrase rejections remain in `rejected_findings`. Module aggregation deduplicates repeated decoded-table IDs. No arbitrary fact cap was added. Reprocessing the preserved stopped selection batch recovers all 25 canonical tables and 45 findings formerly rejected only for placeholder IDs, while retaining semantic errors for subsequent review rather than claiming they are correct.

## V5 guards and bounded repair after the failed four-game reference

The v4 reference failed before gold/export. Its automated review admitted concrete errors, including treating team totals as averages, invented milestone years and a reversed numerical comparison. The frozen artifacts remain diagnostic evidence, not usable gold targets.

The separate `bam_claim_guards.py` now recognizes short labeled dense TOTALS/OPPONENTS/AVERAGES rows, including comma-separated thousands. Such rows require exact validated decoder facts; no rate is inferred without a denominator. Non-null `fact_date` and explicit full dates in prose require cited full dates or an unambiguous cited consecutive season plus month/day. Prediction-year guesses are no longer allowed. A missing year can remain unresolved with null metadata and qualified month/day prose. Date presence does not establish the correct entity/event association. In an offline audit of 225 unique approved v4 findings, 104 had unsupported date metadata; 91 passed the mechanical checks if only that metadata was null. These are not 104 demonstrated false basketball facts.

A narrow literal-comparison check rejects explicit higher/lower comparisons whose stated scalar operands contradict their operator and have matching named metrics/units. It catches the actual Miami 109.9-versus-110.8 reversal. It deliberately skips different metrics/units, ranks, records, nested/multiple values and negation. It does not infer operands from 3-point terminology or assess threshold semantics. Arithmetic consistency does not establish source support.

Repairs reconstruct the exact original first two messages, then add only the latest compact candidate and compact per-finding/report feedback. Prior response/review hashes remain in lineage and all raw artifacts remain on disk. Rejected selection feedback now associates each reason with its candidate fact and citations. Truncated or malformed review responses remain audit-only and are not copied back into the prompt. Complete source evidence is never truncated. Reconstructed actual repair sizes fell from a largest 32,266-token accumulated prompt to 13,857 input tokens; this is a content-equivalent reconstruction, not an exact hash replay of the failed in-memory request. The driver still enforces its full context budget, including reasoning and completion.

Validation: 251 BAM tests passed after integration, including guards, citation/review binding, compact repair lineage and dispatch/cache regressions. No new model benchmark variant was introduced. Production adoption still requires actual complete-report source review; code checks and model approval alone do not certify gold quality.

The subsequent source audit added a deliberately limited precision guard for identical ordered lists of at least two named statistics: cited “averaging at least 20 points, six rebounds and eight assists” cannot support the same list asserted as exact averages. Preserving “at least” passes; separately cited exact prose or validated canonical table evidence can establish exact values. Different numbers, single metrics, other grammatical forms, entity association and ranks still require semantic review. On the unchanged 16 actual-claim inputs it flags only the demonstrated Cunningham threshold error. No benchmark request or expected label was changed.

Expanded sources may supply `source_warnings` as a list of audit cautions. Research requests preserve these under an explicit non-citable audit-metadata label. Canonical source lines remain the sole citation evidence; warnings cannot establish player identity, dates or availability. The latest focused and integrated suite passed 273 BAM tests, including qualifier preservation, independently supported exact averages, and warning/citation separation.

Page extraction `flags` and `method` now remain visible throughout evidence contexts. The driver may attach a hash-validated immutable sidecar entry as `page.coverage_audit` after validating the canonical snapshot. Its annotation changes the research source hash but never changes source lines or admits a transcription. Verified `numeric_image_untranscribed` pages produce explicit undecoded numeric-coverage records. Other sparse-page flags describe extraction uncertainty only; they do not establish that numbers are present. Suspicious-glyph flags remain visible without invalidating otherwise usable text automatically.

`page_coverage_gaps(game)` carries these limits through selection, every module hierarchy level, independent review and gold. Ingestion preserves raw model output and deterministically retains the coverage unknowns in validated selection/module/gold artifacts, even if a generated disposition calls the page irrelevant. A whole-report critic still evaluates any contradictory coverage claims. Read-only checks of BKN `espn:401585678` page 16 and DET `espn:401703421` page 24 verify that both yield a mandatory missing-numeric-coverage record and zero decoded numerical facts. Every page being accounted for does not mean every page was readable. The 235B probe remains independent and its requests are unchanged.

Actual normalized postgame inputs exposed a postmortem context overflow: DET required 33,471 tokens including its 3,500 completion and 4,096 reasoning allowance. Postmortem prompts now retain every canonical line ID/text and page metadata while removing repeated JSON-pointer paths from model input; those pointers remain in immutable normalized records. Four-game totals become 19,019–20,701 tokens. Validated postmortem artifacts still attach complete canonical quotations. Gold prompts compact those citations to source/page/line references while preserving every observation, foreseeable hypothesis, unforeseeable limitation, timeline entry and market-view statement. Unexpected postmortem fields are rejected rather than silently forwarded.

Gold completion space is 8,192 tokens. Real existing partial/final report combinations measured 6,823–7,644 compact output tokens, motivating the increase from 7,000. No complete three-module final set existed in v4, so combined gold and gold-review context measurements are explicitly sizing proxies. Their nominal totals reach 42,821 and 44,970 tokens respectively; actual context must be checked after generation. A larger model context also requires corresponding driver review-budget and helper-context settings. The system must fail explicitly on an over-budget complete request instead of truncating source evidence. The integrated suite passed 287 BAM tests after these changes.
