# Research quality audit

Model and prompt choices use source-reading accuracy on the shortest **training**
packet, Miami at Boston on December 2, 2024. Target-game outcomes and forecast
scores are not used. The manual reference is
[the full-packet worked example](../full_game_notes/worked_example.md).

## Rejected trial configurations

- **7B, line citations:** confused a streak of games with consecutive made shots
  and treated conditional scoring splits as current facts.
- **7B, quoted evidence:** frequently paraphrased rather than copied evidence,
  misread tables and invented current absence claims. Strict source checks
  rejected those responses; the trial was stopped.
- **14B, line citations and excerpt-only support checking:** completed all 15
  pages and extracted 73 claims. Its support checker rejected 15, but manual
  review found remaining errors: older game-log rows described as the latest
  game, and previous DNP-CD entries described as current status. The checker
  could not reliably detect a later row that was absent from its excerpt.

These trial artifacts remain under `data/probes/research_smoke_v1`, `v2` and
`v3`. None has been used to train the revised forecaster. The original trained
baseline and first-page pilot remain unchanged.

A **32B full-page trial** still confused field goals made with points and accepted
incorrect averages during its checking pass. For example, it called Bam
Adebayo's five made field goals on December 1 five points, although the table
records 13 points. This configuration was also rejected; artifacts remain in
`data/probes/research_smoke_v4`. Larger size alone did not fix the table-reading
problem.

## Revised quality gate

The checker now receives the entire source page portion, including uncited
later rows, as well as each proposed claim and its cited line numbers. Citations
are copied by code from the preserved source text. The full corpus is processed
page by page, with oversized pages split without dropping any source lines.

`game_note_tables.py` now decodes known player-table headers, verifies field
counts, dates, `PTS = 2*FGM + 3FGM + FTM` and rebound totals, then computes means
in code. Its present adapter covers 30 tables in two Miami packets, producing
51 canonical facts. This is not leaguewide table-format coverage. LM
interpretations of dense statistical rows are excluded; unrecognized tables
remain in the source audit. Novel numerical values absent from cited prose are
also excluded. All source lines still reach the research stage.

`scripts/check_research_quality.py` runs the 15-page training packet and twelve
targeted checks: eight misleading season/recency/current-injury/column statements
must be rejected; three exact December 1 player-minute facts and the road
back-to-back fact must be accepted. Challenge statements are stored separately
from actual research inputs. The checkpoint predates every game in the cohort.

This gate is a targeted regression check, not a measurement of factual accuracy
across the corpus. A second generation from the same model can share the first
generation's errors. Source citations establish traceability, not correctness
by themselves. All accepted and rejected claims remain available for inspection.

## What passed and what failed

The selected September 18, 2024 four-bit Qwen2.5-14B checkpoint passed **10 of
12 semantic challenges**, not all twelve. The checker incorrectly accepted a
52-game three-point streak assigned to Duncan Robinson rather than Tyler Herro,
and the statement that Bam Adebayo scored five rather than thirteen points in
the December 1 game. The raw responses and failures remain visible in
[quality_gate.json](quality_gate.json). This is evidence that same-model checking
cannot certify generated factual summaries.

The forecast input design therefore uses **the actual cited source lines**, with
two neighboring lines and available player/season headings, instead of any
model-generated paraphrase, temporal scope or caveat. The model still selects
passages; selection can omit relevant evidence. The frozen encoder can also
misinterpret correctly copied text. Quotations establish faithful input and
traceability, not perfect semantic understanding or predictive value.

Code-derived table facts enter directly. In addition to the Miami adapter,
`cavaliers_note_tables.py` decodes 833 player tables across 49 Cleveland packets,
producing 1,982 facts after column, date, scoring and rebound checks. Blank
participation details in 248 tables remain explicitly unknown; no current
absence is inferred. Neither adapter proves coverage of every statistical
format in the corpus.

The separate training-readiness checks pass: canonical Miami table values and
arithmetic agree with the manual reference; an adversarial invented summary,
scope and caveat cannot enter the forecast prompt; the actual source quotations
and surrounding context are preserved. `ready_for_source_evidence_run` refers
only to this constrained input design. `semantic_checker_all_passed` remains
**false**. This change was made before revised forecast training or evaluation.

## Current execution

Full-corpus research is running locally over all 107 packets: 3,783 physical
pages, 3,858 page portions and 11,570,203 initial extraction-prompt tokens.
Every nonblank source line is supplied; the researcher curates up to eight
prose facts per portion, so full source processing is distinct from forwarding
every sentence to the forecaster. All retained evidence is subsequently encoded
across complete chunks without a first-page or first-N-item cutoff.

Two malformed page responses initially stopped the run. Invalid individual
claims are now retained as rejected audit items, and a bare JSON array of valid
support decisions is normalized to the requested object wrapper without changing
its decisions. Original raw generations, request hashes and generation-protocol
hashes are preserved. Compatible responses are revalidated rather than relabeled
as newly generated. See [progress.json](progress.json) for completed packets.

The forecast-input preflight on the first three completed Cleveland training
packets includes every retained item and all added code-derived table facts in
three chunks per game. The full revised head has not yet been trained. Its
80/13/14 train/dev/test membership is unchanged; the old test is explicitly an
exploratory repeated comparison, not a fresh final benchmark.

## Completed corpus provenance audit

The full research run completed all 107 packets, 3,783 PDF pages and 3,858 page
portions. The independent source/response validator passed complete line coverage,
raw-generation parsing, citation copying, support-decision coverage, checkpoint
identity and imported execution provenance. There were 17,100 proposed claims:
8,112 received a supported verdict, 7,763 remained unclear, and 1,225 were rejected
as unsupported. Deduplication left 8,092 retained evidence items. These counts
measure the checker’s decisions, **not** verified factual accuracy.

The unchanged forecaster preparation added 1,982 independently decoded Cleveland
table facts, giving 10,074 total evidence items in 352 complete forecast chunks
(1,749,153 tokens across all games). No generated prose summaries enter the
forecast inputs. Every retained source-evidence item is covered. The original
10/12 semantic challenge result and its two known failures remain unchanged.
See [validation.json](validation.json) and the
[exact forecast evidence index](../researched_forecaster/evidence_index.md).
