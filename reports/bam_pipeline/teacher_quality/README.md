# Teacher and reviewer quality benchmark

Status: the first 14B and 72B runs exposed substantial false acceptance; see
[the independent adjudication](baseline_adjudication.md). Source-first version 2
is prepared on the same cases and labels. Full-game report judgments remain
pending generated outputs.

The fixed support challenge contains **40 cases from 20 training games**:
12 existing Miami table/date/scope cases, 20 independently source-checked positive
claims spanning the diverse publisher reference, and 8 explicit contrasts for
wrong player/team, preseason versus regular season, ended streaks, stale injury
status, historical season context, conditional records mistaken for causality,
and incomplete sentence citations. There are 24 supported and 16 nonacceptance
labels. The other 40 proposed reference facts have not been silently promoted to
adjudicated gold.

Run the same `data/processed/bam_quality/teacher_reference_v1/requests.jsonl`
through both candidate models. Labels and adjudication notes live separately in
`expected.json`; they are never included in requests. The benchmark checks the
actual snapshot split, source hashes, archive cutoff and canonical citation lines.
No label files, postgame artifacts or target outcomes were used to select cases.
Prior-game results quoted inside verified pregame notes retain their dated scope.

The Qwen2.5-14B tokenizer measured 2,154–6,072 input tokens per request (median
4,213.5); requests reserve up to 1,200 output tokens. Source context includes the
complete cited pages, rather than only the candidate's potentially incomplete
citation. The model must assess the candidate citation as supplied and can propose
repair citations while returning `unclear` for incomplete support.

```bash
.venv/bin/python scripts/bam_quality.py prepare
.venv/bin/python scripts/bam_quality.py score --responses MODEL_RESPONSES.jsonl --model MODEL_ID --output MODEL_SCORE.json
.venv/bin/python scripts/bam_quality.py compare --scores FIRST_SCORE.json SECOND_SCORE.json --output COMPARISON.json
```

Response records use `request_id`, `model`, `text` (or `response`/`output`),
`finish_reason`, and optional `elapsed_seconds` or `latency_seconds`. Scoring checks
case identity, explicit verdict, rationale and real source citations. Missing,
invalid, truncated and duplicated responses never become correct rejections.
Report binary support accuracy, false acceptances, false exclusions and exact
three-way accuracy where an exact label was adjudicated. The 12 legacy labels
remain binary, matching the original challenge; `unclear` is nonacceptance, not
a claim of an exact `unsupported` judgment. Timing sums are per-response durations,
not concurrent batch wall time. Model comparisons require identical benchmark
hashes and case counts.

This purposive challenge contains correlated facts and contrasts. It is not a
random corpus factual-error estimate and does not establish forecast improvement.
A larger model may perform worse. Inspect individual false acceptances and false
exclusions alongside full-report coverage and interpretation quality.

`manual_review_set.json` selects three complete training games using publisher,
layout and source availability alone: Cleveland on December 27, 2024; Denver on
January 27, 2024; and Detroit on February 28, 2025. Expanded source packets are in
`data/processed/bam_quality/teacher_reference_v1/manual_review_games.jsonl`.
The rubric records factual support, entity/date/season correctness, citation
completeness, salient and opposing evidence coverage, basketball interpretation,
uncertainty, and exclusion of hindsight facts. Scores remain null until actual
selection/module/postmortem/gold artifacts are reviewed. A pending cell is never
a pass.

Validation: 14 focused benchmark tests pass. The research pipeline additionally
passes 39 tests, including repaired semantic-review parent binding.

Version 2 requests are in `data/processed/bam_quality/teacher_reference_v2/requests.jsonl`
with a separate `expected.json`. Rebuild only deliberately with
`bam_quality.py prepare-source-first`; the version 1 artifacts stay unchanged.
The v2 scorer requires source evidence and all six attribute comparisons, and
records attempted false acceptance separately when a self-contradictory answer
is rejected by the response validator.

Version 3 is frozen separately at
`data/processed/bam_quality/teacher_reference_v3/requests.jsonl`, with its own
`expected.json`. It keeps the same 40 claims, source pages, original candidate
citations and labels. Models identify source/page/line references and describe
the source reading; code supplies the verbatim source quotations. This removes
copy-format failures without treating source copying as semantic certification.
The comparison prompt explicitly requires numerical values together with units
and distinguishes the original candidate citation from a reviewer's repaired
citation. Version 1 and version 2 remain unchanged diagnostics.
