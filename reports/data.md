# Data coverage, provenance and eligibility

The statistical history is broad; the text corpus is selective. The frozen BAM
snapshot contains **1,922 games with pregame notes**, of which **1,768 have an
eligible statistical forecast**. These games support an exploratory comparison
of statistics with text, but they do not represent league-wide availability of
game notes. The distinction matters: a strong result on this corpus would not
automatically transfer to every NBA game.

## From collected material to usable forecasts

The project preserves three different populations. Collection approval identifies
candidate documents under the collector's timestamp and content rules. Snapshot
construction applies stricter revision checks and binds extracted pages to their
sources. Forecast eligibility additionally requires enough prior-game statistics
and a prediction from a model that did not fit that game.

| Stage | Documents | Games | Meaning |
| --- | ---: | ---: | --- |
| Approved notes collection | 2,144 | 1,997 | Archived PDFs, dated PDFs and publisher-timestamped HTML passing collection checks |
| Frozen `bam_notes_v2` snapshot | 2,064 | 1,922 | Excludes 80 HTML revisions modified after their prediction cutoff |
| Forecast-eligible snapshot games | — | 1,768 | Have both accepted notes and an eligible boosting probability |

Removing 80 document versions removes 75 games entirely; some games still have
another eligible document. Of the 154 snapshot games without a forecast, 138
lack sufficient earlier games and 16 belong to the first TRAIN season, for which
no earlier-season model exists. Their notes remain available for research.
The shared dataset's [note audit](../data/processed/note_audit.json) (collection
`review_summary`) and [BAM manifest](../data/processed/bam_manifest.json) contain the
counts and provenance. No report-side copy of either is needed.

| Current role | Seasons | Snapshot games | Forecast-eligible games |
| --- | --- | ---: | ---: |
| TRAIN | 2012–13 through 2023–24 | 1,592 | 1,456 |
| DEV | 2024–25 | 193 | 182 |
| TEST | 2025–26 | 137 | 130 |
| Total | | 1,922 | 1,768 |

The split is defined once in [the baseline configuration](../config/gradient_boosting.json).
The earlier proposed union of 1,980 games was a provisional audit population;
it was not the snapshot ultimately frozen. Historical candidate counts should
not be added to the current cohort.

## Statistical history and its limits

The schedule cache contains 16,829 historical records. Eight non-franchise
exhibitions are excluded, leaving 16,821 candidate games for box scores. Valid
team box scores exist for 16,810, including 2,104 repaired from NBA.com; 16,123
also have complete player statistics. Eleven games remain missing with recorded
reasons. These counts describe the cached periods, not a guarantee of every game
in every season: the recent collection ends April 10. See the
[box-score audit](../data/processed/box_score_audit.json) (`failures`, `source_repairs`).

Features use earlier Eastern-calendar dates within the same season. The target
game and other games on its date cannot enter its rolling statistics. Eligibility
requires both teams to have at least five prior games and five valid prior box
scores. Missing team observations reduce window coverage; incomplete player
windows stay missing rather than implying zero workload. The baseline report
explains the resulting modeling cohort.

The source bytes were retrieved retrospectively. Their checksums establish which
records were used, but cannot establish when every historical correction first
became public. Known completion/availability timestamps are enforced where
present. Possessions and ratings are estimates from box scores, not official
advanced statistics. A small known rebound inconsistency remains in seven 2020–21
ESPN team rows; the frozen dataset has not been silently repaired again.

## What “pregame” establishes

Every game's prediction cutoff is scheduled tipoff minus 60 minutes. Evidence has
two distinct strengths:

- **Archive-proven:** an archived capture of the source predates the cutoff.
- **Publisher-timestamped:** the publisher's timestamp supports pre-cutoff
  availability, without the same independent capture evidence.

The snapshot contains 1,634 archive-proven and 430 publisher-timestamped
documents. An early article publication date does not prove that a later-edited
body was available then; this is why the snapshot excludes the 80 explicitly late
HTML revisions even though the collector had approved their publication times.

Snapshot construction reconstructs selected content from exact raw bytes and
records hashes for sources and extracted pages. PDFs retain physical-page
coordinates; HTML uses article sections. The source header and result screen
reduce obvious mismatches and postgame contamination. They do not certify every
sentence, decoded table or model interpretation. Outcomes and teacher postmortems
stay separate from pregame inputs. Raw ESPN responses, which can contain recaps
and current news, are never model inputs.

## Coverage is selected, not representative

Discovery depends on surviving archive captures, readable packets and publisher
practices. In the collected corpus before the stricter snapshot filter, Cleveland
supplies 271 documents, Boston 229, Detroit 201, Denver 197 and Memphis 171: almost
half of all documents come from these five publishers. Coverage varies across
seasons and shifts from older archived PDFs toward recent publisher-timestamped
sources. A missing packet means no accepted source was recovered, not that a team
published nothing or that no injury existed.

The initial 100-game feasibility audit established usable injury excerpts but
also showed that an ESPN “preview” route could return postgame material. Later
archive research expanded discovery through rolling PDFs, URL variants and dated
articles. Those searches established collection routes and candidate leads;
index matches were not automatically accepted documents. Their detailed records
remain at Git revision `67596558a188014803724bd60a158ce6f205ec3e`, under
`reports/archive_coverage/`, `reports/past/dataset_pilot/` and
`reports/bam_pipeline/notes_v2_review_20261103/`.

The 2025–26 season has already been inspected in earlier experiments. Its current
TEST label identifies the fixed partition, not a newly untouched benchmark.
Future text comparisons must use identical eligible games across models and
report source/publisher coverage alongside performance.

## Verify without rebuilding

With the canonical `data/` available, these commands read and validate the frozen
sources and snapshot; they do not collect data or fit models:

```bash
.venv/bin/python scripts/validate_notes_v2.py
.venv/bin/python scripts/bam_data.py validate data/processed/bam_pipeline/bam_notes_v2
```

The first checks the collection's policy; the second checks the stricter frozen
snapshot. A pass is an integrity check within those policies, not a claim of
complete historical availability or perfect semantic interpretation.
