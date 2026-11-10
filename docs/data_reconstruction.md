# NBA data guide

The `data-reconstruction` branch implements shared Parquet tables for statistics,
notes and market observations. Model-specific copies are replaced by one dataset,
separate saved models, and compact run records.

## Downloading data after a clone

Install the `local` dependency group from `pyproject.toml` into the repo `.venv`, then run:

```bash
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/nba_dataset.py validate
.venv/bin/python scripts/validate_tabpfn.py --stage train-smoke
.venv/bin/python scripts/validate_tabpfn.py --stage dev
```

The default bundle contains all shared tables, audit metadata, saved boosting model
and required run records. The [release manifest](../config/dataset_release.json)
pins the dataset identity and every archive's URL, byte size and SHA-256 checksum.
Files are hosted on the versioned GitHub Release, outside Git history. No account
or token is needed for this public release.

Current release sizes (decimal units):

| Bundle | Download | Installed files | Archive parts |
| --- | ---: | ---: | ---: |
| Processed | 0.522 GB | 0.571 GB | 1 |
| Raw | 8.145 GB | 21.182 GB | 15 |

The installer temporarily holds extracted files before publication and keeps the
compressed cache for retries. Allow space for both during installation.

Add `--include-raw` for the original sources needed by the full boosting, notes,
Polymarket and BAM validators. Raw bodies and sidecars are installed directly in
the grouped `data/raw/{espn,nba,notes,polymarket}` layout. Their bytes remain
unchanged; the source resolver handles historical paths inside the sidecars.
Neural-network checkpoints and unrelated experiments are not in either bundle.
TabPFN inference still uses its separate `download-model` stage and dependencies.

Downloads are verified before extraction. The installer rejects unsafe archive
paths, reuses matching files, and stops on conflicts instead of overwriting them.
Verified archives are cached under `data/.downloads/<release>/` for retries.
`--assets-dir <directory>` installs the same verified archives offline. `--root`
selects an isolated checkout root for testing; it is not a data-format conversion.

The distribution manifest describes the grouped target layout even when raw files
have not been downloaded. `nba_dataset.py validate` verifies the processed files;
source-level validators require `--include-raw`. Downloaded releases do not contain
legacy folders, and do not need the local migration/cleanup commands below.

## Building a release

The maintainer builds from an already validated shared dataset:

```bash
.venv/bin/python scripts/nba_dataset.py validate
.venv/bin/python scripts/package_data.py --output data/releases/<new-release-tag> --release <new-release-tag>
```

The packager writes deterministic archives and rewrites the checked-in download
manifest only after every source checksum passes. It packages only the published
inventory, preserving source files and excluding unrelated work. Raw parts are
bounded below GitHub's per-asset limit. Portable manifest copies describe the
fresh-clone layout; the maintainer's original manifest is untouched.

Commit the code and download manifest together. Upload the verified archives to a
draft GitHub Release targeting that commit, verify uploaded asset sizes/digests,
then publish and test the public downloader. Published release assets are fixed by
the checked-in checksums; use a new tag and manifest for a changed dataset. A
different release should use a separate checkout/data directory.

## Current maintainer transition

The shared tables are published and all installed validators pass. The migration
recorded 383 tests with one optional tokenizer skip; release verification extends
the suite to 420 tests and checks an isolated checkout. Evidence and measurements
are in [`reports/data_reconstruction/validation.json`](../reports/data_reconstruction/validation.json).
The temporary conversion directory has been removed.
A separate TabPFN adaptation worktree references the original baseline paths, so raw
relocation and deletion of those paths are deferred. The existing source resolver
lets this branch read canonical source references while original raw directories
remain in place. Never remove its inputs while that job or other old readers need them.

Maintainer data remains under the canonical checkout's git-ignored `data/`; local
worktrees link to it. A Git branch isolates code, not this shared directory. Fresh
clones receive their own data directory through the downloader.

## Where things belong

```text
data/
  raw/
    espn/                 original schedules and summary responses
    nba/                  original replacement box-score responses
    notes/                original PDFs, HTML and archive-index responses
    polymarket/           downloaded markets, price history and trade pages
  processed/
    games.parquet
    features.parquet
    labels_train.parquet
    labels_dev.parquet
    labels_test.parquet
    box_score_games.parquet
    team_box_scores.parquet
    player_box_scores.parquet
    feature_lineage.parquet
    sources.parquet
    notes.parquet
    note_pages.parquet
    note_texts.parquet
    market_prices.parquet
    market_audit.parquet
    predictions.parquet
    manifest.json
    feature_dictionary.json
    bam_manifest.json
    note_audit.json
    box_score_audit.json
  models/                 trained models and downloaded weights
reports/                  findings, metrics and compact run records
```

The four raw groupings above are the final layout; while compatibility is needed,
original raw directory names remain. Original response bodies and sidecar bytes
are preserved exactly when regrouped. The resolver maps historical provenance paths
rather than rewriting downloaded evidence.

`processed` is organized by subject. Boosting, TabPFN and BAM read these shared
tables. A run records its selected IDs and data/model/code identities. There is no
need for another permanent copy of the feature matrix or source PDFs per model.

## What each table means

| Table | One row represents | Current scope |
| --- | --- | --- |
| `games` | A scheduled game, with identity, date, cutoff, split and eligibility | 16,829 source rows; `franchise_game` identifies the 16,821 NBA franchise games. The 8 exhibitions remain auditable but never enter model features. |
| `features` | An eligible game and its 252 numeric pregame features | 15,717 games; exactly the frozen values and missingness. |
| `labels_train/dev/test` | A game's result, including home-win label and final scores | All catalog outcomes, physically separated by split. Model loaders select eligible game IDs explicitly. |
| `feature_lineage` | A target game, home/away side, and earlier contributing game | 1,328,789 links; ordered histories and missing-box indicators. |
| `box_score_games` | Completed-game metadata such as duration and source identity | 16,810 games; kept outside the pregame catalog. |
| `team_box_scores` | One side's normalized completed-game statistics | 33,620 team-game rows. |
| `player_box_scores` | One player's normalized appearance | 356,205 rows. |
| `sources` | A normalized box score's provenance record | 16,810 entries; provider, original response reference, hash and retrieval evidence. Other source references live with notes and market observations. |
| `notes` | An admitted game/document link | 2,064 documents for 1,922 games; a selected corpus, not league-wide coverage. |
| `note_pages` | An admitted document page, including its exact citation lines | 53,294 pages; existing extracted text, without re-extracting PDFs. |
| `note_texts` | A review/extraction text record | Includes documents outside the admitted BAM corpus; preserves the original review evidence. |
| `market_prices` | A timestamped market observation or derived horizon observation | 9,640 rows across 2,410 games; `record_kind` distinguishes cutoff, tipoff and horizon records. |
| `market_audit` | A market-matching decision and its audit-only fields | 2,424 evaluated games, including unmatched decisions. |
| `predictions` | One model/run/stage's probability for one game | 16,850 initial rows: boosting plus TabPFN and its paired comparison forecasts. |

A scheduled game may lack features, a forecast or notes. These are different
coverage questions. The 1,104 feature exclusions are recorded in `games.exclusion`;
all have fewer than the required earlier games. The first TRAIN season has usable
features but no earlier-season boosting model, explaining another forecast gap.

The feature cohort is unchanged:

| Split | Seasons | Feature-eligible games |
| --- | --- | ---: |
| TRAIN | 2012–13 through 2023–24 | 13,457 |
| DEV | 2024–25 | 1,123 |
| TEST | 2025–26 | 1,137 |

## Understand one game

TRAIN game `espn:400277795` is Oklahoma City hosting Detroit on November 9, 2012
(ET). Its scheduled start is `2012-11-10T01:00:00Z`; its cutoff is one hour earlier.

| Feature | Value | Meaning at the cutoff |
| --- | ---: | --- |
| `home_previous_games` | 5 | Oklahoma City had played five earlier games that season. |
| `away_previous_games` | 5 | Detroit had played five. |
| `home_season_win_rate` | 0.6 | Oklahoma City had won three of those five. |
| `away_season_win_rate` | 0.0 | Detroit had won none. |
| `difference_season_win_rate` | 0.6 | Home win rate minus away win rate. |

Trace a statistic through `feature_lineage` to the earlier games, then the normalized
boxes and their `sources` references to the original responses. The target game's
outcome stays in its label table. Raw ESPN responses never become model prompts or
feature columns.

The generated `feature_dictionary.json` describes each column's position, meaning,
side, window, unit, formula, source fields and missingness rule. The 252 features
comprise 18 basic statistics, 144 team-statistic columns, 72 player/coverage columns,
and 18 team-coverage columns. Derived possession/rating statistics are estimates;
prior player workload does not establish a current lineup.

## Read the tables

Parquet is a binary table format: inspect it through a dataframe, SQL tool or table
viewer. The feature dictionary supplies the human-readable meanings.

Use this branch's `.venv`; dependencies are pinned in `pyproject.toml` (`local` group).
To inspect a few columns without loading the audit history:

```python
import pyarrow.parquet as pq

sample = pq.read_table(
    "data/processed/features.parquet",
    columns=["game_id", "home_previous_games", "home_season_win_rate"],
    filters=[("game_id", "=", "espn:400277795")],
)
print(sample.to_pylist())
```

Model code uses `scripts/nba_dataset.py`. `Dataset.inputs(splits={"train"})` returns
the exact original feature records in model order. `Dataset.labels({"train"})`
opens only TRAIN labels and defaults to the eligible cohort. `Dataset.history()`
is an outcome-side reconstruction API for audited feature rebuilding; it must not
be used by a pregame text consumer. BAM's shared input loader opens neither labels
nor completed-game boxes.

## Integrity and migration

The manifest pins schemas, ordered features, split policy, file hashes, a stable
input dataset identity, original manifests, and field-preservation checks. New
Parquet hashes identify new bytes; original hashes remain recorded as historical
evidence. Saved prediction cohorts also have their own logical fingerprints, so
appending an unrelated forecast cannot invalidate or silently alter an earlier run.
Prediction writes use a recoverable transaction; readers complete or roll back an
interrupted write before opening the tables.

The conversion preserves every model feature, null, label, forecast, history link,
exclusion, source admission decision and page/line citation. The original boosting,
TabPFN, notes, market and BAM validators passed before conversion. Replacement
validators check the new representation and replay the saved boosting models to
reproduce existing predictions. No model is retrained and no new experiment is run.

Historical box scores were retrieved retrospectively. Earlier ET game dates and
explicit availability timestamps, where present, guard feature construction. Exact
historical correction times remain unverified. Retrieval time is never relabeled
as pregame publication time. Notes and market prices retain their separate archived,
publisher and observation-time evidence.

The migration commands below apply to the maintainer's legacy data, not a downloaded
release. They are staged deliberately:

```bash
.venv/bin/python scripts/nba_dataset.py stage data/processed/.reconstruction-staging
.venv/bin/python scripts/nba_dataset.py validate data/processed/.reconstruction-staging
.venv/bin/python scripts/nba_dataset.py publish data/processed/.reconstruction-staging --keep-raw-layout
```

The compatibility publication leaves original paths intact for active consumers.
After those consumers have switched, `relocate-raw` moves sources with byte checks
and rollback support. Run the validators below against the installed layout before
`cleanup --validated`. Cleanup refuses unknown/changed files, removes only mapped
superseded records, and preserves other work such as the live TabPFN adaptation.
Do not use the cleanup flag as a substitute for the validation or dependency check.
The full unit suite below also requires the optional `tabpfn` dependency group; the source-level validators themselves need only the baseline environment.

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/validate_gradient_boosting.py
.venv/bin/python scripts/validate_tabpfn.py --stage train-smoke
.venv/bin/python scripts/validate_tabpfn.py --stage dev
.venv/bin/python scripts/validate_notes_v2.py
.venv/bin/python scripts/validate_polymarket.py
.venv/bin/python scripts/bam_data.py validate data/processed
```

Collectors refuse to overwrite the published shared dataset. Schedule and box-score
collection require an explicit empty output directory; market collection accepts
an unpublished output directory. Notes collection is guarded while the dataset is
frozen; expanding its reviewed corpus requires a separately staged collection. Unrelated historical pilot material and operational
budget records are outside the migrated model-input cohorts; they are not silently
deleted. Git history preserves code and checked-in reports, not git-ignored data.

## Measured storage and loading

The old feature-and-provenance JSONL is 262,281,139 bytes. The new float64 feature
table is 22,824,904 bytes; features plus game metadata and lineage total 24,573,477
bytes. The 16 shared Parquet tables total 527,930,700 bytes (about 503.5 MiB),
including both the admitted citation pages and broader document-review text.
Manifests, audit metadata and saved models are additional.

A local warm-cache comparison loaded the identical 15,717 × 252 NumPy matrix in
median 1.083 seconds from JSONL versus 0.0113 seconds by projecting the Parquet
feature columns (three timed reads after one warmup per format). This measures
matrix loading, not training or full audit reconstruction. Original formats and
new tables temporarily coexist during the compatibility window, so disk space has
not yet been reclaimed.

Format references: [Apache Parquet](https://parquet.apache.org/docs/overview/),
[PyArrow](https://arrow.apache.org/docs/python/parquet/parquet.html).
