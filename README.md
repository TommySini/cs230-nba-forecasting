# CS230 NBA forecasting: dataset pilot

A reproducible **data feasibility audit**, not a trained model or a final benchmark.
Uses free public source responses. No paid accounts, API keys, teacher-model calls,
or purchases are required for this pilot.

## Run

Python 3.9+ is required. Install the PDF dependencies in a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/collect_pilot.py
.venv/bin/python scripts/validate_pilot.py
.venv/bin/python -m unittest discover -s tests -v
```

Alternatively, with uv:

```bash
uv run --with-requirements requirements.txt python scripts/collect_pilot.py
```

Once collected, reproduce the outputs without network requests:

```bash
.venv/bin/python scripts/collect_pilot.py --offline
```

`--stage schedules` stops after building the historical game list and sample.
`--limit 3` runs a small debugging audit and **replaces the processed audit with
that subset**; rerun without a limit to restore the complete 100-game outputs.
The collector caches successful responses and permanent HTTP failures, verifies
cached checksums, throttles requests, and checkpoints processed files after each
game. To retry a cached HTTP failure, remove its `.meta.json` file first. Do not
remove successful cache entries if you need exact offline reproducibility.

## What is sampled

`config/pilot.json` selects 34 games from November 1–10, 2023 and 33 each from the
same window in 2024 and 2025, using seed 230. Regular-season schedules from October
1 through November 10 in each year provide the preceding results. Sampling is
independent of outcomes, document availability, and model performance.

This early-season sample is intentionally small. It does not establish archive
coverage for entire seasons. Before scaling, audit midseason and late-season
windows too. Select an actual model and verify its training cutoff before setting
the held-out evaluation period; these games are not yet contamination-cleared.

## Files

| Path | Purpose |
| --- | --- |
| `reports/pilot_findings.md` | Findings and next steps from the completed run |
| `reports/coverage.json` | Machine-readable coverage counts |
| `reports/game_audit.csv` | Per-game timestamp and source decisions |
| `data/raw/` | Downloaded source bytes and provenance sidecars |
| `data/processed/game_history.jsonl` | Historical game list, **including outcomes** |
| `data/processed/sample_manifest.jsonl` | Fixed audit sample, **including outcomes** |
| `data/processed/inputs.jsonl` | Pregame feature records and matched injury excerpts |
| `data/processed/labels.jsonl` | Target outcomes, separate from inputs |
| `data/processed/audit.jsonl` | All source decisions and attempted URLs |
| `data/processed/narrative_candidates/` | Unapproved team-note text; not model input |

Raw and processed data are git-ignored. The collector and reports can be shared
without accidentally publishing an archive of third-party content. Check source
permissions before redistribution or training on licensed narrative material.

## Sources and known limitations

**Game results:** ESPN's public daily scoreboard endpoint. It is undocumented and
may change. NBA's `leaguegamelog` endpoint timed out during the initial probe, so
the pilot uses ESPN. We retain only identity, historical scheduled start, final
scores, and season when normalizing games. Live/current news, rosters, standings,
leaders, odds, and target-game box scores never become model features. Historical
start times are taken from the retrieved schedule, not an archived pregame
schedule version; rescheduling must be audited for a final benchmark.

**Prior-game features:** calculated from earlier ET calendar dates in the same
season. Includes win rate, last-five point differential, and rest. Excluding
same-date games conservatively avoids assuming a game had finished just because
its scheduled start preceded another game's cutoff. This pilot does not collect
player-level box scores. Source game IDs allow every aggregate to be reproduced.

**Injuries:** official `ak-static.cms.nba.com/referee/injury/` PDFs. The internal
report timestamp, interpreted in America/New_York with daylight saving, must be
no later than scheduled tipoff minus 60 minutes. The URL filename alone is not
trusted. The parser respects game-date and matchup boundaries, including omitted
repeated times and page-spanning rows. Output is a game-specific text excerpt,
not a fully normalized player injury table. Missing team rows do not imply a
healthy roster; `NOT YET SUBMITTED` remains explicitly flagged.

The URL search probes at most four hourly slots and three filename formats per
slot, stopping after an eligible snapshot within an hour of the cutoff. It is
**not an exhaustive archive discovery or a guarantee of the latest snapshot**.
HTTP 403/404 means a candidate could not be fetched, not that no report exists.

**ESPN previews:** a historical URL labeled `preview` can return an article whose
type is `Recap`, alongside current news. The collector rejects non-preview
articles, wrong games, missing timestamps, and post-cutoff modifications. Even
eligible preview candidates require a content review and are not included by
default. Raw responses contain postgame material and must never be used directly
as model inputs.

**Team notes:** only a known dated Cavaliers PDF URL pattern is implemented in
this pilot. Other teams are explicitly marked `not_attempted_no_adapter`.
Successful PDFs are narrative candidates only. PDF creation/modification metadata
does not prove public availability before the cutoff. Validate the matchup,
publication timing, contents, and usable sections before promoting them to inputs.
This adapter's hit rate is not a leaguewide text-coverage estimate.

## Next collection decisions

1. Expand dated team-note discovery, using source links and archive indexes;
   inspect a midseason and late-season sample before backfilling entire years.
2. Establish publication evidence for narrative documents. Save upcoming sources
   before games for a prospective test set (no recurring collector is installed).
3. Apply through Stanford Libraries for licensed news/API access if public
   narratives remain sparse. Access has a waiting list; model-training permission
   needs to be checked with the library.
4. Add player statistics only if useful to the baseline. Add historical odds
   after coverage is established; that is a separate, optional paid step.
5. Manually audit samples against the original documents, choose the backbone,
   then freeze chronological splits and dataset versions.

Relevant source documentation:

- https://nba.balldontlie.io/
- https://github.com/swar/nba_api
- https://official.nba.com/nba-injury-report-2026-27-season/
- https://guides.library.stanford.edu/gearup/lexisnexis
- https://the-odds-api.com/liveapi/guides/v4/
