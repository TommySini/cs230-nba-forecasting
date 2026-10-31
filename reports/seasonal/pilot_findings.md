# NBA dataset pilot findings

This is a feasibility audit, not a training-ready text dataset or a forecasting benchmark.

- Sample: **60 games**, **30 teams**. January 15-24 and April 1-10 in 2024, 2025 and 2026; 10 deterministic games per window, not a train/test benchmark.
- Historical results retained for prior-game features: **3621**.
- Matched injury excerpts with internal timestamps at or before the cutoff: **60/60**.
- Matched reports containing a NOT YET SUBMITTED entry: **0**.
- Purchases and paid API calls: **none**.

## Coverage by season

| Season | Games | Matched injuries |
| --- | ---: | ---: |
| 2023–24 | 20 | 20 |
| 2024–25 | 20 | 20 |
| 2025–26 | 20 | 20 |

## Coverage by window

| Season | Window | Dates | Games | Matched injuries |
| --- | --- | --- | ---: | ---: |
| 2023 | midseason | 2024-01-15 to 2024-01-24 | 10 | 10 |
| 2023 | late_season | 2024-04-01 to 2024-04-10 | 10 | 10 |
| 2024 | midseason | 2025-01-15 to 2025-01-24 | 10 | 10 |
| 2024 | late_season | 2025-04-01 to 2025-04-10 | 10 | 10 |
| 2025 | midseason | 2026-01-15 to 2026-01-24 | 10 | 10 |
| 2025 | late_season | 2026-04-01 to 2026-04-10 | 10 | 10 |

## Narrative audit

ESPN preview-route results:

- `rejected_not_preview`: 60

A preview URL is not evidence of pregame content. The responses are checked using the article's actual type, game ID, publication time and modification time. No rejected article or surrounding current-news feed enters the inputs.

Dated Cavaliers PDF results (only this team's archive pattern was tested):

- `not_attempted_no_adapter`: 55
- `candidate_requires_matchup_and_publication_audit`: 5

Downloaded narrative candidates:

| Game | PDF pages | Source |
| --- | ---: | --- |
| 2024-01-20 CLE@ATL | 40 | [Original PDF](https://cdn.nba.com/teams/uploads/sites/1610612739/2024/01/240120-cleatl-notes.pdf) |
| 2024-01-24 CLE@MIL | 40 | [Original PDF](https://cdn.nba.com/teams/uploads/sites/1610612739/2024/01/240124-clemil-notes.pdf) |
| 2025-01-20 PHX@CLE | 43 | [Original PDF](https://cdn.nba.com/teams/uploads/sites/1610612739/2025/01/250120-phxcle-notes.pdf) |
| 2025-04-04 CLE@SAS | 45 | [Original PDF](https://cdn.nba.com/teams/uploads/sites/1610612739/2025/04/250404-clesas-notes.pdf) |
| 2026-04-10 CLE@ATL | 51 | [Original PDF](https://cdn.nba.com/teams/uploads/sites/1610612739/2026/04/260410-cleatl-notes.pdf) |

These PDFs are stored separately pending matchup/content and publication-time review. PDF metadata alone is not proof of pregame public availability. No narrative candidates are approved as model inputs yet.

## What the files can support now

The pilot provides final-outcome labels, prior-game team features, and timestamp-filtered injury text. Inputs and labels are separate JSONL files. Every injury excerpt links to raw source bytes with a checksum and retrieval metadata. This is sufficient to prototype the loader and a structured-data baseline; it does not yet establish the rich-text dataset proposed for the LLM experiment.

## Limits and next actions

1. These bounded windows do not establish full-season coverage; compare the early-season and seasonal audits before expanding.
2. Expand dated team-note discovery beyond Cleveland. The unattempted teams are not counted as missing archives.
3. Verify narrative publication timing and extract relevant sections; keep unverified documents quarantined.
4. Choose the backbone and ensure the held-out games are after its training cutoff. No train/validation/test split is assigned here.
5. Injury URL probing is bounded and may not locate the latest available report. Missing candidates would not establish absence of an archive.
6. Historical scheduled starts are retrieved retrospectively; rescheduled games need a final timing audit.
7. No paid odds, player-level box scores, teacher-generated targets, or recurring future-game capture have been added.

## Reproduce and validate

```bash
.venv/bin/python scripts/collect_pilot.py --config config/seasonal.json --offline
.venv/bin/python scripts/validate_pilot.py --config config/seasonal.json
.venv/bin/python -m unittest discover -s tests -v
```

See `game_audit.csv` for per-game decisions, `coverage.json` for counts, and the repository README for source and schema details.
