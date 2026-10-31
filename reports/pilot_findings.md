# NBA dataset pilot findings

This is a feasibility audit, not a training-ready text dataset or a forecasting benchmark.

- Sample: **100 games**, **30 teams**. November 1-10 in each configured season; early-season feasibility sample, not a train/test benchmark.
- Historical results retained for prior-game features: **435**.
- Matched injury excerpts with internal timestamps at or before the cutoff: **100/100**.
- Matched reports containing a NOT YET SUBMITTED entry: **0**.
- Purchases and paid API calls: **none**.

## Coverage by season

| Season | Games | Matched injuries |
| --- | ---: | ---: |
| 2023–24 | 34 | 34 |
| 2024–25 | 33 | 33 |
| 2025–26 | 33 | 33 |

## Coverage by window

| Season | Window | Dates | Games | Matched injuries |
| --- | --- | --- | ---: | ---: |
| 2023 | early_season | 2023-11-01 to 2023-11-10 | 34 | 34 |
| 2024 | early_season | 2024-11-01 to 2024-11-10 | 33 | 33 |
| 2025 | early_season | 2025-11-01 to 2025-11-10 | 33 | 33 |

## Narrative audit

ESPN preview-route results:

- `rejected_not_preview`: 100

A preview URL is not evidence of pregame content. The responses are checked using the article's actual type, game ID, publication time and modification time. No rejected article or surrounding current-news feed enters the inputs.

Dated Cavaliers PDF results (only this team's archive pattern was tested):

- `not_attempted_no_adapter`: 93
- `candidate_requires_matchup_and_publication_audit`: 7

Downloaded narrative candidates:

| Game | PDF pages | Source |
| --- | ---: | --- |
| 2023-11-08 CLE@OKC | 39 | [Original PDF](https://cdn.nba.com/teams/uploads/sites/1610612739/2023/11/231108-cleokc-notes.pdf) |
| 2024-11-06 CLE@NOP | 34 | [Original PDF](https://cdn.nba.com/teams/uploads/sites/1610612739/2024/11/241106-clenop-notes.pdf) |
| 2024-11-08 GSW@CLE | 36 | [Original PDF](https://cdn.nba.com/teams/uploads/sites/1610612739/2024/11/241108-gswcle-notes.pdf) |
| 2025-11-02 ATL@CLE | 52 | [Original PDF](https://cdn.nba.com/teams/uploads/sites/1610612739/2025/11/251102-atlcle-notes.pdf) |
| 2025-11-05 PHI@CLE | 36 | [Original PDF](https://cdn.nba.com/teams/uploads/sites/1610612739/2025/11/251105-phicle-notes.pdf) |
| 2025-11-07 CLE@WAS | 37 | [Original PDF](https://cdn.nba.com/teams/uploads/sites/1610612739/2025/11/251107-clewas-notes.pdf) |
| 2025-11-10 CLE@MIA | 37 | [Original PDF](https://cdn.nba.com/teams/uploads/sites/1610612739/2025/11/251110-clemia-notes.pdf) |

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
.venv/bin/python scripts/collect_pilot.py --config config/pilot.json --offline
.venv/bin/python scripts/validate_pilot.py --config config/pilot.json
.venv/bin/python -m unittest discover -s tests -v
```

See `game_audit.csv` for per-game decisions, `coverage.json` for counts, and the repository README for source and schema details.
