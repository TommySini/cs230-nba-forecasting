# Additional narrative source audit

Purposive, search-discovered source examples; not a random sample or a team/league coverage estimate.

No article is approved as model input. Retrospectively retrieved publisher timestamps are evidence to investigate, not proof of an unchanged pregame version.

| Team | Game | Published (UTC) | Modified (UTC) | Cutoff (UTC) | Decision |
| --- | --- | --- | --- | --- | --- |
| NYK | [2025-01-21 NYK@BKN](https://www.nba.com/knicks/news/game-notes-knicks-vs-nets-january-21-2025) | 2025-01-21T18:12:08Z | 2025-03-31T15:15:22Z | 2025-01-21T23:30:00Z | after_cutoff, home_team_missing_from_title_or_opening |
| NYK | [2025-01-25 SAC@NYK](https://www.nba.com/knicks/news/game-notes-knicks-vs-kings-january-25-2025) | 2025-01-25T15:25:41Z | 2025-03-31T15:12:33Z | 2025-01-25T23:30:00Z | after_cutoff |
| NYK | [2025-04-05 NYK@ATL](https://www.nba.com/knicks/news/knicks-vs-hawks-game-notes-april-5-2025) | 2025-04-04T20:50:36Z | 2025-04-07T15:58:42Z | 2025-04-05T18:00:00Z | after_cutoff |
| POR | [2025-04-03 POR@TOR](https://www.nba.com/blazers/news/game-notes-portland-trail-blazers-33-43-13-25-away-toronto-raptors-28-48-17-21-home) | 2025-04-03T16:37:29Z | 2025-04-03T16:37:29Z | 2025-04-03T22:30:00Z | candidate_requires_content_and_archive_review |

NYK 2025-01-21: Title names the Nets, but the opening paragraph describes Atlanta at Madison Square Garden.

## Archive discovery

- [ORL index](https://magicpr.wordpress.com/game-notes/): The inspected page links to rolling nba.com/gamenotes PDFs; it does not establish historical archive availability. Found 2 rolling PDF links.

## Next step

Expand the promising dated article source with a schedule-defined sample and independent pregame snapshot evidence. Review article-only text before promotion; keep rejected revisions quarantined. Rolling PDFs are candidates for prospective capture, not historical backfill.

Rebuild: `.venv/bin/python scripts/audit_narrative_sources.py --offline` after collecting the seasonal schedules.
