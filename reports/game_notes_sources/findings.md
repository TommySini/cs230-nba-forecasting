# Rich game-note source discovery

User clarified the target comparison: original boosting versus boosting probability plus game notes. The earlier neural pilot used injury excerpts only.

These are purposively selected source examples. No new narrative has been approved as input; this is not a leaguewide coverage estimate.

| Source | Format | Pages | Decision |
| --- | --- | ---: | --- |
| [NYK 2026-01-19](https://www.nba.com/knicks/news/knicks-vs-mavericks-game-notes-january-19-2026) | Article | - | rejected |
| [NYK 2026-01-28](https://www.nba.com/knicks/news/knicks-vs-raptors-game-notes-january-28-2026) | Article | - | candidate_requires_content_and_archive_review |
| [NYK rolling packet](https://www.nba.com/gamenotes/knicks.pdf) | PDF | 13 | rolling_current_version_not_historical_evidence |
| [CLE 2025-02-21](https://cdn.nba.com/teams/uploads/sites/1610612739/2025/02/250221-nykcle-notes.pdf) | PDF | 44 | candidate_requires_content_and_publication_review |

## Historical availability checks

- NYK 2026-01-19: no_snapshot_returned; replay: no_pre_cutoff_replay_selected. Publisher modification: 2026-01-23T20:16:36Z.
- NYK 2026-01-28: pre_cutoff_snapshot_requires_bytes_review; replay: archived_article_parse_failed. Publisher modification: 2026-01-28T18:27:40Z.
- CLE 2025-02-21: pre_cutoff_snapshot_requires_bytes_review; replay: pre_cutoff_replay_content_requires_review. Publisher modification: Fri, 21 Feb 2025 15:37:40 GMT.

The live Knicks PDF is a reusable URL; it cannot stand in for historical game packets. The dated Cavaliers example provides a promising archive format. Existing 12 Cavaliers candidates and previously rejected Knicks article revisions remain separate. The 140 injury-covered games are not automatically a 140-game game-note dataset.

Next: establish pre-cutoff versions and review relevant sections, then count eligible train/dev/test games. The hybrid must be trained using out-of-sample boosting probabilities and evaluated against the unchanged booster on identical held-out games.

Reproduce: `.venv/bin/python scripts/audit_game_notes.py --offline`. Raw downloads, provenance, and quarantined extracted text are git-ignored. No model training is performed by this audit.
