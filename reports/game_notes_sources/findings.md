# Rich game-note source discovery

User clarified the target comparison: original boosting versus boosting probability plus game notes. The earlier neural pilot used injury excerpts only.

These are purposively selected source examples. **1 reviewed pregame excerpt** is now available for constructing the future hybrid cohort. Existing model inputs are unchanged; this is not a leaguewide coverage estimate.

| Source | Format | Pages | Decision |
| --- | --- | ---: | --- |
| [NYK 2026-01-19](https://www.nba.com/knicks/news/knicks-vs-mavericks-game-notes-january-19-2026) | Article | - | rejected |
| [NYK 2026-01-28](https://www.nba.com/knicks/news/knicks-vs-raptors-game-notes-january-28-2026) | Article | - | rejected_content_wrong_venue |
| [NYK rolling packet](https://www.nba.com/gamenotes/knicks.pdf) | PDF | 13 | rolling_current_version_not_historical_evidence |
| [CLE 2025-02-21](https://cdn.nba.com/teams/uploads/sites/1610612739/2025/02/250221-nykcle-notes.pdf) | PDF | 44 | approved_reviewed_pregame_excerpt |

## Historical availability checks

- NYK 2026-01-19: no_snapshot_returned; replay: no_pre_cutoff_replay_selected. Publisher modification: 2026-01-23T20:16:36Z.
- NYK 2026-01-28: pre_cutoff_snapshot_requires_bytes_review; replay: pre_cutoff_replay_content_requires_review. Publisher modification: 2026-01-28T18:27:40Z.
  Review: Pregame archive timing and article-text equality verified. Full article review found an opening that places the game at Madison Square Garden, while the joined schedule has New York visiting Toronto. It also has a stale Last game vs SAC heading above a Toronto recap. Keep this version quarantined; do not silently repair or use it.
- CLE 2025-02-21: pre_cutoff_snapshot_requires_bytes_review; replay: pre_cutoff_replay_content_requires_review. Publisher modification: Fri, 21 Feb 2025 15:37:40 GMT.
  Review: Rendered first page reviewed: February 21, 2025, New York at Cleveland, 7pm ET. Team records 45-10 and 37-18 match prior-only schedule history. Target game is listed with tipoff rather than a completed score. The identical PDF was independently captured at 21:37:43 UTC, before the 23:00 UTC prediction cutoff. Only the first-page excerpt is approved; other pages require section review before model use.

The live Knicks PDF is a reusable URL; it cannot stand in for historical game packets. The dated Cavaliers example provides a promising archive format. Existing 12 Cavaliers candidates and previously rejected Knicks article revisions remain separate. The 140 injury-covered games are not automatically a 140-game game-note dataset.

The first-page Cavaliers excerpt is pinned to exact archived bytes in `config/game_notes_reviews.json`; the other 43 pages remain outside the approved excerpt. The January 28 Knicks archive is genuinely pregame, but a manual review found a wrong venue and a stale heading, so it remains quarantined. No broad claim about Knicks coverage follows from these examples.

Next: expand pre-cutoff versions and review relevant sections, then count eligible train/dev/test games. The hybrid must be trained using out-of-sample boosting probabilities and evaluated against the unchanged booster on identical held-out games.

Reproduce: `.venv/bin/python scripts/audit_game_notes.py --offline`. Raw downloads, provenance, and quarantined extracted text are git-ignored. No model training is performed by this audit.
