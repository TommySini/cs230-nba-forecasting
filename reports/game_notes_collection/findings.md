# Expanded historical game-note audit

The audit checked the dated Cavaliers archive pattern and rolling game-note URLs
for all 30 NBA teams against the cached 2023–24 through 2025–26 regular-season
schedule. It preserved **804 candidate packet versions**. **482 versions covering
404 distinct games** have verified archived bytes captured before tipoff minus
60 minutes. These counts describe observed source availability, not representative
leaguewide coverage.

Content review retained **256 games across 19 note publishers**. The collection
stage itself never approves model inputs; its `approved_for_model: 0` field refers
to that stage. [The content review](content_review.json) records the subsequent
approvals and rejection reasons. Multiple packets for one game are resolved by a
fixed rule: prefer more extractable front-page words, then publisher and hash.
No outcome or model result is used to select a packet.

| Season | Approved distinct games | Used in hybrid experiment |
| --- | ---: | ---: |
| 2023–24 | 149 | 0 |
| 2024–25 | 76 | 76 |
| 2025–26 | 31 | 31 |

The hybrid cohort uses only games after the pinned September 2024 language-model
checkpoint. Its **107 games** are split into **80 train / 13 dev / 14 test** on
whole ET dates. Original-booster fitting IDs are excluded from hybrid dev and
test. See [the frozen protocol](../boosting_notes/protocol.json).

## What was verified

Every selected source retains the publisher URL, actual archive replay URL,
Memento capture time, retrieval time, raw PDF and SHA-256 checksum. The actual
replay must match the intended URL and precede the game cutoff. A dated filename,
PDF creation date or retrospectively fetched Last-Modified header is insufficient.
Transient connection failures are preserved and retried; an unavailable lookup is
not evidence that no historical packet exists. Philadelphia's index lookup
remained unavailable in this collection.

The availability API returns the nearest capture, which can be after the cutoff.
For dated URLs with such a result, a bounded CDX query also checked the preceding
two days. None of those fallback queries produced an additional eligible capture
in this run; failed queries remain explicitly recorded.

Front-page layouts were visually reviewed for 19 publishers, with source hashes
and page numbers in `config/game_notes_content_review.json`. Each selected page
must also pass mechanical matchup, date, season, home/away and prior-record or
game-number checks. A Chicago packet naming United Center for a game at Cleveland
was manually excluded. Missing text extraction, stale/mismatched headers,
inconsistent prior records and unreviewed templates remain excluded.

This is template review plus per-document checks, **not manual fact-checking of
every sentence**. Full packets remain preserved. Model evidence consists of the
front page, with schedule rows removed and a fixed source-order 2,700-token
budget. Selected lengths are recorded. The material includes player form,
lineup/status information, matchup history and additional numerical statistics;
it is broader than the earlier injury-report excerpts.

## Limits

Archive availability, PDF extraction and reviewable templates determine coverage.
The 14-game test cohort uses Cavaliers notes for 11 games, with one packet each
from Brooklyn, Boston and Memphis. Results cannot establish leaguewide effects.
The original test season was inspected in earlier experiments, so this remains
an exploratory benchmark. A comparison of boosting with the full correction
system does not isolate narrative information from recalibration or added player
statistics.

The original statistics-only model and the previous injury-text experiment are
preserved. New raw sources, processed excerpts and models are local and git-ignored
in the current worktree.

## Reproduce

```bash
.venv/bin/python scripts/collect_game_notes.py --offline
.venv/bin/python scripts/review_game_notes.py
.venv/bin/python scripts/run_boosting_notes.py --stage prepare
.venv/bin/python scripts/run_boosting_notes.py --stage train --device mps
.venv/bin/python scripts/run_boosting_notes.py --stage evaluate --device mps
.venv/bin/python scripts/validate_boosting_notes.py --device mps
```

The initial online collection omits `--offline`. For a completed experiment,
the validator replays saved artifacts without retraining. Source/configuration
changes invalidate the frozen protocol and require preparation and training again.

Sources: [NBA dated game-note example](https://cdn.nba.com/teams/uploads/sites/1610612739/2025/02/250221-nykcle-notes.pdf),
[official rolling notes example](https://www.nba.com/gamenotes/knicks.pdf),
[Internet Archive availability API](https://archive.org/help/wayback_api.php).
