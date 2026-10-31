# Source document spot checks

Inspected rendered source pages on 2026-10-30. These are targeted parser checks,
not a random estimate of error rates or a substitute for the complete validator.

- **2024-01-15 ORL@NYK, ESPN 401585185:** page 3 of the
  [official injury PDF](https://ak-static.cms.nba.com/referee/injury/Injury-Report_2024-01-15_01PM.pdf)
  separates the target from the preceding Detroit/Washington continuation and the
  following San Antonio/Atlanta matchup. Both target teams have rows. Its internal
  1:30 PM EST timestamp is 18:30 UTC, before the 19:00 UTC prediction cutoff.
  The filename's `01PM` is not the internal report time.
- **2024-04-01 BOS@CHA, ESPN 401585715:** pages 1–2 of the
  [official injury PDF](https://ak-static.cms.nba.com/referee/injury/Injury-Report_2024-04-01_05PM.pdf)
  show Boston's rows continuing onto page 2, followed by Charlotte, and ending
  before Memphis/Detroit. The internal 5:30 PM EDT timestamp is 21:30 UTC,
  before the 22:00 UTC prediction cutoff. This checks a real page continuation
  and daylight saving conversion.
- **2024-01-20 CLE@ATL:** page 1 of the
  [dated Cavaliers notes](https://cdn.nba.com/teams/uploads/sites/1610612739/2024/01/240120-cleatl-notes.pdf)
  has the intended matchup, date, venue and 7:30 PM EST start. This confirms only
  the first-page identity; it does not establish publication timing or approve
  the full document as input.

Raw byte paths and SHA-256 checksums are in the seasonal `audit.jsonl` and
`inputs.jsonl`. All narrative documents remain excluded from model inputs.
