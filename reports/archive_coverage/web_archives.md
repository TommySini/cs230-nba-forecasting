# Web-archive coverage of NBA team game notes (pre-cutoff captures)

Research date: 2026-11-01. Scope: count archived captures of team media-notes packets that a
third-party archive recorded before each game's prediction cutoff (scheduled tip minus 60 min).
Research only. Scratch scripts and raw query results are kept locally (git-ignored) in
`data/research/archive_coverage/` (`scripts/`, `data/`, `pdfs/`, `html/`).

Labels used below:

- **VERIFIED**: read directly from an archive index, or checked against downloaded bytes.
- **INDEX-MATCHED**: a CDX capture timestamp falls inside a schedule-defined window. The packet
  content is not checked, apart from the 16 sample PDFs.
- **ESTIMATED**: depends on a heuristic or an assumed tip time.

---

## 0. Headline

| | Team-games | Distinct games |
|---|---:|---:|
| Current project (2023-24 to 2025-26, verified bytes) | 440 | **404** |
| Wayback rolling PDFs, same seasons, strict window (INDEX-MATCHED) | 640 | 570 |
| Union of project and strict Wayback, same seasons | n/a | **629 (+225)** |
| Union with loose window (<=7 d; upper bound with known false positives) | n/a | 705 (+301) |
| Arquivo.pt, in addition (INDEX-MATCHED, all seasons) | +21 | +18 |
| Common Crawl, untruncated, in addition (INDEX-MATCHED; 2016-17 to 2018-19 only) | +57 | +51 |
| Wayback rolling PDFs, all seasons 2012-13 to 2025-26, regular season, strict | **2,624** | **2,392** |
| Same, including play-in, playoffs and Cup final (no 2025-26 postseason schedule) | 2,831 | 2,577 |
| Knicks HTML game-notes articles, 2024-25 to 2025-26 (see section 2) | 27 regular season + about 7 playoff (ESTIMATED) | 24 new regular-season games |
| **All sources combined, 2023-24 to 2025-26 regular season:** project, Wayback strict, Arquivo.pt (+13 in this window), Knicks HTML (+24), archive.today (+2) | | **about 668 candidate games (vs 404)** |
| **All sources combined, 2012-13 to 2025-26 regular season** (ESTIMATED by summing the rows above: 2,392 + 59 project-only + about 17 Arquivo + about 48 CC + 24 + 2) | | **about 2,540 candidate games** |

**The under-count did not come mainly from the nearest-capture API.** `scripts/collect_game_notes.py`
already ran a full CDX query (`collapse=digest`) for each rolling URL. The 225 additional games in
2023-26 come from four collection gaps:

1. **118 team-games were candidates in the project audit, but the replay fetch failed with
   `network_error` ("Connection refused", i.e. Internet Archive throttling).** Across the audit,
   143 candidate rows (130 rolling plus 13 dated) end in `replay_unverified/network_error`. That
   leaves 133 team-games with no verified version. These are not evidence of absence. A slower retry
   should recover most of them.
2. **PHI: 71 team-games.** The project's Philadelphia index lookup failed. The Wayback has 647
   distinct `sixers.pdf` versions.
3. **LAL: 67 team-games.** Most Lakers captures sit on query-string variants
   (`lakers.pdf?ls=footer_nav`, `?cid=nav_news&ls=nav_extended_news`, ...). An exact-URL CDX query
   misses them: 904 of 1,061 Lakers CDX rows are on such variants.
4. Smaller gains: revisit records, which `filter=statuscode:200` drops, and the
   `to=20260410` bound.

**Historical depth is the main upside.** The rolling URL has been captured since March 2012.
Seasons 2012-13 to 2022-23 add about 1,980 index-matched regular-season team-games that the project
never looked at. 2015-16, 2020-21, 2021-22 and 2022-23 are the best-covered seasons.

**Coverage is very concentrated.** BOS, DET, PHI, LAL, DEN, MEM, MIA and MIL account for 76%
(2,000 of 2,624) of strict regular-season matches. Most other teams have only 1 to 8 matches per season. Wayback capture volume fell from
about 200 per month (Jan 2023 to Sep 2024) to 10 to 60 per month after October 2024. That timing
matches the Internet Archive's October 2024 outage, but this explanation is a hypothesis. As a
result, 2024-25 and 2025-26 are thin.

---

## 1. Rolling PDFs `www.nba.com/gamenotes/{slug}.pdf` in the Wayback Machine

### 1.1 Queries (VERIFIED)

Master query, with no status filter and no collapse, so that redirects, 403s and revisits are
visible:

```
http://web.archive.org/cdx/search/cdx?url=www.nba.com/gamenotes/*&output=json&fl=urlkey,timestamp,original,mimetype,statuscode,digest,length
```

- 12,311 rows, of which 12,298 are unique, across 177 distinct urlkeys, from 1999 (`gamenotes.html`)
  to 2026-10.
- **Completeness check:** `&showNumPages=true` returns 3. Pages `&page=0`, `&page=1` and `&page=2`
  were fetched separately. Their union is identical to the unpaged result (12,298 = 12,298 tuples),
  so the unpaged query is complete as of today.
- The prefix query matches the URL after SURT normalisation. It therefore covers `http`/`https`,
  `nba.com` with or without `www`, `:80`, `gamenotes//slug.pdf` (double slash) and every
  query-string variant. That is why it is a superset of the per-URL queries.
- Per-URL control (VERIFIED):
  `http://web.archive.org/cdx/search/cdx?url=https://www.nba.com/gamenotes/lakers.pdf&output=json&fl=urlkey,timestamp,original,mimetype,statuscode,digest,length`
  returns 118 rows, exactly the prefix rows whose urlkey is `com,nba)/gamenotes/lakers.pdf`. The
  prefix query has 1,050 rows starting with `lakers.pdf`. The other 932 are on query-string
  variants, which an exact-URL query never returns.
- The project's own per-URL query (from `reports/game_notes_collection/rolling_indexes.json`) is,
  for example:
  `https://web.archive.org/cdx/search/cdx?url=https%3A%2F%2Fwww.nba.com%2Fgamenotes%2Fceltics.pdf&output=json&filter=statuscode%3A200&from=20231001&to=20260410&collapse=digest&fl=timestamp%2Coriginal%2Cdigest`

Slug-to-team mapping: the current 30 slugs plus `bobcats` (CHA), `76ers` (PHI) and `trailblazers`
(POR). `hornets` before June 2014 means the New Orleans Hornets; from 2014-15 it means Charlotte.
Typos and truncated slugs (`griz%e2%80%a6`, `magi`, `knicls`, `PACERS.pdf`, ...) were never 200
PDFs and were ignored.

Capture types on team URLs (VERIFIED): 12,088 rows, made up of:

- 6,260 `200 application/pdf` (plus 1 `application/octet-stream`)
- 5,057 `warc/revisit` (status `-`)
- 670 redirects
- 101 4xx/5xx/other

A `filter=statuscode:200` query drops every revisit. Revisits never introduce a new digest except
in 296 orphan cases (7 distinct digests, almost all offseason 2020). They do, however, add evidence
times. Versions here are computed over 200s plus revisits, using the earliest timestamp per digest.

### 1.2 Per-team capture summary (VERIFIED)

| Team | Slug(s) | All CDX rows | 200 PDF | Revisit | Redirect | 4xx/5xx/other | Rows on query-string variants | Distinct digests | First | Last |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| ATL | hawks | 107 | 100 | 5 | 1 | 1 | 7 | 78 | 20120311 | 20260920 |
| BKN | nets | 169 | 152 | 10 | 6 | 1 | 9 | 112 | 20120316 | 20261023 |
| BOS | celtics | 1865 | 775 | 948 | 131 | 11 | 5 | 647 | 20120323 | 20260223 |
| CHA | hornets, bobcats | 72 | 56 | 7 | 8 | 1 | 6 | 48 | 20120316 | 20260204 |
| CHI | bulls | 66 | 59 | 2 | 4 | 1 | 6 | 52 | 20120316 | 20251219 |
| CLE | cavaliers | 72 | 58 | 3 | 10 | 1 | 6 | 50 | 20120316 | 20251205 |
| DAL | mavericks | 69 | 55 | 1 | 10 | 3 | 6 | 49 | 20120317 | 20260112 |
| DEN | nuggets | 1238 | 495 | 726 | 10 | 7 | 6 | 392 | 20120417 | 20260827 |
| DET | pistons | 1421 | 615 | 762 | 41 | 3 | 5 | 542 | 20120316 | 20261016 |
| GSW | warriors | 478 | 174 | 116 | 184 | 4 | 24 | 111 | 20120316 | 20261012 |
| HOU | rockets | 74 | 64 | 3 | 6 | 1 | 6 | 58 | 20120316 | 20260912 |
| IND | pacers | 128 | 91 | 28 | 8 | 1 | 6 | 82 | 20120316 | 20260123 |
| LAC | clippers | 75 | 54 | 5 | 12 | 4 | 6 | 48 | 20120316 | 20260109 |
| LAL | lakers | 1061 | 586 | 426 | 40 | 9 | 904 | 454 | 20120323 | 20261023 |
| MEM | grizzlies | 1255 | 517 | 712 | 19 | 7 | 9 | 416 | 20120323 | 20260822 |
| MIA | heat | 717 | 397 | 288 | 25 | 7 | 3 | 316 | 20120316 | 20260924 |
| MIL | bucks | 673 | 393 | 271 | 4 | 5 | 4 | 351 | 20120316 | 20260130 |
| MIN | timberwolves | 111 | 89 | 16 | 6 | 0 | 7 | 74 | 20120323 | 20260125 |
| NOP | pelicans (+ NOH-era hornets) | 62 | 50 | 4 | 7 | 1 | 5 | 42 | 20120316 | 20261012 |
| NYK | knicks | 59 | 54 | 1 | 3 | 1 | 5 | 45 | 20120323 | 20260320 |
| OKC | thunder | 157 | 105 | 31 | 19 | 2 | 7 | 94 | 20120316 | 20260318 |
| ORL | magic | 89 | 56 | 10 | 16 | 7 | 3 | 54 | 20120324 | 20261001 |
| PHI | sixers, 76ers | 1362 | 738 | 596 | 20 | 8 | 5 | 647 | 20120316 | 20260916 |
| PHX | suns | 85 | 66 | 4 | 15 | 0 | 5 | 62 | 20120323 | 20260119 |
| POR | blazers, trailblazers | 83 | 60 | 6 | 11 | 6 | 5 | 51 | 20120316 | 20260317 |
| SAC | kings | 70 | 57 | 3 | 8 | 2 | 6 | 53 | 20120316 | 20260322 |
| SAS | spurs | 77 | 60 | 4 | 12 | 1 | 4 | 56 | 20120317 | 20260226 |
| TOR | raptors | 225 | 145 | 52 | 24 | 4 | 4 | 127 | 20120316 | 20261022 |
| UTA | jazz | 109 | 88 | 15 | 5 | 1 | 5 | 76 | 20120316 | 20251204 |
| WAS | wizards | 59 | 51 | 2 | 5 | 1 | 3 | 48 | 20120316 | 20250818 |
| **All** | | **12088** | **6260** | **5057** | **670** | **101** | **1082** | **5235** | | |

The 5,235 distinct digests exclude placeholder payloads (section 4). 87 of them were first seen in
the current 2026-27 season (Aug to Oct 2026), so they are absent from the season tables below.

### 1.3 Unique digests per team per season (VERIFIED)

Each digest is counted once, in the season of its earliest capture. A season runs from August 1 to
July 31; Jul to Oct 2020 counts as 2019-20, for the bubble.

| Team | 2012-13 | 2013-14 | 2014-15 | 2015-16 | 2016-17 | 2017-18 | 2018-19 | 2019-20 | 2020-21 | 2021-22 | 2022-23 | 2023-24 | 2024-25 | 2025-26 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ATL | 8 | 11 | 11 | 5 | 3 | 5 | . | 1 | 6 | 9 | 4 | 3 | 5 | 1 | 72 |
| BKN | 2 | 44 | 6 | 7 | 4 | 4 | 2 | 1 | 9 | 6 | 5 | 5 | 4 | 8 | 107 |
| BOS | 15 | 43 | 20 | 68 | 11 | 13 | 9 | 7 | 73 | 98 | 128 | 141 | 11 | 6 | 643 |
| CHA | . | 6 | 4 | 2 | 3 | 3 | 3 | . | 4 | 5 | 4 | 2 | 8 | 2 | 46 |
| CHI | 3 | 7 | 4 | 3 | 4 | 2 | 1 | 2 | 2 | 5 | 6 | 3 | 5 | 2 | 49 |
| CLE | 1 | 5 | 4 | 2 | 4 | 3 | 1 | 1 | 4 | 6 | 4 | 3 | 8 | 2 | 48 |
| DAL | 1 | 6 | 4 | 3 | 1 | 3 | 2 | 1 | 3 | 5 | 4 | 4 | 7 | 2 | 46 |
| DEN | 5 | 7 | 5 | 3 | 2 | 10 | 4 | 3 | 77 | 74 | 81 | 98 | 16 | 5 | 390 |
| DET | 1 | 6 | 17 | 80 | 13 | 29 | 8 | 8 | 98 | 22 | 113 | 130 | 9 | 5 | 539 |
| GSW | 2 | 6 | 4 | 5 | 5 | 14 | 8 | 6 | 6 | 16 | 8 | 5 | 14 | 9 | 108 |
| HOU | 1 | 7 | 6 | 5 | 4 | 4 | 3 | 1 | 3 | 6 | 3 | 4 | 6 | 2 | 55 |
| IND | 2 | 7 | 6 | 3 | 3 | 11 | 27 | 2 | 2 | 3 | 2 | 2 | 5 | 4 | 79 |
| LAC | 1 | 10 | 5 | 3 | 1 | 3 | 1 | 5 | 2 | 3 | 4 | 1 | 5 | 2 | 46 |
| LAL | 4 | 33 | 12 | 63 | 15 | 14 | 45 | 13 | 12 | 11 | 110 | 98 | 12 | 8 | 450 |
| MEM | 3 | 6 | 5 | 4 | 15 | 13 | 19 | 5 | 84 | 86 | 70 | 93 | 7 | 4 | 414 |
| MIA | 3 | 7 | 6 | 31 | 13 | 12 | 22 | 17 | 81 | 98 | 6 | 4 | 6 | 6 | 312 |
| MIL | . | 12 | 6 | 4 | 3 | 7 | 8 | 2 | 6 | 8 | 137 | 136 | 12 | 7 | 348 |
| MIN | 3 | 29 | 6 | 4 | 3 | 3 | 3 | 3 | 3 | 5 | 3 | 2 | 4 | 1 | 72 |
| NOP | 1 | 4 | 3 | 4 | 2 | 3 | 3 | 3 | 5 | 3 | . | 2 | 5 | 1 | 39 |
| NYK | 1 | 6 | 5 | 3 | 3 | 4 | 1 | 1 | 4 | 4 | 1 | 5 | 4 | 2 | 44 |
| OKC | 4 | 9 | 7 | 3 | 4 | 13 | 28 | 2 | 2 | 2 | 3 | 1 | 8 | 6 | 92 |
| ORL | . | 7 | 5 | 3 | 3 | 4 | 7 | 4 | 5 | 5 | 2 | 2 | 4 | 1 | 52 |
| PHI | . | 6 | 18 | 84 | 19 | 13 | 10 | 6 | 111 | 127 | 101 | 136 | 5 | 6 | 642 |
| PHX | 3 | 6 | 6 | 7 | 2 | 5 | 1 | 5 | 4 | 5 | 4 | 4 | 5 | 4 | 61 |
| POR | 3 | 7 | 5 | 3 | 3 | 4 | 2 | 1 | 4 | 3 | 4 | 2 | 6 | 2 | 49 |
| SAC | 2 | 6 | 5 | 5 | 2 | 2 | 3 | 3 | 3 | 4 | 3 | 2 | 6 | 5 | 51 |
| SAS | 3 | 7 | 4 | 4 | 4 | 4 | 4 | 2 | 3 | 3 | 4 | 2 | 4 | 5 | 53 |
| TOR | 1 | 15 | 7 | 3 | 3 | 9 | 4 | 3 | 1 | 5 | 51 | 5 | 13 | 2 | 122 |
| UTA | 2 | 34 | 4 | 4 | 2 | 3 | 1 | 2 | 4 | 5 | 1 | 2 | 7 | 2 | 73 |
| WAS | . | 14 | 6 | 4 | 2 | 2 | 2 | 1 | 3 | 3 | 2 | 2 | 4 | 1 | 46 |
| **All** | **75** | **373** | **206** | **422** | **156** | **219** | **232** | **111** | **624** | **635** | **868** | **899** | **215** | **113** | **5148** |

### 1.4 Matching captures to games

Schedules (read-only):

- **2023-24 to 2025-26 regular season:** the project's
  `data/processed/seasonal/game_history.jsonl` (ESPN; ends 2026-04-10; All-Star rows dropped).
- **2015-16 to 2024-25, including postseason:** the NBA's
  `https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/{YYYY}/league/00_full_schedule.json`,
  which gives `gdtutc` + `utctm` tip times. The file for 2025 is a preseason snapshot and was used
  only where history is missing. 2010 to 2014 return 403.
- **2012-13 to 2014-15:** ESPN
  `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{1..30}/schedule?season={2013..2015}&seasontype={2,3}`.

So no season needed the timestamp-only heuristic in the end (validation in section 4).

**Strict rule (primary):** a team-game counts when a digest was **first** seen within both of these
bounds:

- after the team's previous scheduled tip plus 2.5 h (a proxy for when that game ends), and
- at or before the cutoff (tip minus 60 min), and no more than 36 h before it (as in the project).

**Loose rule (upper bound):** the same, but with up to 7 days before the cutoff.

A capture in the window whose digest was already seen earlier is a **stale packet** and is not
counted.

**Team-games with a new digest in the strict window, regular season** (INDEX-MATCHED; each team row
counts that team's packet):

| Team | 2012-13 | 2013-14 | 2014-15 | 2015-16 | 2016-17 | 2017-18 | 2018-19 | 2019-20 | 2020-21 | 2021-22 | 2022-23 | 2023-24 | 2024-25 | 2025-26 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ATL | 4 | 6 | 3 | 2 | 2 | 2 | . | . | 3 | 3 | 2 | 1 | 3 | 1 | 32 |
| BKN | . | 24 | 2 | 4 | 1 | 2 | 1 | . | 6 | 5 | 2 | 3 | 3 | 4 | 57 |
| BOS | 11 | 25 | 10 | 43 | 4 | 6 | 6 | 2 | 42 | 47 | 58 | 58 | 6 | 5 | 323 |
| CHA | . | 2 | 2 | . | 2 | 1 | . | . | . | 2 | 2 | . | 4 | 1 | 16 |
| CHI | 1 | 3 | 2 | 1 | 1 | 2 | . | 1 | 1 | 2 | 5 | . | 3 | 1 | 23 |
| CLE | . | 1 | . | 1 | 2 | 1 | 1 | . | 2 | 2 | 2 | 2 | 3 | 1 | 18 |
| DAL | . | 2 | 2 | 1 | 1 | 1 | . | 1 | 1 | 2 | 1 | 1 | 6 | 1 | 20 |
| DEN | 3 | 4 | 2 | 1 | 1 | 5 | 3 | . | 44 | 44 | 62 | 64 | 9 | 3 | 245 |
| DET | . | 1 | 7 | 48 | 6 | 18 | 5 | 4 | 54 | 11 | 60 | 64 | 5 | 3 | 286 |
| GSW | 1 | 3 | 2 | 2 | 4 | 8 | 5 | 3 | 4 | 3 | 4 | 2 | 4 | 4 | 49 |
| HOU | . | 3 | 1 | 1 | 3 | 2 | 2 | . | 1 | 2 | 1 | 2 | 4 | 2 | 24 |
| IND | 1 | 5 | 3 | 1 | 2 | 5 | 7 | . | 1 | 1 | 2 | . | 3 | 3 | 34 |
| LAC | . | 5 | 1 | 2 | 1 | 3 | . | 1 | 1 | 2 | 2 | . | 4 | 1 | 23 |
| LAL | 1 | 23 | 5 | 43 | 11 | 8 | 20 | 4 | 4 | 7 | 62 | 64 | 6 | 5 | 263 |
| MEM | 2 | 3 | . | 2 | 8 | 7 | 7 | 2 | 46 | 37 | 35 | 54 | 5 | 2 | 210 |
| MIA | 1 | 3 | 1 | 12 | 5 | 4 | 12 | 7 | 58 | 52 | 1 | 2 | 3 | 2 | 163 |
| MIL | . | 9 | 3 | 3 | 3 | 5 | 2 | . | 1 | 3 | 68 | 72 | 8 | 3 | 180 |
| MIN | 2 | 19 | 1 | 1 | 2 | 2 | 1 | 1 | 1 | 2 | 2 | . | 2 | 1 | 37 |
| NOP | . | 2 | . | 2 | 1 | 2 | 2 | . | 3 | 1 | . | . | 3 | . | 16 |
| NYK | . | 4 | 2 | 2 | 3 | 1 | 1 | . | . | 2 | . | 1 | 2 | 1 | 19 |
| OKC | 1 | 2 | 4 | 1 | 4 | 7 | 12 | . | . | 1 | 1 | . | 5 | 3 | 41 |
| ORL | . | 4 | 1 | 1 | 1 | 2 | 6 | 2 | 2 | 2 | . | . | 1 | 1 | 23 |
| PHI | . | 3 | 11 | 52 | 9 | 5 | 6 | 3 | 58 | 57 | 55 | 68 | . | 3 | 330 |
| PHX | 1 | 2 | 2 | 2 | 1 | 1 | 1 | 1 | . | 2 | 2 | 1 | 4 | 2 | 22 |
| POR | 1 | 2 | 1 | 1 | . | 1 | . | . | 3 | 2 | 1 | 1 | 4 | 1 | 18 |
| SAC | 1 | 3 | 3 | 1 | 1 | 1 | 3 | . | 2 | 2 | 2 | . | 3 | 3 | 25 |
| SAS | 1 | 3 | . | 2 | 3 | 3 | 2 | . | 1 | 1 | 1 | 1 | 1 | 2 | 21 |
| TOR | 1 | 4 | . | . | 2 | 4 | 1 | . | . | 3 | 25 | 2 | 8 | 1 | 51 |
| UTA | 1 | 19 | 3 | 2 | . | 1 | . | . | 1 | 2 | 1 | . | 2 | 1 | 33 |
| WAS | . | 8 | 2 | 2 | 2 | 1 | . | . | 2 | 1 | 2 | . | 2 | . | 22 |
| **All** | **34** | **197** | **76** | **236** | **86** | **111** | **106** | **32** | **342** | **303** | **461** | **463** | **116** | **61** | **2624** |

**Season totals** (INDEX-MATCHED):

| Measure | 2012-13 | 2013-14 | 2014-15 | 2015-16 | 2016-17 | 2017-18 | 2018-19 | 2019-20 | 2020-21 | 2021-22 | 2022-23 | 2023-24 | 2024-25 | 2025-26 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Distinct digests (first seen in season) | 75 | 373 | 206 | 422 | 156 | 219 | 232 | 111 | 624 | 635 | 868 | 899 | 215 | 113 | 5148 |
| Team-games, regular season, strict | 34 | 197 | 76 | 236 | 86 | 111 | 106 | 32 | 342 | 303 | 461 | 463 | 116 | 61 | 2624 |
| Distinct games, regular season, strict | 33 | 172 | 73 | 223 | 80 | 89 | 102 | 32 | 316 | 281 | 421 | 417 | 95 | 58 | 2392 |
| Team-games incl. play-in/playoffs/Cup final, strict | 35 | 202 | 79 | 255 | 90 | 121 | 107 | 38 | 363 | 364 | 494 | 503 | 119 | 61 | 2831 |
| Distinct games incl. postseason, strict | 34 | 177 | 76 | 241 | 84 | 98 | 103 | 38 | 337 | 330 | 452 | 451 | 98 | 58 | 2577 |
| Team-games, regular season, loose (<=7 d) | 42 | 234 | 114 | 293 | 108 | 157 | 170 | 40 | 374 | 335 | 507 | 484 | 164 | 83 | 3105 |
| Distinct games, regular season, loose | 41 | 200 | 107 | 275 | 101 | 127 | 161 | 39 | 341 | 308 | 457 | 437 | 131 | 79 | 2804 |
| Regular-season team-games scheduled (2012-14 from ESPN team schedules, approximate; 2023-26 history ends 2026-04-10) | 2464 | 2464 | 2464 | 2460 | 2460 | 2460 | 2460 | 2118 | 2160 | 2460 | 2460 | 2392 | 2402 | 2432 | 33656 |


Distinct games count a game once, even when both teams' packets were captured.

### 1.5 Comparison with the project's 404 games (2023-24 to 2025-26 regular season)

The project's verified set comes from
`data/processed/game_notes_collection/audit.jsonl`, rows with `text_path`: 482 versions,
440 team-games, 404 games.

| Team | Project verified team-games | This audit (strict) | Overlap | New here | Project candidates lost to network errors |
|---|---:|---:|---:|---:|---:|
| ATL | 4 | 5 | 4 | 1 | 0 |
| BKN | 7 | 10 | 7 | 3 | 0 |
| BOS | 60 | 69 | 56 | 13 | 13 |
| CHA | 5 | 5 | 5 | 0 | 0 |
| CHI | 4 | 4 | 4 | 0 | 0 |
| CLE | 57 | 6 | 6 | 0 | 14 |
| DAL | 8 | 8 | 8 | 0 | 0 |
| DEN | 57 | 76 | 57 | 19 | 18 |
| DET | 59 | 72 | 55 | 17 | 19 |
| GSW | 8 | 10 | 6 | 4 | 1 |
| HOU | 5 | 8 | 5 | 3 | 2 |
| IND | 6 | 6 | 6 | 0 | 0 |
| LAC | 5 | 5 | 5 | 0 | 0 |
| LAL | 8 | 75 | 8 | 67 | 0 |
| MEM | 50 | 61 | 50 | 11 | 13 |
| MIA | 7 | 7 | 7 | 0 | 0 |
| MIL | 62 | 83 | 61 | 22 | 25 |
| MIN | 3 | 3 | 3 | 0 | 0 |
| NOP | 2 | 3 | 2 | 1 | 0 |
| NYK | 3 | 4 | 3 | 1 | 1 |
| OKC | 8 | 8 | 8 | 0 | 0 |
| ORL | 2 | 2 | 2 | 0 | 0 |
| PHI | 0 | 71 | 0 | 71 | 0 |
| PHX | 2 | 7 | 2 | 5 | 4 |
| POR | 0 | 6 | 0 | 6 | 5 |
| SAC | 0 | 6 | 0 | 6 | 6 |
| SAS | 0 | 4 | 0 | 4 | 4 |
| TOR | 6 | 11 | 6 | 5 | 5 |
| UTA | 1 | 3 | 1 | 2 | 2 |
| WAS | 1 | 2 | 1 | 1 | 1 |
| **All** | **440** | **640** | **378** | **262** | **133** |

- Distinct games: project 404, this audit 570, union 629, of which **225 are new**.
- 133 team-games were project candidates that never got a verified version because of network
  errors. 118 of them are in the strict set here.
- In the CLE row, "This audit" counts only the rolling `cavaliers.pdf`. The project's 57 include
  the dated CLE PDFs.

- The project has 62 team-games that the strict rule here does not reproduce. 51 are CLE dated
  PDFs (a different URL family). The other 11 (BOS 4, DET 4, GSW 2, MIL 1) most likely fall before the
  lower bound used here (previous tip plus 2.5 h). This was not checked case by case.
- With the loose rule, the union is 705 games (+301). The extra loose-only matches are
  unreliable (see the NOP sample in section 1.6).

### 1.6 Sample verification (VERIFIED, 16 PDFs downloaded)

Every sample was fetched as `https://web.archive.org/web/{ts}id_/{original}`. All 16 returned:

- HTTP 200,
- a `Memento-Datetime` equal to the requested timestamp, and
- a SHA-1 of the bytes (base32) equal to the CDX `digest`.

So the CDX digest is a usable content hash.

| # | Team | Matched game (ET) | First capture (UTC) | Result |
|---|---|---|---|---|
| 0 | BOS | 2015-12-05 at SAS | 2015-12-05 10:56 | OK: "Boston Celtics (11-8) at San Antonio Spurs ... Saturday, December 5, 2015" (page 1, beneath the roster) |
| 1 | DET | 2016-01-06 at BOS | 2016-01-05 21:57 | OK: "DETROIT PISTONS (19-16) AT BOSTON CELTICS ... JANUARY 6, 2016" |
| 2 | PHI | 2016-03-27 at GSW | 2016-03-27 08:55 | OK: "UPCOMING SCHEDULE Sun. Mar. 27 at Golden State" |
| 3 | LAL | 2016-03-13 vs NYK | 2016-03-13 03:59 | OK: query-string URL `lakers.pdf?ls=nav_extended_news&cid=nav_news`; header "LAKERS (14-52) VS. NEW YORK KNICKS ... Game #67" |
| 4 | GSW | 2017-10-25 vs TOR | 2017-10-25 05:55 | OK |
| 5 | OKC | 2019-02-03 at BOS | 2019-02-03 12:33 | OK |
| 6 | DEN | 2021-03-19 vs CHI | 2021-03-19 03:49 | OK |
| 7 | MIA | 2022-04-05 vs CHA | 2022-04-05 13:55 | OK |
| 8 | MIL | 2022-10-28 vs NYK | 2022-10-28 02:46 | OK |
| 9 | TOR | 2022-12-21 at NYK | 2022-12-20 23:23 | OK: "UPDATED: Dec. 20, 2022 ... at NEW YORK KNICKS ... DECEMBER 21" |
| 10 | PHI | 2024-03-25 at SAC | 2024-03-25 21:41 | OK, but the front pages have **no extractable text** (Word export with glyph bullets). Page 9 reads "Through March 24 (39-32) PHILADELPHIA" and page 10 has the Kings' stats. OCR would be needed. |
| 11 | LAL | 2024-02-25 at PHX | 2024-02-25 00:14 | OK: `?ls=footer_nav` variant |
| 12 | DAL | 2025-03-09 vs PHX | 2025-03-09 06:52 | Partly: the cover reads "03/09/2025 VS. PHOENIX", but the body still says "Friday, March 7, 2025" (the previous home game) |
| 13 | CHA | 2025-12-01 at BKN | 2025-12-01 18:23 | OK |
| 14 | NOP | (loose only) 2024-12-26 vs HOU | 2024-12-23 23:21 | **WRONG**: the 12-22 vs DEN packet, first captured 23 h after that game's tip |
| 15 | BKN | (2013-14 heuristic check) | 2013-12-24 01:39 | Packet for 12-23 vs IND (7:30 pm EST = 00:30 UTC). Captured **69 min after tip**. A "before 22:00 UTC on the capture date" rule would wrongly accept it for 12-24. |

**Precision in the samples:** strict matches are 13/14 fully consistent and 14/14 right on
matchup. Loose-only matches are 0/1. Treat the strict counts as candidate counts that still need
the project's header/date/record checks. Treat loose-only matches as unreliable.

---

## 2. Team HTML game-notes articles and older team-site formats

### 2.1 Knicks per-game articles (VERIFIED index; tip times from schedules)

Query:

```
http://web.archive.org/cdx/search/cdx?url=www.nba.com/knicks/news/*&output=json&fl=urlkey,timestamp,original,mimetype,statuscode,digest,length&filter=original:.*game-notes.*
```

- 160 rows over 76 urlkeys; 75 of them have a `200 text/html` capture.
- First article: `game-notes-knicks-vs-bucks-january-12-2025`. Two slug patterns are in use:
  `game-notes-knicks-vs-{opp}-{month}-{d}-{yyyy}` (Jan 2025) and
  `knicks-vs-{opp}-game-notes-{month}-{d}-{yyyy}[-round-N-game-M]`.
- Typos occur (`wizzards`, `miami-heat`) and one slug has an `&amp` suffix. Parse the date from the
  slug, then join it to the schedule on team and ET date.
- The category page `https://www.nba.com/knicks/category/game-notes` was captured only 3 times
  (2025-03-12, 2025-12-25, 2026-06-18). Crawling the category page in the archive will not find the
  articles. The CDX prefix query will.

The earliest 200 capture of each article URL, against the cutoff:

| Season | First capture before cutoff | First capture after cutoff | No schedule (2025-26 postseason): before an assumed 16:00 UTC cutoff / after |
|---|---:|---:|---:|
| 2024-25 | 5 | 11 | 0 / 1 (typo slug, date not in schedule) |
| 2025-26 | 22 | 19 | 7 / 10 |
| **Total** | **27** | **30** | **7 / 10 (ESTIMATED)** |

- 26 of the pre-cutoff regular-season articles fall within `game_history`, and **24 of those games
  are not covered** by the project, by strict Wayback rolling PDFs, or by Arquivo.pt.
- By comparison, the Knicks rolling PDF (`knicks.pdf`) has only 3 to 5 matched games per season.
  For the Knicks, the HTML articles are the better archive source.
- Sample (VERIFIED): `https://web.archive.org/web/20260406140151id_/https://www.nba.com/knicks/news/knicks-vs-hawks-game-notes-april-6-2026`
  returned `Memento-Datetime: Mon, 06 Apr 2026 14:01:51 GMT`. The page embeds
  `datePublished 2026-04-06T13:25:20Z`, which is first-party and so not proof on its own. The
  capture is 8.5 h before the cutoff (7:30 pm ET tip).
- **Pitfall:** the `id_` body arrived **gzip-encoded** (`Content-Encoding: gzip` is passed
  through), so decompress before parsing.
- Neither archive.today nor Common Crawl has any of these articles: the archive.today timemap
  returned 404, and Common Crawl had 0 records for `com,nba)/knicks/news/knicks-vs-*` in every
  crawl checked.

### 2.2 Other teams' HTML game notes (partial, INDEX-MATCHED)

Query template, one per team:

```
http://web.archive.org/cdx/search/cdx?url=www.nba.com/{slug}/news/*&output=json&fl=urlkey,timestamp,original,mimetype,statuscode,digest,length&filter=original:.*notes.*
```

Use the simple filter. With `(?i)` and alternation, the server returned 503/504 more often. If the
unpaged query still fails, fall back to `&page=N` after `&showNumPages=true`.

Cheaper prefix probes were also run. `url=www.nba.com/{slug}/news/game-notes*` covers the
Knicks' first slug pattern and the Drupal-era landing and per-game patterns. The Internet Archive
answered many requests with 503 or 504, so only some teams were finished.

| Team | Result (VERIFIED from CDX) |
|---|---|
| NYK | 75 per-game articles (section 2.1) |
| DEN | 9 per-game Drupal-era articles, `www.nba.com/nuggets/news/game-notes-{away}-vs-{home}-{mmddyy}` (Nov 2014 to Mar 2015). **All 9 were first captured after the cutoff**, by 5.8 h to years, so they give no pre-cutoff evidence. |
| BOS | Only the landing page `www.nba.com/celtics/news/game-notes` (25 captures, 2013-01 to 2022-06) |
| CHI | Only the landing page `bulls/news/game-notes` (767 captures, 2016 to 2025) and the old rolling `bulls/news/notes.html` and `notes.pdf` (section 2.3) |
| CHA | Only old New Orleans Hornets pages (`hornets/news/gamenotes`, `hornets/news/hornets-2012-13-game-notes`) and one 2006 postgame page |
| ATL, BKN, CLE, DAL, DET, GSW, HOU, IND, LAC, LAL, MEM | No `news/game-notes*` URL in the Wayback. ATL and BKN were also checked with the full `/news/*` notes filter, again with nothing found. |
| MIA, MIL, MIN, NOP, OKC, ORL, PHI, PHX, POR, SAC, SAS, TOR, UTA, WAS | **Not checked** (IA 503/504). TOR's landing page `news-media/game-notes` only links to the rolling PDF (section 2.3). |

Per-game HTML game notes with pre-cutoff captures were found only for the Knicks, among the
16 teams checked. Other slug patterns (for example `{slug}-vs-...-game-notes`) were probed only for
ATL. The remaining teams need the full `/news/*` filter query above, or patterns supplied by the
site-mapping agent.


### 2.3 Team-site formats before the 2021 redesign (VERIFIED from CDX)

| Era | URL pattern | Evidence | Usefulness |
|---|---|---|---|
| Turner era, about 2001 to 2012 | **Rolling HTML:** `www.nba.com/{team}/news/notes.html` | Bulls: 428 captures from 2001-11-05; about 95 distinct digests in 2001-02 to 2012-13. In 2013-14 there are 66 digests, which fall in 33 strict CHI windows. From 2014-15 every capture has the same digest, so the page was frozen. | Pre-2012 seasons are outside the schedules used here. HTML digests can change without new notes, so the 2013-14 matches need content checks. |
| Turner era | **Rolling PDF:** `www.nba.com/{team}/news/notes.pdf` | Bulls: 72 captures, 2003-08 to 2024; about 44 distinct digests in 2003 to 2012 | Small. |
| Turner era | **Per-game HTML:** `www.nba.com/{team}/news/{pre|post}gamenotes_{opp}_{ddmonyyyy}.html` | `www.nba.com/hornets/news/postgamenotes_magic_13oct2006.html` (5 captures, Nov 2006) | **Postgame material, so exclude.** Pregame siblings may exist under the same pattern. |
| Drupal era, about 2012 to 2021 | **Landing pages:** `www.nba.com/{team}/news/game-notes`, `.../news/gamenotes`, `.../news/{team}-{season}-game-notes` | Celtics: 25 captures, 2013-01 to 2022. Bulls: 767 captures, 2016-01 to 2025-08. Hornets (NOH): `news/gamenotes` (2012-10 to 2013-02) and `news/hornets-2012-13-game-notes`. | The sampled Bulls landing page (`https://web.archive.org/web/20220112041537id_/https://www.nba.com/bulls/news/game-notes`, title "Chicago Bulls Game Notes", `datePublished 2016-01-06`, `dateModified 2019-04-10`) holds **no notes text**. It only links to `https://www.nba.com/gamenotes/bulls.pdf` and to `https://www.nba.com/resources/static/team/v2/bulls/schedule/...pdf`. So the league rolling PDF *was* the team-site game-notes link, and section 1 already covers it. |
| Drupal era static files | `www.nba.com/resources/static/team/v2/{team}/...`; `www.nba.com/{team}/sites/{team}/files/...` | Size checks: `www.nba.com/celtics/sites/celtics/files/*` = 27 CDX pages, `www.nba.com/knicks/sites/knicks/files/*` = 21 pages. `i.cdn.turner.com/nba/nba/.element/media/2.0/teamsites/*` = 168 pages (too large to scan unpaged). | **Not completed.** The filtered unpaged query
`http://web.archive.org/cdx/search/cdx?url=www.nba.com/celtics/sites/celtics/files/*&output=json&fl=urlkey,timestamp,original,mimetype,statuscode,digest,length&filter=original:.*notes.*`
returned 504 twice. A paged scan (`&page=0..26`) is needed, at about 30 s per page today. The
same template applies with `knicks` (21 pages) and with
`www.nba.com/resources/static/team/v2/*`. The Bulls landing page suggests these folders mostly
held schedules and media guides, and that game notes pointed to the rolling PDF. That is an
inference, not checked. |
| WordPress era, 2021 onward | `cdn.nba.com/teams/uploads/sites/{teamId}/{YYYY}/{MM}/...notes.pdf` (the CLE adapter); `www.nba.com/{team}/news/...game-notes...`; landing pages such as `www.nba.com/raptors/news-media/game-notes` and `www.nba.com/cavaliers/team/game-notes-media-guide` | `cdn.nba.com/teams/uploads/sites/1610612739/*` (CLE) = 40 CDX pages | **Not completed.** The filtered query
`http://web.archive.org/cdx/search/cdx?url=cdn.nba.com/teams/uploads/sites/1610612739/*&output=json&fl=urlkey,timestamp,original,mimetype,statuscode,digest,length&filter=original:.*notes.*`
was queued but not run, given today's 503/504 rate. Expected gain for CLE is small. The project's
dated adapter already checked a 2-day pre-cutoff CDX window for every CLE game, and only
13 dated candidates failed on network errors. For other team IDs (1610612737 to 1610612766), a
per-month prefix such as `cdn.nba.com/teams/uploads/sites/{id}/{YYYY}/{MM}/*` with
`filter=original:.*notes.*` is the cheapest way to look for dated uploads. |

**Post-2021 landing pages (VERIFIED, 2 HTML samples):**

- `www.nba.com/raptors/news-media/game-notes` has 1,190 captures (2021-12-14 to 2026-10-22,
  797 digests). The sampled capture (`https://web.archive.org/web/20240205090931id_/https://www.nba.com/raptors/news-media/game-notes`)
  only links to `/gamenotes/raptors.pdf`.
- `www.nba.com/cavaliers/team/game-notes-media-guide` has 789 captures (2022 to 2026, near-daily
  in 2022-24). The sampled capture (`https://web.archive.org/web/20240205060059id_/https://www.nba.com/cavaliers/team/game-notes-media-guide`)
  links to the **current dated packet**, `https://cdn.nba.com/teams/uploads/sites/1610612739/2024/02/240203-clesas-notes.pdf`.
  So the Cavs landing-page captures give a third-party timestamp for *when each dated packet
  URL was live*. That could supplement CLE games whose PDF bytes were not captured before the
  cutoff. It is weaker evidence: link existence, with bytes fetched later. It has not been
  quantified here; doing so would mean downloading about 790 HTML captures.
- `www.nba.com/resources/static/team/v2/*` is 275 CDX pages, too large to scan today.

Rule of thumb: for the 2012 to 2021 period, `www.nba.com/gamenotes/{slug}.pdf` is the
team-site game-notes document. The older `/{team}/news/notes.(html|pdf)` and per-game
`gamenotes_*.html` URLs only matter for seasons before 2012-13.


---

## 3. Other archives

| Archive | Query used | Result | Adds captures the Wayback lacks? |
|---|---|---|---|
| **Arquivo.pt** (VERIFIED index) | `https://arquivo.pt/wayback/cdx?url=www.nba.com/gamenotes/&matchType=prefix&output=json&limit=100000`. Note: `url=.../gamenotes/*` returns an empty body, so use `matchType=prefix` without `*`. | 222 rows (2018 to 2025), 186 digests. 54 digests are also in the Wayback, so the digest scheme is the same. **51 digests are not in the Wayback.** | **Yes: +21 strict team-games, 18 of them new distinct games** beyond Wayback plus project (+34 loose). They cluster in sweep crawls on 2025-12-06 and 2025-12-22, which helps the thin 2025-26 season. Not byte-verified. Several have 30 to 36 h lead times, so stale-packet risk is high. |
| **archive.today / archive.ph** (VERIFIED) | `https://archive.ph/timemap/{http,https}://www.nba.com/gamenotes/{slug}.pdf` for all 30 slugs, plus `lakers.pdf?ls=footer_nav` and one Knicks HTML article | 27 slugs: explicit "TimeMap does not exists". One memento each for `nets.pdf` (2025-01-12 00:58 UTC), `rockets.pdf` (2024-01-26 14:47 UTC) and `grizzlies.pdf` (2023-10-11 20:56 UTC, preseason). The service throttled after about 50 requests (60 s timeouts); slower retries completed. | **Marginal: 2 candidate team-games.** BKN at UTA 2025-01-12 (23 h before cutoff) and HOU at CHA 2024-01-26, both with no Wayback capture in the window. archive.today exposes no digests, and its PDF mementos may be renders. Unverified. |
| **Common Crawl** (VERIFIED index) | The index server `https://index.commoncrawl.org/CC-MAIN-YYYY-WW-index?url=www.nba.com/gamenotes/*&output=json` returned 504 for most crawls today. Instead, the CDX shards were read directly: binary search on `https://data.commoncrawl.org/cc-index/collections/{crawl}/indexes/cluster.idx` with HTTP Range, then a range-fetch of the matching `cdx-NNNNN.gz` block (`scripts/cc_rangeidx.py`). | All 126 crawls from CC-MAIN-2013-20 to CC-MAIN-2026-43 were read. 59 crawls contain at least one record: 338 records in total, 263 of them `200 application/pdf`. **63 PDF records are `"truncated": "length"`**: CC cuts large payloads, so those bytes are unusable. No crawl from 2025-05 onward contains any game-notes PDF. 112 of the 338 records have the identical urlkey and timestamp in the Wayback CDX, so the Wayback indexes part of CC. | **Yes, for 2016-17 to 2018-19.** Untruncated CC records contribute 123 digests that the Wayback lacks, giving **+57 strict team-games and 51 new distinct games** beyond Wayback plus Arquivo.pt (2016-17: 1, 2017-18: 15, 2018-19: 41). They are spread across almost every team. CC adds nothing for 2023-26. |
| **Memento aggregator** (VERIFIED unavailable) | `http://timetravel.mementoweb.org/timemap/json/http://www.nba.com/gamenotes/knicks.pdf` | DNS for `timetravel.mementoweb.org` no longer resolves (the service is retired). The public MemGator (`https://memgator.cs.odu.edu/timemap/json/...`) returns "404 page not found" even for `http://www.stanford.edu/`, so it is broken. | Not testable via an aggregator. Each archive was queried directly instead. |
| **Library of Congress Web Archives** (VERIFIED blocked) | `https://webarchive.loc.gov/all/timemap/link/https://www.nba.com/gamenotes/celtics.pdf` and `https://webarchive.loc.gov/all/cdx?url=www.nba.com/gamenotes/*&output=json` | HTTP 403 Cloudflare "Just a moment..." challenge for curl and for WebFetch | Unknown, low prior. LoC collects selectively by theme and embargoes recent captures. A browser session would be needed. |
| Archive-It (extra) | `https://wayback.archive-it.org/all/timemap/cdx?url=www.nba.com/gamenotes/&matchType=prefix` | 403 "Request forbidden by administrative rules"; timemap behind a "Session Verification" wall | Unknown. Public Archive-It collections usually also appear in the Wayback. |

Summary:

- **Common Crawl** is useful for 2017-2019: 51 new games, which helps the many teams the
  Wayback barely covered in those years. It is useless for 2023-26.
- **Arquivo.pt** adds about 18 games, mostly in 2025-26.
- **archive.today** adds 2.
- The Memento aggregator, LoC and Archive-It could not be queried.


---

## 4. Pitfalls (with measured frequencies)

1. **Redirects are not documents.** Team URLs have 670 redirect captures: 553 `http://` to
   https, 94 https, and 23 `gamenotes//slug.pdf` double-slash URLs. A 301 capture has no PDF body.
   A replay without `id_` follows it to whichever capture is *nearest*, which can be later than the
   cutoff. Always replay `.../web/{ts}id_/{original}` and require `Memento-Datetime == ts`. All
   16 samples passed this check.
2. **403 and 404 HTML pages.** There are 28 `403 text/html` captures (about 750 bytes) clustered
   on 2023-01-21, 2023-03-06, 2023-03-14 and 2023-05-16. They are the same second across many
   teams, so a crawler was being blocked. There are also 51 `404 text/html` and 18
   `404 application/xml` captures (Azure `BlobNotFound`), and 2 `200 text/html` captures on typo
   URLs. Filter on `statuscode==200` **and** `mimetype==application/pdf` (or a revisit), then check
   that the body starts with `%PDF-`.
3. **Revisit records.** 5,057 rows are `warc/revisit` with status `-`, so `filter=statuscode:200`
   silently drops them. They repeat an earlier digest. Keep them as evidence times, but date each
   version by its earliest capture of any type.
4. **Placeholder digests shared across teams.**
   - `3I42H3S6NNFQ2MSVX7XZKYAYSCX5QBYJ` is the SHA-1 of an **empty body**. It appears under 12
     team URLs.
   - `P37GKNRZ6O7CHIW7QUG2B5PVSDBLKDVH` appears under 7 teams in Oct to Nov 2020, as revisits
     only.
   - Three more digests are shared by pairs of teams.

   26 digest-team pairs were dropped. Rule: discard any digest seen under more than one team URL.
5. **Identical digests reused, i.e. stale packets.**
   - Under the strict rule, 2,625 team-game windows (all seasons, including postseason) contain a
     new digest. A further **381 windows contain captures only of an older digest**: the team had
     not yet replaced the previous game's packet. These are not evidence for the later game.
   - Separately, there are 59 "A to B to A" reversions, where an older digest reappears after a
     newer one.
   - The opposite failure also happens: a **new** digest can be a late first capture of the
     *previous* game's packet. The NOP sample shows this, so first-seen-after-the-previous-game is
     not sufficient. A content check (header date, opponent, prior record) is mandatory.
   - The DAL sample shows that even a correct packet can carry stale internal lines.
6. **Multiple versions per game.** Of the 2,625 fresh windows, 264 have 2 new digests and 6 have
   3 (same-day updates). Choose the **latest pre-cutoff** version, never a later one.
7. **Timezones.**
   - CDX timestamps are 14-digit **UTC**.
   - Tip times are ET. Evening tips at 19:00 to 22:30 ET fall at 23:00 to 03:30 UTC, i.e. on the
     *next* UTC date. EDT (UTC-4) applies until the first Sunday of November and again from
     mid-March; EST (UTC-5) applies in between.
   - Convert the tip time to UTC from the schedule (`scheduled_start_utc`, or `gdtutc` + `utctm`).
     Never compare the ET date with the UTC date.
   - Of the earliest strict-window captures, 1,878 fall on the same UTC date as the ET game date,
     **722 on the UTC day before**, 25 on the UTC day after (before a late tip), and 135 on the
     game date after 22:00 UTC.
8. **The timestamp-only heuristic fails.** A rule of "new digest captured before 22:00 UTC on a
   game day" is weak in two ways. Only 2,016 of 3,916 (51%) such days, 2015-16 onward, are actual
   ET game days for that team. It would also lose about 30% of genuine pre-cutoff captures (the
   day-before and late-evening cases above). The BKN 2013 sample was captured 69 min *after* tip
   but before 22:00 UTC on the next UTC date. Schedules are available for every season from
   2012-13 (section 1.4), so use them instead.
9. **Lead time.** Deciles of (cutoff minus earliest new-digest capture) are 3.1, 6.5, 10.5, 14.1,
   16.6, 19.5, 22.7, 25.5 and 29.6 h. Almost all genuine captures are within 36 h, which supports
   the project's `rolling_max_age_hours: 36`.
10. **Query-string variants and URL normalisation.** Query these with a prefix query, or with
    `matchType=prefix` on `www.nba.com/gamenotes/{slug}.pdf`. Lakers: 904 of 1,061 rows; Warriors:
    24 `utm_*` rows. A Wayback replay of the variant URL returns the same blob.
11. **Text extraction.** Some packets, for example the Sixers in 2023-24 (Word export), have
    front pages with no extractable text. They need OCR, or the inner pages. This is not an archive
    problem, but it removes otherwise valid captures from the pipeline.
12. **Throttling looks like absence.** The project lost 143 replay fetches to
    `Connection refused`. In this audit, the Internet Archive answered heavy prefix queries with a
    "Temporarily Offline" HTML page (HTTP 200) and with 503/504. Treat these as missing data, check
    the body before parsing JSON, and retry slowly: 12 s or more between CDX requests, with
    exponential back-off.
13. **Capture volume is not stationary.** About 200 captures per month (Jan 2023 to Sep 2024)
    against 2 to 60 per month after Oct 2024 (Feb 2025: 110). Coverage reflects crawler behaviour,
    not team behaviour, so any per-season or per-team analysis inherits this selection.
14. **Truncated captures (Common Crawl).** 63 of 263 CC PDF records carry
    `"truncated": "length"`. Drop them: the digest refers to a partial payload.
15. **Gzip-encoded `id_` bodies.** For HTML pages the Wayback passes through
    `Content-Encoding: gzip`. A rendered (non-`id_`) replay rewrites links and pulls embedded
    resources from other timestamps, so do not use it as evidence.
16. **Postgame material on the same sites.** Old team sites also published *post*game notes, for
    example `www.nba.com/hornets/news/postgamenotes_magic_13oct2006.html`. Exclude any URL
    containing `postgame`.


---

## 5. Recommended next steps for the collector (not done here)

1. Re-run the project's replay fetch for the 133 team-game candidates that ended in
   `network_error`, using one worker, a 10 to 15 s gap and retries. This alone should add up to
   118 team-games in 2023-26.
2. Replace the per-URL `url=https://www.nba.com/gamenotes/{slug}.pdf&filter=statuscode:200`
   query with the prefix query from section 1.1, without the status filter. Include revisits.
   Drop placeholder digests.
3. Fix the Philadelphia lookup (647 distinct versions exist).
4. Extend the window back to March 2012, using the NBA and ESPN schedules listed in section 1.4.
   Apply the same header, date and record review. Expect roughly 1,980 more regular-season
   team-game candidates before content review, mostly from BOS, DET, PHI, LAL, DEN, MEM and MIA.
5. Add Common Crawl for 2016-17 to 2018-19 using `scripts/cc_rangeidx.py` (direct
   `cluster.idx` range reads; the public index API kept returning 504). It adds 51 games. Skip
   truncated records, and fetch bytes from `https://data.commoncrawl.org/{filename}` with
   `Range: bytes={offset}-{offset+length-1}`.
6. Add the Arquivo.pt CDX as a second source. Its digests are compatible with the Wayback's
   (54 shared), and it adds 18 games.
7. For HTML articles, store the raw `id_` capture of the article page only. Do not use the
   rendered replay: embedded scripts and JSON are pulled from *other* timestamps.

