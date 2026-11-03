# Game notes on nba.com team sites: map of all 30 teams

Research date: 2026-11-01 (2026-27 season under way). Probes ran from 13:05 to about 14:15 PT. I used a browser User-Agent, paced requests at 0.8 to 1.5 s, and made about 380 requests to nba.com and cdn.nba.com (mostly small JSON, sitemaps and HEAD requests) plus about 25 attempts to the Internet Archive (about half rejected with 429). Nothing was written under the repository or worktrees. Raw probe outputs are in the scratchpad (`raw/`), next to this file.

Legend: **[V]** = I fetched it and checked the result. **[I]** = inferred (from sitemap titles, heuristics, or the project's own reports). Unmarked statements in the tables come from fetched data unless noted.

## 0. Headline

* **Only two team sites publish HTML pregame game notes at scale.**
  * **Knicks:** category and tag `game-notes`, 170 posts **[V]**. These run from 2025-01-10 to today, and every post maps to a distinct NYK game: 2024-25 = 62, 2025-26 = 102, 2026-27 = 6 of 6 played so far.
  * **Trail Blazers:** "Game Notes: …" articles tagged `preview`, 237 in the sitemap **[V]**. Counts are 2021-22 = 79, 2023-24 = 81 and 2024-25 = 62. The series stopped after the 2025-10-08 preseason game, so there are 0 for 2025-26 and 2026-27.
  * **Timberwolves:** a negligible tag of 5 posts (2015 and 2020-21).
  * **The other 27 teams:** no `game-notes` category or tag. Their API returns HTTP 500 = nonexistent term **[V]**, and their sitemaps have no "game notes" titles since 2018 **[V]**.
* **Every team still publishes a rolling PDF**, `https://www.nba.com/gamenotes/{slug}.pdf`. All 30 return 200 to HEAD with a browser UA, with `Last-Modified` between 2026-10-29 and 2026-11-01 **[V]**.
  * `knicks.pdf` is **not** dead. It returns 403 only to the default curl/python User-Agent (Akamai rule) and 206/200 to a browser UA **[V]**.
  * So for 2026-27 onward, a scheduled "fetch-now" capture of the 30 rolling PDFs can cover nearly every game.
* **Dated CDN PDFs exist only for Cleveland.** The pattern `cdn.nba.com/teams/uploads/sites/{team_id}/YYYY/MM/YYMMDD-{away}{home}-notes.pdf` is still live for CLE in 2026-27: 5 of 6 games return 200, with Last-Modified on the morning of game day **[V]**. It returned 403 on all 58 guesses for the other 29 teams **[V]**.
* **The best enumerator is the JSON endpoint behind "Load More":**
  `https://www.nba.com/{slug}/api/content/{category|tag}/{term}?page=N` **[V]**.
  * Page size is fixed at 8, and each response includes full bodies and both timestamps. Knicks = 22 requests.
  * `?page=N` on the HTML category page is **ignored**: it always returns page 1 **[V]**.
  * For discovery across all teams, the per-team news sitemaps (`sitemap_team_{name}_news.xml`, 1 to 2 requests per team) list every article URL with title and publication date. They contain all 170 Knicks posts, with dates equal to the API `date` (UTC) in 170 of 170 **[V]**.
* **Knicks timestamp integrity is the weak point.**
  * 167 of 170 posts were published at least 60 minutes before tip (median lead about 10 h).
  * Only 60 of 170 have `modified` at or before the cutoff. 110 were modified after the game, but 90 of those 110 edits are bulk housekeeping (3 or more posts edited in the same UTC hour, a median of 15 days after the game).
  * Wayback has a pre-cutoff capture for 36 of 170 posts.
  * Posts verifiable today (CMS-clean or with a pre-cutoff capture): **22 of 62 (2024-25), 60 of 102 (2025-26), 0 of 6 (2026-27)**.
  * In **4 of 4** sampled pre-tip Wayback captures, the bulk post-game edits left the body text identical (3 cases) or only fixed a wrong-opponent heading (1 case). The edits added hyperlinks to highlight videos of earlier games. No results were inserted (section 4).
  * The 20 isolated (non-bulk) post-game edits remain untested.
* **Blazers integrity is much better.** I mapped 125 of their notes to the project schedule. All 125 were published at least 1 h before tip (lead deciles 3.2 to 9.1 h), 123 were never modified after the cutoff, and 97 were never modified at all **[V]**.

## 1. Per-team map

**Method [V]:**

* For each of the 30 slugs, I probed `/{slug}/api/content/category/game-notes?page=1` and `/{slug}/api/content/tag/game-notes?page=1` (60 requests).
* I downloaded every team news sitemap (35 files) and classified titles. I separated "game notes / pregame notes / media notes" from "postgame notes", and looked for recurring preview series.
* I sent a HEAD to each rolling PDF and read the team `_base` sitemaps for static notes pages.
* I guessed CLE-style dated PDF names for 2 games per team (section 5).

**How to read the table:**

* The *pregame preview series* column is **[I]**: a regex count of sitemap titles such as "Game Preview", "Gameday preview", "things to know", "shootaround" and "gameday update", by season (Aug–Jul).
  * These are team-written previews, not PR notes packets. They are pregame text, but they need their own leakage review.
  * Celtics counts are contaminated: "Keys to the Game: Celtics 125, Bulls 111" is a **post**game recap, and about 95 of 133 2025-26 hits carry final scores.
  * The Blazers' beat-writer previews ("Trail Blazers Host Kings…") do not match the regex. The Blazers `preview` tag holds 631 posts from 2017 to 2026 **[V]**.
* The *project-verified* column is from `reports/game_notes_collection/coverage.json` (`verified_by_publisher`): pre-cutoff archived rolling or dated PDF versions over the 2023-24 to 2025-26 regular seasons. These are versions, not distinct games. **[I]**: I read the report but did not re-verify it.
* "LM" = the HTTP `Last-Modified` of the rolling PDF on 2026-11-01. Rolling PDFs are overwritten per game, so they are only useful as fetch-now captures or through Wayback.

| Team | Slug / site ID | HTML game notes on team site (verified) | Rolling PDF `nba.com/gamenotes/{slug}.pdf` HEAD 2026-11-01 | Dated `cdn.nba.com/teams/uploads` PDF | Static page linking notes | Pregame preview series (count of titles 2024-25 / 2025-26 / 2026-27*) | Earliest news URL in sitemap | Project-verified pre-cutoff rolling-PDF versions 2023-26 |
|---|---|---|---|---|---|---|---|---|
| ATL | hawks / 1610612737 | None found (category/tag `game-notes` -> 500; no "game notes" titles 2018+ in sitemap) | 200, LM 01 Nov 2026 17:14:41 | No (2 guesses 403) | - | 0 / 0 / 0 | 2001-05-15 | 4 |
| BKN | nets / 1610612751 | None found (category/tag `game-notes` -> 500; no "game notes" titles 2018+ in sitemap) | 200, LM 01 Nov 2026 16:14:01 | No (2 guesses 403) | - | 0 / 0 / 0 | 2002-04-25 | 7 |
| BOS | celtics / 1610612738 | None found since 2007 (2005-07 HTML "game notes" in sitemap: 42) | 200, LM 31 Oct 2026 20:10:19 | No (2 guesses 403) | - | mostly postgame "Keys to the Game" recaps (173 / 123 / 10 raw hits); "Game Preview" 68 in 2023-24 only | 2001-01-08 | 68 |
| CHA | hornets / 1610612766 | None found (category/tag `game-notes` -> 500; no "game notes" titles 2018+ in sitemap) | 200, LM 01 Nov 2026 17:52:55 | No (2 guesses 403) | - | 0 / 0 / 0 | 2003-06-11 | 5 |
| CHI | bulls / 1610612741 | None found (category/tag `game-notes` -> 500; no "game notes" titles 2018+ in sitemap) | 200, LM 01 Nov 2026 17:33:39 | No (2 guesses 403) | [/bulls/game-notes](https://www.nba.com/bulls/game-notes) -> rolling PDF | 0 / 0 / 0 | 2000-10-28 | 4 |
| CLE | cavaliers / 1610612739 | None found (category/tag `game-notes` -> 500; no "game notes" titles 2018+ in sitemap) | 200, LM 01 Nov 2026 15:16:18 | Yes: `sites/1610612739/YYYY/MM/YYMMDD-{away}{home}-notes.pdf` (2026-27 still live) | [/cavaliers/team/game-notes-media-guide](https://www.nba.com/cavaliers/team/game-notes-media-guide) -> current dated PDF (page modified each game day) | 95 / 104 / 9 | 2002-09-25 | 57 |
| DAL | mavericks / 1610612742 | None found (category/tag `game-notes` -> 500; no "game notes" titles 2018+ in sitemap) | 200, LM 01 Nov 2026 17:56:17 | No (2 guesses 403) | - | 52 / 88 / 10 | 1998-10-30 | 8 |
| DEN | nuggets / 1610612743 | None found (category/tag `game-notes` -> 500; no "game notes" titles 2018+ in sitemap) | 200, LM 29 Oct 2026 19:09:15 | No (2 guesses 403) | - | 2 / 1 / 0 | 2000-08-18 | 59 |
| DET | pistons / 1610612765 | None found (category/tag `game-notes` -> 500; no "game notes" titles 2018+ in sitemap) | 200, LM 01 Nov 2026 17:48:45 | No (2 guesses 403) | - | 0 / 2 / 0 | 2001-11-02 | 73 |
| GSW | warriors / 1610612744 | None found (category/tag `game-notes` -> 500; no "game notes" titles 2018+ in sitemap) | 200, LM 31 Oct 2026 14:23:22 | No (2 guesses 403) | - | 95 / 84 / 7 | 2001-04-27 | 8 |
| HOU | rockets / 1610612745 | None found (category/tag `game-notes` -> 500; no "game notes" titles 2018+ in sitemap) | 200, LM 01 Nov 2026 17:08:00 | No (2 guesses 403) | - | 0 / 0 / 0 | 2000-01-04 | 5 |
| IND | pacers / 1610612754 | None found (category/tag `game-notes` -> 500; no "game notes" titles 2018+ in sitemap) | 200, LM 31 Oct 2026 18:18:55 | No (2 guesses 403) | - | 109 / 86 / 11 | 2000-04-10 | 6 |
| LAC | clippers / 1610612746 | None found (category/tag `game-notes` -> 500; no "game notes" titles 2018+ in sitemap) | 200, LM 31 Oct 2026 17:04:16 | No (2 guesses 403) | - | 72 / 80 / 5 | 2000-12-17 | 5 |
| LAL | lakers / 1610612747 | None pregame. "Postgame Notes" 2020-21 (80), which is postgame and unusable | 200, LM 01 Nov 2026 00:00:37 | No (2 guesses 403) | - | 23 / 0 / 0 | 2000-11-16 | 8 |
| MEM | grizzlies / 1610612763 | None found. "Bear Necessities: key notes, numbers and trends entering …" pregame series 2018-20 (57), discontinued | 200, LM 01 Nov 2026 04:31:52 | No (2 guesses 403) | - | 0 / 0 / 0 | 2002-04-24 | 55 |
| MIA | heat / 1610612748 | None found (category/tag `game-notes` -> 500; no "game notes" titles 2018+ in sitemap) | 200, LM 31 Oct 2026 15:57:23 | No (2 guesses 403) | - | 1 / 0 / 0 | 1998-04-20 | 7 |
| MIL | bucks / 1610612749 | None found (category/tag `game-notes` -> 500; no "game notes" titles 2018+ in sitemap) | 200, LM 01 Nov 2026 18:06:09 | No (2 guesses 403) | - | 0 / 0 / 0 | 2001-10-02 | 75 |
| MIN | timberwolves / 1610612750 | Minor. Tag `game-notes` = 5 posts (2015, 2020-21); 6 "Pregame Notes" 2015-16. `/timberwolves/game-notes-wolves-<opp>` | 200, LM 31 Oct 2026 20:17:15 | No (2 guesses 403) | - | 17 / 79 / 0 | 2001-10-31 | 3 |
| NOP | pelicans / 1610612740 | None since 2011 (legacy `news/notes.html` 2008-11; "Postgame Notes" 2014-15) | 200, LM 01 Nov 2026 17:49:57 | No (2 guesses 403) | [/pelicans/game-notes](https://www.nba.com/pelicans/game-notes) -> rolling PDF | 183 / 137 / 16 | 2000-02-08 | 2 |
| NYK | knicks / 1610612752 | **Yes.** Category + tag `game-notes` (170 posts, 2025-01-10 to now). `/knicks/news/{knicks-vs-<opp>-game-notes-<month>-<d>-<yyyy>[-suffix]}` | 200, LM 01 Nov 2026 18:58:51 | No (2 guesses 403) | - | 0 / 0 / 0 | 2001-12-01 | 3 |
| OKC | thunder / 1610612760 | None found (category/tag `game-notes` -> 500; no "game notes" titles 2018+ in sitemap) | 200, LM 01 Nov 2026 16:02:27 | No (2 guesses 403) | - | 0 / 0 / 0 | 2003-06-18 | 8 |
| ORL | magic / 1610612753 | None found (category/tag `game-notes` -> 500; no "game notes" titles 2018+ in sitemap) | 200, LM 01 Nov 2026 12:28:18 | No (2 guesses 403) | - | 89 / 91 / 9 | 1999-12-07 | 2 |
| PHI | sixers / 1610612755 | None pregame. "Postgame Notes" 2025-26 (9), postgame and unusable | 200, LM 01 Nov 2026 16:51:55 | No (2 guesses 403) | - | 67 / 97 / 7 | 2008-01-28 | 0 |
| PHX | suns / 1610612756 | None found (category/tag `game-notes` -> 500; no "game notes" titles 2018+ in sitemap) | 200, LM 30 Oct 2026 00:15:27 | No (2 guesses 403) | - | 75 / 86 / 3 | 1997-09-18 | 2 |
| POR | blazers / 1610612757 | **Yes, ended.** "Game Notes: …" articles, tag `preview` (not `game-notes`); 237 in sitemap: 2021-22 79, 2023-24 81, 2024-25 62, last 2025-10-08. `/blazers/news/game-notes-<teams-records>` | 200, LM 01 Nov 2026 19:44:55 | No (2 guesses 403) | - | tag `preview`: 631 posts 2017-2026 (beat-writer previews; regex count 0) | 2006-04-27 | 0 |
| SAC | kings / 1610612758 | None found (category/tag `game-notes` -> 500; no "game notes" titles 2018+ in sitemap) | 200, LM 01 Nov 2026 19:13:30 | No (2 guesses 403) | - | 0 / 1 / 0 | 2001-11-03 | 0 |
| SAS | spurs / 1610612759 | None found (category/tag `game-notes` -> 500; no "game notes" titles 2018+ in sitemap) | 200, LM 01 Nov 2026 18:17:23 | No (2 guesses 403) | - | 0 / 0 / 0 | 2001-06-14 | 0 |
| TOR | raptors / 1610612761 | None found (category/tag `game-notes` -> 500; no "game notes" titles 2018+ in sitemap) | 200, LM 01 Nov 2026 17:28:40 | No (2 guesses 403) | [/raptors/news-media/game-notes](https://www.nba.com/raptors/news-media/game-notes) -> rolling PDF | 86 / 82 / 0 | 2000-11-27 | 6 |
| UTA | jazz / 1610612762 | None found (category/tag `game-notes` -> 500; no "game notes" titles 2018+ in sitemap) | 200, LM 01 Nov 2026 18:33:47 | No (2 guesses 403) | - | 69 / 85 / 9 | 2006-06-16 | 1 |
| WAS | wizards / 1610612764 | None found (category/tag `game-notes` -> 500; no "game notes" titles 2018+ in sitemap) | 200, LM 01 Nov 2026 17:35:05 | No (2 guesses 403) | - | 2 / 0 / 0 | 2001-04-06 | 1 |

## 2. How to enumerate (verified mechanics)

### 2a. Team-site content API (recommended for timestamps and bodies) **[V]**

* **Endpoint.** `GET https://www.nba.com/{slug}/api/content/category/{term}?page=N`, or `/api/content/tag/{term}?page=N`.
  * Example: <https://www.nba.com/knicks/api/content/category/game-notes?page=2>.
  * This is the call the Next.js "Load More" button makes. Chunk `1432-*.js` builds `"/api/content/".concat(archiveType,"/").concat(slug)` with `{page}`. The build manifest rewrites `/{team}/api/:path*` to `/api/{team}/:path*`.
* **Response.** `{entity:{id,name,slug,taxonomy}, items:[…8…], pageNext, pages, total, count}`.
  * Knicks `category/game-notes`: entity id 1095, total 170, pages 22.
  * Knicks `tag/game-notes`: id 1096, also 170.
* **Each item includes:**
  * `id`, `slug`, `title`, `permalink`
  * `date` (published, UTC, editable in WP), `modified` (UTC), `timestamp` (ms epoch of `date`)
  * `taxonomy.{categories,tags,games,teams}`
  * `seo.canonicalUrl` (knicks.com mirror), `attributes.subhead`
  * **`contentStructured`**: the full body as HTML and text blocks.
* **Page size is fixed at 8.**
  * `per_page`, `perPage` and `limit` are ignored.
  * `count=50` triggers an Akamai "Access Denied" 500. Do not use it.
* **A nonexistent term returns HTTP 500 with an empty body.** That is how I tested all 30 teams for `category/game-notes` and `tag/game-notes`. I confirmed that a real category works (`/celtics/api/content/category/news` returns 200).
* **Transient 500s happen.** Blazers page 20 failed once, then succeeded on retry. Retry with backoff.
* **`?page=N` on the SSR HTML page does not paginate.** `https://www.nba.com/knicks/category/game-notes?page=3` returns page 1 items with `pageNext: 2`. The `__NEXT_DATA__` on the HTML page is page 1 only.
* **`taxonomy.games` is unreliable.** Editors copy it from the previous post: 29 of 170 Knicks posts point to the wrong game and 48 have none. Map posts to games by **title date + opponent**. Three titles carry a wrong date, for example "Pistons … February 5, 2026" for a game on 2026-02-06.

### 2b. Sitemaps (recommended for discovery across all teams) **[V]**

* **Hierarchy:**
  * <https://www.nba.com/robots.txt>
  * leads to `https://www.nba.com/sitemap_index.xml`
  * leads to `https://www.nba.com/sitemap_team_index.xml`
  * leads to `sitemap_team_{name}_index.xml`
  * leads to `sitemap_team_{name}_news.xml`, plus `_base`, `_games`, `_videos`, `_photos` and `_players`.
* **Naming quirks:**
  * PHI is `76ers` (`sitemap_team_sixers_news.xml` returns 403).
  * POR is listed literally as `sitemap_team_trail blazers_index.xml`, which works URL-encoded as `trail%20blazers`.
  * CLE, NOP and ORL split news into `_news_1.xml` and `_news_2.xml`.
* **Contents.** `news:publication_date` (UTC date of WP `date`) and `news:title` for every article, going back to 1997-2008 depending on team. Size is 1.9k to 16.8k URLs per team.
* **One crawl of all 30 = 35 requests**, about 70 MB uncompressed. This found the Blazers series, which uses tag `preview`, not `game-notes`. It also found the static game-notes pages in `_base`:
  * <https://www.nba.com/bulls/game-notes>
  * <https://www.nba.com/pelicans/game-notes>
  * <https://www.nba.com/raptors/news-media/game-notes>. These three just link the rolling PDF.
  * <https://www.nba.com/cavaliers/team/game-notes-media-guide>. This links the **current dated PDF** (`261029-cleatl-notes.pdf` on 2026-11-01). Its WP `modified` changes each game day, so Wayback captures of it can list historical CLE filenames.
* **The sitemap gives only a date, not a time.** Use the content API or the article page `__NEXT_DATA__.props.pageProps.pageObject.{date,modified}` for times.

### 2c. Things that do not work **[V]**

* **WordPress REST.**
  * `https://manage-teams.nba.com/knicks/wp-json/` returns 403 `rest_unauthenticated_user`.
  * `https://www.nba.com/knicks/wp-json/wp/v2/posts` returns 404.
* **Per-team sitemaps or robots under the team path.** `/knicks/sitemap.xml` and `/knicks/robots.txt` return 404.
* **`content-api-prod.nba.com`.** The public `public/1/leagues/nba/teams` endpoint works (team IDs and slugs), but team-site article listings are fetched server-side. The `/{slug}/api/content/...` proxy above is the practical public interface.
* **`cdn.nba.com/static/json/staticData/scheduleLeagueV2.json`** returned 403 to a plain request. I did not try to work around it.
* Use the team schedule proxy instead: `https://www.nba.com/{slug}/api/schedule?season=2025-26`. It returns the full season with playoffs, with `gameId` and `gameTimeUTC`. `season=2025` returns an empty list. With no parameter it returns the current season.

### 2d. Politeness note

* robots.txt disallows `/api/*` at the site root. The team proxy lives under `/{slug}/api/…`, which that rule does not literally match, but it is an internal endpoint. Keep request volume low; the whole Knicks archive is 22 requests.
* Responses are 240 to 360 KB per page because they include bodies.

## 3. Knicks in detail **[V]**

### 3a. Source facts

* **Listing.** <https://www.nba.com/knicks/category/game-notes>, with a tag twin at <https://www.nba.com/knicks/tags/game-notes>. Same 170 posts.
* **Enumeration.** `https://www.nba.com/knicks/api/content/category/game-notes?page=1..22` (22 requests, done 2026-11-01).
* **Format.** HTML only, no PDF links in any of the 170 bodies. Body length is 2.2k to 18.5k characters (median about 4.1k).
* **Typical sections:**
  * header line (date, time, arena, game number)
  * TV/Radio
  * "This Day in Knicks History"
  * "Last game played vs {opp}" (a recap of the previous meeting)
  * "Last game" (the Knicks' previous game)
  * player notes and streaks
  * series and playoff notes
  * Each body has a canonical mirror on `knicks.com/news/...`.
* **URL patterns over time:**
  * Jan 2025 to Mar 2025: `/knicks/news/game-notes-knicks-vs-{opp}-{month}-{d}-{yyyy}`
  * Mar 2025 onward: `/knicks/news/knicks-vs-{opp}-game-notes-{month}-{d}-{yyyy}`
  * Playoffs: `…-round-{r}-game-{g}`, `…-eastern-conference-finals-game-{g}` and `…-nba-finals-game-{g}`, with some variation such as `knicks-vs-celtics-round-2-game-1-notes-may-…`.
  * Bracketed slug suffixes are not guessable reliably. Enumerate; do not guess.
* **Timestamps.**
  * `date` is the published time. It is editable: for example `2025-01-17T12:00:00Z` with `modified` 2025-01-16T21:34Z is a scheduled post, and `2025-10-21T23:59:59Z` was clearly hand-set.
  * `modified` is the last save.
  * The sitemap `publication_date` equals the UTC date of `date` for 170 of 170.
  * The article page `__NEXT_DATA__.props.pageProps.pageObject` exposes the same `date` and `modified`.
* **Game mapping.** All 170 map to distinct NYK games via title date + opponent, checked against `/knicks/api/schedule?season=2024-25|2025-26` and the current season. Three title dates are off by 1 to 2 days. The `taxonomy.games` field is wrong for 29 posts and missing for 52.

### 3b. Counts by season

| Season | Posts | RS | PO | Other | Coverage of NYK games | Published at or before tip − 60 min | `modified` at or before cutoff | Any Wayback capture | Wayback capture at or before cutoff | Verifiable now (CMS-clean or pre-cutoff capture) |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 2024-25 | 62 | 44 | 18 | 0 | every game from 2025-01-10 (44 of 44 RS remaining, 18 of 18 PO). None before 2025-01-10 | 62 | 18 | 17 | 6 | 22 |
| 2025-26 | 102 | 82 | 19 | 1 (Cup final) | 82 of 82 RS, 19 of 19 PO, Cup final | 101 | 42 | 57 | 30 | 60 |
| 2026-27 (to 11-01) | 6 | 6 | 0 | 0 | 6 of 6 played | 4 | 0 | 0 | 0 | 0 |
| **Total** | **170** | 132 | 37 | 1 | | 167 | 60 | 74 | 36 | 82 |

* **Publication lead before tip.** Median 10.0 h (2024-25) and 9.6 h (2025-26), versus 4.5 h for 2026-27 so far.
* **Three posts carry a `date` after the cutoff:**
  * 2026-10-20 PHI@NYK: `date` 2026-10-23T20:00:50Z, 69 h after tip.
  * 2026-10-25 ORL@NYK: 2026-10-26T01:00:50Z, 2 h after tip.
  * 2026-06-03 Finals G1: 2026-06-05T19:06Z, 43 h after tip.
  * The identical `:00:50` seconds on the two October posts suggest a re-publish or a date reset. Without a Wayback capture, no pregame version can be shown for these.
* **Post-game modification.** 110 posts have `modified` after the cutoff, a median of 352 h (about 15 days) after tip.
  * 90 of the 110 fall in bulk-edit hours, meaning 3 or more posts edited in the same UTC hour: 2025-03-31T13–15 (19 posts), 2026-01-23T17–20 (23), 2026-05-12T17 (18), 2026-10-31T02 (6), and others.
  * This pattern points to housekeeping (taxonomy, game tagging, template fixes) rather than result insertion. Content diffs on samples are in section 4.

### 3c. Older seasons and other URL formats

* **No Knicks HTML game notes exist before 2025-01-10.** The first is a template draft, <https://www.nba.com/knicks/news/example-fast-break> (created 2024-12-17, modified 2025-01-10, content = the Jan 10, 2025 OKC notes). It duplicates `game-notes-knicks-vs-thunder-january-10-2025`.
* **The Knicks news sitemap has no other notes series.** Its only older items are:
  * `https://www.nba.com/knicks/news/notes.html` (2012-03-25, "Game Notes"), a legacy page. It now returns 301 with an empty body. Its Wayback CDX was not queried before I stopped because of Internet Archive 429s. By its date it predates the period of interest.
  * 2019-20 `knicks/matchups/knicks-vs-{opp}-{m}-{d}-{yy}` "Matchup Preview" pages. These were published weeks ahead (for example 2019-09-13 for November games) as ticketing pages, not notes.
* **For 2023-24 and earlier 2024-25, the only Knicks source is the rolling `https://www.nba.com/gamenotes/knicks.pdf`.** Wayback has 9 captures of it from 2023-10 to 2026-04, and the project verified 3 pre-cutoff versions (`reports/game_notes_collection/rolling_indexes.json`, `coverage.json`).
* **The Knicks category page itself has 3 Wayback captures:** 20250312193646, 20251225072514 and 20260618123740. Each embeds the 8 latest posts with full bodies.

## 4. Timestamp integrity: archived pregame version vs current version **[V]**

**Archive index (1 CDX request each):**

* **Knicks.** `http://web.archive.org/cdx/search/cdx?url=nba.com/knicks/news/&matchType=prefix&filter=original:.*game-notes.*&output=json` returned **160 captures**.
  * 74 of the 170 posts have at least one 200 capture.
  * 36 have a capture at or before tip − 60 min: 2024-25 = 6, 2025-26 = 30, 2026-27 = 0.
  * 22 posts have both a pre-cutoff capture and a post-cutoff `modified`. These are the ideal before-and-after test cases.
* **Blazers.** `url=nba.com/blazers/news/game-notes&matchType=prefix` returned **349 captures of 95 URLs**.
  * 2023-24: 70 of 82 posts captured, but only 9 before the cutoff.
  * 2024-25: 19 of 62 captured, 0 before the cutoff.
  * 2021-22: 2 of 79. Most of those URLs use a different slug prefix (`trail-blazers-game-notes-…`), which this query did not cover.
  * Blazers captures are mostly post-game, but Blazers CMS timestamps are clean (section 1, Appendix B).

**Method.**

* Replay the latest pre-cutoff capture with `https://web.archive.org/web/{ts}id_/{url}`.
* Parse `__NEXT_DATA__.props.pageProps.pageObject` from both the archived and the current page.
* Compare `contentStructured` block text, then all other fields.
* The archived `pageObject` reports its own `modified`. In all 4 samples that equals `date`, so each capture is the **original** publication.

| Post | Published | Wayback capture (UTC) | Tip (UTC) | Current `modified` | Body text archived → current | What changed |
|---|---|---|---|---|---|---|
| [NYK vs LAC, Mar 7 2025](https://www.nba.com/knicks/news/knicks-vs-clippers-game-notes-march-7-2025) | 03-07 17:24 | [20250307221908](https://web.archive.org/web/20250307221908/https://www.nba.com/knicks/news/knicks-vs-clippers-game-notes-march-7-2025) | 03-08 03:30 | 03-31 14:45 (bulk, +563 h) | 11 → 11 blocks, 2221 → 2223 characters, 1 line differs | Heading fixed from "Last game played vs **LA Lakers**:" to "vs **LA Clippers**:", the archived pregame text had a template error. A highlight link was added to the recap of the previous game (Lakers, 3/6). SEO fields were blanked. **No result of the target game.** |
| [NYK @ BKN, Nov 24 2025](https://www.nba.com/knicks/news/knicks-vs-nets-game-notes-november-24-2025) | 11-24 15:08 | [20251124161843](https://web.archive.org/web/20251124161843/https://www.nba.com/knicks/news/knicks-vs-nets-game-notes-november-24-2025) | 11-25 00:30 | 2026-01-20 17:20 (bulk, +1361 h) | **identical** (11 blocks, 2808 characters) | Only hyperlinks added: highlight videos of the **prior** NYK–BKN game (11-09-2025) and of the previous game (ORL 11-22-2025). An empty `pageComponents` field was removed. |
| [LAC @ NYK, Jan 7 2026](https://www.nba.com/knicks/news/knicks-vs-clippers-game-notes-january-7-2026) | 01-06 17:45 | [20260107070608](https://web.archive.org/web/20260107070608/https://www.nba.com/knicks/news/knicks-vs-clippers-game-notes-january-7-2026) | 01-08 00:30 | 01-23 19:49 (bulk, +379 h) | **identical** (11 blocks, 2971 characters) | Highlight links inserted into the 3/26/25 prior-meeting recap; `\xa0` became `&nbsp;`. |
| [PHI @ NYK, May 4 2026, R2 G1](https://www.nba.com/knicks/news/knicks-vs-76ers-game-notes-may-04-2026-round-2-game-1) | 05-04 13:22 | [20260504150247](https://web.archive.org/web/20260504150247/https://www.nba.com/knicks/news/knicks-vs-76ers-game-notes-may-04-2026-round-2-game-1) | 05-05 00:00 | 05-12 17:17 (bulk, +185 h) | **identical** (10 blocks, 3408 characters) | Whitespace/`&nbsp;` normalisation only. |
| [PHI @ NYK, Apr 1 2025](https://www.nba.com/knicks/news/knicks-vs-76ers-game-notes-april-2-2025) (slug says Apr 2) | 03-31 20:10 | 20250401174048 (exists in CDX) | 04-01 23:30 | 04-02 16:17 (**single edit +17 h**, not bulk) | **not compared**: Internet Archive returned HTTP 429 on every replay attempt from 13:57 to 14:07 PT | Unresolved. This is the most informative remaining case because the edit was isolated and came the morning after the game. |

**Conclusions.**

* **Bulk edits are cosmetic.** In 4 of 4 samples, the bulk post-game modifications changed no body text except one correction to a template heading. They added hyperlinks to highlight videos of **earlier** games and normalised whitespace. None inserted the target game's result. These bulk edits account for 90 of the 110 post-cutoff modifications.
* **The other 20 single post-game edits are untested.** For example, Apr 1 2025 was modified 17 h after tip. They need per-post diffs before use.
* **Stripping HTML is required.** The inserted `href`s carry other games' stat lines in their slugs, for example `highlights-towns-34-pts-knicks-vs-clippers`.
* **The archived pregame text itself contains errors** (Mar 7 2025: wrong opponent in a heading). It is pregame-valid but noisy. The project already quarantined a wrong-venue post from 2026-01-28.
* **`date` alone is not proof of availability.** Three posts show `date` after the cutoff, one shows `modified` < `date` (scheduled), and one shows a hand-set `23:59:59`. A third-party capture, or a CMS `modified` at or before the cutoff, is needed.
* **Internet Archive behaviour during this session.** It was "Temporarily Offline" (503) around 13:06 and 13:20 PT, then returned 429s intermittently. Other agents writing in this same scratchpad `research/` folder appear to be hitting it at the same time. Budget about 1 IA request per 20 to 60 s and retry on 429 with a 5-minute backoff.

## 5. Dated PDFs on `cdn.nba.com/teams/uploads/sites/{team_id}/YYYY/MM/` **[V]**

* **Cleveland control, 2026-27 regular season.** HEAD `…/1610612739/2026/10/{YYMMDD}-{away}{home}-notes.pdf`:

| Game | Status | Last-Modified (UTC) | Tip (UTC) |
|---|---|---|---|
| 261022-clephi | 200 | 2026-10-22 15:50:41 | 23:00 |
| 261024-bkncle | 200 | 2026-10-24 17:31:35 | 23:30 |
| 261026-mincle | 200 | 2026-10-26 14:52:52 | 23:00 |
| 261028-wascle | **403** (absent or other name) | – | 23:00 |
| 261029-cleatl | 200 | 2026-10-29 15:21:48 | 10-30 00:00 |
| 261031-clemil | 200 | 2026-10-31 15:30:19 | 11-01 00:00 |

* **Further controls:**
  * Jan 2026: `260123-saccle` returns 200, but `260121-clecha` returns **403**. Even CLE has gaps or irregular names, so do not assume completeness.
  * Older verified examples (200): `250221-nykcle` (LM 2025-02-21 15:37:40) and `260410-cleatl` (LM 2026-04-10 14:03:34).
* **Status 403 here means "no such object"** (S3/CloudFront). Last-Modified falls on the morning of game day, but the project correctly treats a retrospectively read header as insufficient proof on its own.
* **Other 29 teams.** I tried the same scheme under each team's own site ID (lowercase `{away}{home}`) for one home game and one away game between 2026-01-20 and 2026-03-20. **All 58 returned 403.** Example: <https://cdn.nba.com/teams/uploads/sites/1610612737/2026/01/260123-phxatl-notes.pdf>.
* **Conclusion.** The guessable dated-filename convention is a Cavaliers practice, not a league template.
  * Other teams upload only to the rolling `nba.com/gamenotes/{slug}.pdf`.
  * If any other team uses some other filename on the CDN, it would surface in a Wayback CDX prefix query over `cdn.nba.com/teams/uploads/sites/` filtered on `notes`. That query was queued, but see section 4 status: the Internet Archive was rate-limiting during this session.
* **The Cavaliers page <https://www.nba.com/cavaliers/team/game-notes-media-guide> is the discovery hook for CLE.** It currently links `…/2026/10/261029-cleatl-notes.pdf`, and its WP `modified` is 2026-10-29T15:22:19Z. Its Wayback captures would reveal irregular CLE filenames such as the missing 10/28 game.
  * **[V] CDX of that page:** 758 captures with status 200 and 727 distinct digests, from 2022-06-02 to 2026-01-06.
  * By season: 2023-24 = 340 captures, 2024-25 = 77, 2025-26 = 1.
  * Replaying the 2023-24 captures would list nearly every CLE dated filename that season, including irregular ones.
  * I did not replay them because the Internet Archive was rate-limiting.
* **The CDX prefix query over all of `cdn.nba.com/teams/uploads/sites/` filtered on `notes` failed.** It returned 504 twice, then an IncompleteRead, so it is too large for one call. Split it by team ID and year, with one request per minute.

## 6. Obtainable games per team per season (fetch now + archive checks)

**Definitions:**

* **"Available now"** = the document exists today on the team site.
* **"Verifiable"** = provably pre-cutoff today, meaning either:
  * CMS `date` and `modified` are both at or before tip − 60 min, which still trusts the publisher clock and assumes no unlogged edits; or
  * a Wayback 200 capture at or before the cutoff.
* Rolling-PDF historical counts are the project's existing verified pre-cutoff versions (2023-24 to 2025-26 regular seasons, all teams combined). **[I]**: taken from `reports/game_notes_collection/coverage.json`, not re-run here.
* Prospective 2026-27 counts **[I]** assume a scheduled fetch of every rolling PDF on game days, with fetch time and HTTP `Last-Modified` stored. Because all 30 rolling PDFs were live on 2026-11-01 **[V]**, the ceiling is about 82 regular-season games per team plus playoffs.

| Team | Source | 2023-24 | 2024-25 | 2025-26 | 2026-27 played to 11-01 | 2026-27 rest (prospective) |
|---|---|---|---|---|---|---|
| NYK | HTML category `game-notes` | 0 (HTML starts 2025-01-10) | **62** available (44 RS + 18 PO); **22** verifiable now | **102** available (82 RS + 19 PO + Cup); **60** verifiable now | 6 available, 4 published pre-cutoff, **0** verifiable (all modified 10-31; no captures yet) | about 76 RS + PO if captured at publish time (HTML posted a median of 4.5 to 10 h pre-tip) |
| NYK | rolling `knicks.pdf` | project: 3 verified versions over 2023-26 | (incl.) | (incl.) | live | about 76 + PO |
| POR | HTML "Game Notes" (tag `preview`) | **81** posts (75 map to project RS schedule; the rest are preseason or the last week); timestamps CMS-clean for 70 of 70 mapped via API; Wayback pending | **62** posts (58 RS); 55 of 55 mapped CMS-clean | 0 (series stopped; 1 preseason post 2025-10-08) | 0 | 0 HTML; rolling `blazers.pdf` only |
| POR | rolling `blazers.pdf` | project: 0 verified | – | – | live | about 76 + PO |
| CLE | dated CDN PDF + rolling | project: 57 verified versions over 2023-26 (dated pattern) | (incl.) | (incl.) | 5 of 6 dated PDFs present (header LM pre-tip; needs independent capture) | about 76 + PO (dated PDF and page) |
| MIN | HTML tag `game-notes` | 0 | 0 | 0 | 0 | rolling only |
| BOS, DEN, DET, MEM, MIL | rolling PDF (Wayback-rich) | project: 68 / 59 / 73 / 55 / 75 verified versions over 2023-26 | (incl.) | (incl.) | Wayback unknown (not queried) | about 76 + PO each |
| 20 other teams | rolling PDF (Wayback-sparse) | project: 0 to 8 verified versions each over 2023-26 (ATL 4, BKN 7, CHA 5, CHI 4, DAL 8, GSW 8, HOU 5, IND 6, LAC 5, LAL 8, MIA 7, NOP 2, OKC 8, ORL 2, PHX 2, TOR 6, UTA 1, WAS 1, PHI 0 (index lookup failed), SAC 0, SAS 0) | (incl.) | (incl.) | unknown | about 76 + PO each |

**Season totals from team-site HTML only (new relative to the project's 404):**

| Season | Available on team sites now | Verifiable now |
|---|---|---|
| 2023-24 | about 81 games (POR) | about 70+ (POR CMS-clean) |
| 2024-25 | about 124 (NYK 62 + POR 62; at most 2 shared NYK–POR games) | about 22 (NYK) + about 55–62 (POR, CMS-clean) |
| 2025-26 | 102 (NYK) | 60 |
| 2026-27 to date | 6 | 0 |

**Notes:**

* **The Knicks "verifiable" number can rise toward "available"** if the bulk post-game edits prove cosmetic. In the one completed diff, the only change fixed a wrong-opponent heading: "Last game played vs LA Lakers" became "vs LA Clippers" (section 4). That is not results leakage, but it shows the archived pregame text can itself contain template errors.
* **Each Knicks post also embeds two retrospective blocks:** "Last game" (the previous Knicks game, a recap) and "Last game played vs {opp}". These are pregame-legitimate, but they are outcome text about *earlier* games and must be time-checked against the target game.
* **The 2026-27 opportunity is prospective capture, not team-site HTML.** Only NYK still writes HTML notes; POR stopped, and the other 27 teams never did. All 30 rolling PDFs plus the CLE dated PDFs are live.
* **The union of HTML team-site sources adds about 300 game-team documents** across 2023-27 that the existing 404-game set mostly lacks. The project currently has only 3 NYK and 0 POR verified rolling versions.

## 7. Recommended enumeration and collection method

1. **Discovery (once per season, about 35 requests).**
   * Pull `sitemap_team_index.xml` and each team's `*_news*.xml`.
   * Filter titles on `game notes|pregame notes|media notes`. Exclude `postgame`.
   * Watch for new series. A team may start an HTML series the way the Knicks did in January 2025.
2. **HTML notes: timestamps and bodies.** For each series with a taxonomy term, page through `https://www.nba.com/{slug}/api/content/{category|tag}/{term}?page=1..pages`, 8 items per page:
   * Knicks: `knicks` + `category/game-notes`, 22 pages.
   * Blazers: `blazers` + `tag/preview`, 79 pages, then filter titles on "Game Notes". About 95 Blazers notes (mostly 2021-22) lack the tag. Get their timestamps from the article page `__NEXT_DATA__`; their URLs are in the sitemap.
   * Store `date`, `modified`, `id`, the raw item JSON and the fetch time.
   * Map to games by title date + opponent against `/{slug}/api/schedule?season=YYYY-YY`. Never use `taxonomy.games` or the slug date.
3. **Archive verification.**
   * One CDX prefix query per team path, for example `url=nba.com/knicks/news/&matchType=prefix&filter=original:.*game-notes.*`, which returned 160 captures for the Knicks in one call. Then take the latest 200 capture at or before tip − 60 min, replay it with `id_` and parse `pageObject`.
   * Also check the Knicks category page captures. Each one carries 8 post bodies.
   * Treat CMS-clean posts (`modified` at or before the cutoff) as a separate, weaker evidence tier from a Wayback capture.
4. **Prospective 2026-27: start now.**
   * Each game day, poll all 30 `https://www.nba.com/gamenotes/{slug}.pdf` with a browser UA, for example every 2 to 3 h from 10:00 ET until tip − 60 min. The default curl UA gets 403.
   * Also poll the Knicks API page 1 and the Cavaliers dated PDF and page.
   * Save the bytes, SHA-256, fetch time and `Last-Modified`. Optionally submit each to Save Page Now for third-party timestamping.
   * This is the only route to near-league-wide coverage. Team-site HTML covers 1 team now, versus 2 at the 2023-25 peak.
5. **Content hygiene for model input.**
   * Strip HTML; post-game edits add highlight-video hyperlinks.
   * Keep "Last game" and "Last game played vs" recaps only if those games are before the target game.
   * Quarantine template errors: wrong opponent headings, stale venues, wrong slugs and dates (the project already found a wrong-venue Knicks post).
   * Flag any post whose `date` is after the cutoff (3 Knicks posts).

## Appendix B. Trail Blazers "Game Notes" series **[V]**

* **Example.** <https://www.nba.com/blazers/news/game-notes-portland-trail-blazers-35-44-14-26-away-utah-jazz-16-63-9-30-home>
  * `date` = `modified` = 2025-04-09T19:55:12Z; tip 2025-04-10T01:00Z.
  * Taxonomy: category `news`, tag `preview`, team 1610612757. No `game-notes` term exists on this site.
  * Body (about 4.1k characters): header (records, date, time, arena), "SERIES NOTES" (all-time, home and away, streak, last wins), "TRAIL BLAZERS NOTES" (player form from the previous game, streaks), and opponent notes.
* **Title and URL patterns:**
  * 2021-22: "Trail Blazers Game Notes: {Away} (W-L) vs. {Home} (W-L)".
  * 2023-25: "Game Notes: {Team} (W-L[, home/away W-L]) {vs|@|at} {Team} (…)[, 7:00 p.m.]".
  * Slugs embed both records, for example `game-notes-portland-trail-blazers-10-25-at-new-york-knicks-21-15-430-pm`. They cannot be guessed; enumerate them.
* **Counts from the sitemap** (`https://www.nba.com/sitemap_team_trail%20blazers_news.xml`, by Aug–Jul season, pregame only):

| Season | Posts | Months |
|---|---:|---|
| ≤2020-21 (sporadic "Pregame Notes") | 13 | 2010–2020 |
| 2021-22 | 79 | 2021-10 to 2022-04 |
| 2022-23 | 1 | 2022-10 |
| 2023-24 | 81 | 2023-10 to 2024-04 (75 map to project RS schedule) |
| 2024-25 | 62 | 2024-10 to 2025-04 (58 map to project RS schedule) |
| 2025-26 | 1 | 2025-10-08 preseason, then stopped |

* **Tag enumeration.** `https://www.nba.com/blazers/api/content/tag/preview?page=12..40` (29 requests). The `preview` tag has 631 posts and 79 pages: beat-writer previews from 2017 to 2026, plus the notes.
  * 142 notes carry the tag: 11 in 2021-22, 73 in 2023-24, 58 in 2024-25.
  * 125 of them map to project-schedule games. All 125 were published 1 h or more before tip (lead deciles 3.2 to 9.1 h, median about 6.4 h).
  * 123 of 125 have `modified` at or before the cutoff, and 97 of 125 have `modified == date`.
  * Two were modified after tip: 2024-02-04 POR@DEN, 8 seconds after tip, and 2024-03-22 DEN@POR, 18 h after.
* **Gaps.** Most 2021-22 notes lack the tag. The tag pages show their 2021 items with `modified` 2021-09-27 (a CMS migration). Get timestamps for untagged notes from each article page.
* **Wayback.** CDX prefix `nba.com/blazers/news/game-notes`, results in section 4.

## Appendix A. Knicks post list (all 170, via `/knicks/api/content/category/game-notes?page=1..22`) **[V]**

* "Pub lead" = hours from `date` to scheduled tip. A negative value means published after tip.
* "Mod vs cutoff" compares `modified` to tip − 60 min.
* "Wayback pre-cutoff" = the latest 200 capture at or before the cutoff. "post-only" = captured only after the cutoff.
* Types: RS = regular season, PO = playoffs.
* Games come from `/knicks/api/schedule?season=…`, matched by title date + opponent.

#### 2024-25

| Game (ET) | Matchup | Type | Title | Published (UTC) | Modified (UTC) | Pub lead (h) | Mod vs cutoff | Wayback pre-cutoff | Flags |
|---|---|---|---|---|---|---:|---|---|---|
| 2025-01-10 | OKC@NYK | RS | [Game Notes: Knicks vs Thunder January 10, 2025](https://www.nba.com/knicks/news/game-notes-knicks-vs-thunder-january-10-2025) | 2025-01-10T14:25 | 2025-01-15T15:29 | 10.1 | after (+111h) | - |  |
| 2025-01-12 | MIL@NYK | RS | [Game Notes: Knicks vs Bucks January 12, 2025](https://www.nba.com/knicks/news/game-notes-knicks-vs-bucks-january-12-2025) | 2025-01-12T14:38 | 2025-01-16T14:46 | 5.4 | after (+91h) | post-only |  |
| 2025-01-13 | DET@NYK | RS | [Game Notes: Knicks vs Pistons January 13, 2025](https://www.nba.com/knicks/news/game-notes-knicks-vs-pistons-january-13-2025) | 2025-01-13T17:13 | 2025-01-16T14:39 | 7.3 | after (+62h) | post-only |  |
| 2025-01-15 | NYK@PHI | RS | [Game Notes: Knicks vs 76ers January 15, 2025](https://www.nba.com/knicks/news/game-notes-knicks-vs-76ers-january-15-2025) | 2025-01-15T12:00 | 2025-01-16T14:39 | 12.0 | after (+15h) | - |  |
| 2025-01-17 | MIN@NYK | RS | [Game Notes: Knicks vs Timberwolves January 17, 2025](https://www.nba.com/knicks/news/game-notes-knicks-vs-timberwolves-january-17-2025) | 2025-01-17T12:00 | 2025-01-16T21:34 | 12.5 | before | - |  |
| 2025-01-20 | ATL@NYK | RS | [Game Notes: Knicks vs Hawks January 20, 2025](https://www.nba.com/knicks/news/game-notes-knicks-vs-hawks-january-20-2025) | 2025-01-20T14:46 | 2025-03-31T15:16 | 5.2 | after (+1675h) | - |  |
| 2025-01-21 | NYK@BKN | RS | [Game Notes: Knicks vs Nets January 21, 2025](https://www.nba.com/knicks/news/game-notes-knicks-vs-nets-january-21-2025) | 2025-01-21T18:12 | 2025-03-31T15:15 | 6.3 | after (+1647h) | - | games-taxonomy wrong |
| 2025-01-25 | SAC@NYK | RS | [Game Notes: Knicks vs Kings January 25, 2025](https://www.nba.com/knicks/news/game-notes-knicks-vs-kings-january-25-2025) | 2025-01-25T15:25 | 2025-03-31T15:12 | 9.1 | after (+1551h) | - |  |
| 2025-01-27 | MEM@NYK | RS | [Game Notes: Knicks vs Grizzlies January 27, 2025](https://www.nba.com/knicks/news/game-notes-knicks-vs-grizzlies-january-27-2025) | 2025-01-27T15:21 | 2025-03-31T15:10 | 9.1 | after (+1503h) | - |  |
| 2025-01-29 | DEN@NYK | RS | [Knicks vs Nuggets Game Notes: January 29, 2025](https://www.nba.com/knicks/news/game-notes-knicks-vs-nuggets-january-29-2025) | 2025-01-29T14:38 | 2025-02-09T00:31 | 9.9 | after (+240h) | - |  |
| 2025-02-01 | LAL@NYK | RS | [Knicks vs Lakers Game Notes: February 1, 2025](https://www.nba.com/knicks/news/knicks-vs-lakers-game-notes-february-1-2025) | 2025-02-01T11:00 | 2025-02-09T00:31 | 14.5 | after (+167h) | post-only |  |
| 2025-02-03 | HOU@NYK | RS | [Knicks vs Rockets Game Notes: February 3, 2025](https://www.nba.com/knicks/news/knicks-vs-rockets-game-notes-february-3-2025) | 2025-02-03T15:58 | 2025-02-09T00:30 | 8.5 | after (+120h) | - |  |
| 2025-02-04 | NYK@TOR | RS | [Knicks vs Raptors Game Notes: February 4, 2025](https://www.nba.com/knicks/news/knicks-vs-raptors-game-notes-february-4-2025) | 2025-02-04T17:45 | 2025-03-31T15:06 | 6.7 | after (+1311h) | post-only |  |
| 2025-02-08 | BOS@NYK | RS | [Knicks vs Celtics Game Notes: February 8, 2025](https://www.nba.com/knicks/news/knicks-vs-celtics-game-notes-february-8-2025) | 2025-02-08T16:09 | 2025-03-31T15:04 | 9.3 | after (+1214h) | post-only |  |
| 2025-02-11 | NYK@IND | RS | [Knicks vs Pacers Game Notes: February 11, 2025](https://www.nba.com/knicks/news/knicks-vs-pacers-game-notes-february-11-2025) | 2025-02-11T15:01 | 2025-02-11T15:02 | 9.5 | before | 20250211192636 |  |
| 2025-02-12 | ATL@NYK | RS | [Knicks vs Hawks Game Notes: February 12, 2025](https://www.nba.com/knicks/news/knicks-vs-hawks-game-notes-february-12-2025) | 2025-02-12T19:05 | 2025-02-12T19:05 | 5.4 | before | - |  |
| 2025-02-20 | CHI@NYK | RS | [Knicks vs Bulls Game Notes: February 20, 2025](https://www.nba.com/knicks/news/knicks-vs-bulls-game-notes-february-20-2025) | 2025-02-20T11:00 | 2025-03-31T15:01 | 13.5 | after (+927h) | - |  |
| 2025-02-21 | NYK@CLE | RS | [Knicks vs Cavaliers Game Notes: February 21, 2025](https://www.nba.com/knicks/news/knicks-vs-cavaliers-game-notes-february-21-2025) | 2025-02-21T17:03 | 2025-03-31T15:00 | 6.9 | after (+903h) | post-only |  |
| 2025-02-23 | NYK@BOS | RS | [Knicks vs Celtics Game Notes: February 23, 2025](https://www.nba.com/knicks/news/knicks-vs-celtics-game-notes-february-23-2025) | 2025-02-23T12:41 | 2025-03-31T14:59 | 5.3 | after (+861h) | - |  |
| 2025-02-26 | PHI@NYK | RS | [Knicks vs 76ers Game Notes: February 26, 2025](https://www.nba.com/knicks/news/knicks-vs-76ers-game-notes-february-26-2025) | 2025-02-25T21:48 | 2025-03-31T14:55 | 26.2 | after (+783h) | - |  |
| 2025-02-28 | NYK@MEM | RS | [Knicks vs Grizzlies Game Notes: February 28, 2025](https://www.nba.com/knicks/news/knicks-vs-grizzlies-game-notes-february-28-2025) | 2025-02-28T14:33 | 2025-03-02T14:38 | 10.4 | after (+38h) | - |  |
| 2025-03-02 | NYK@MIA | RS | [Knicks vs Heat Game Notes: March 2, 2025](https://www.nba.com/knicks/news/knicks-vs-heat-game-notes-march-2-2025) | 2025-03-02T14:45 | 2025-03-31T14:54 | 8.2 | after (+688h) | post-only |  |
| 2025-03-04 | GSW@NYK | RS | [Knicks vs Warriors Game Notes: March 4, 2025](https://www.nba.com/knicks/news/knicks-vs-warriors-game-notes-march-4-2025) | 2025-03-03T21:20 | 2025-03-31T14:49 | 27.2 | after (+638h) | - |  |
| 2025-03-06 | NYK@LAL | RS | [Knicks vs Lakers Game Notes: March 6, 2025](https://www.nba.com/knicks/news/knicks-vs-lakers-game-notes-march-6-2025) | 2025-03-06T15:21 | 2025-03-31T14:46 | 11.6 | after (+588h) | post-only |  |
| 2025-03-07 | NYK@LAC | RS | [Knicks vs Clippers Game Notes: March 7, 2025](https://www.nba.com/knicks/news/knicks-vs-clippers-game-notes-march-7-2025) | 2025-03-07T17:24 | 2025-03-31T14:45 | 10.1 | after (+563h) | 20250307221908 |  |
| 2025-03-10 | NYK@SAC | RS | [Knicks vs Kings Game Notes: March 10, 2025](https://www.nba.com/knicks/news/knicks-vs-kings-game-notes-march-10-2025) | 2025-03-10T14:01 | 2025-03-31T14:43 | 12.5 | after (+492h) | - |  |
| 2025-03-12 | NYK@POR | RS | [Knicks vs Trailblazers Game Notes: March 12, 2025](https://www.nba.com/knicks/news/knicks-vs-trailblazers-game-notes-march-12-2025) | 2025-03-12T13:59 | 2025-03-31T14:37 | 12.0 | after (+445h) | - |  |
| 2025-03-15 | NYK@GSW | RS | [Knicks vs Warriors Game Notes: March 15, 2025](https://www.nba.com/knicks/news/knicks-vs-warriors-game-notes-march-15-2025) | 2025-03-14T20:56 | 2025-03-31T13:37 | 27.6 | after (+373h) | post-only |  |
| 2025-03-17 | MIA@NYK | RS | [Knicks vs Heat Game Notes: March 17, 2025](https://www.nba.com/knicks/news/knicks-vs-heat-game-notes-march-17-2025) | 2025-03-17T14:23 | 2025-03-31T13:37 | 9.1 | after (+326h) | 20250317212753 |  |
| 2025-03-19 | NYK@SAS | RS | [Knicks vs Spurs Game Notes: March 19, 2025](https://www.nba.com/knicks/news/knicks-vs-spurs-game-notes-march-19-2025) | 2025-03-19T14:11 | 2025-03-27T15:27 | 9.8 | after (+183h) | post-only |  |
| 2025-03-20 | NYK@CHA | RS | [Knicks vs Hornets Game Notes: March 20, 2025](https://www.nba.com/knicks/news/knicks-vs-hornets-game-notes-march-20-2025) | 2025-03-20T15:56 | 2025-03-27T15:14 | 7.1 | after (+160h) | post-only |  |
| 2025-03-22 | WAS@NYK | RS | [Knicks vs Wizards Game Notes: March 22, 2025](https://www.nba.com/knicks/news/knicks-vs-wizzards-game-notes-march-21-2025) | 2025-03-21T19:36 | 2025-03-27T15:12 | 28.4 | after (+111h) | 20250322091138 |  |
| 2025-03-25 | DAL@NYK | RS | [Knicks vs Mavericks Game Notes: March 25, 2025](https://www.nba.com/knicks/news/knicks-vs-mavericks-game-notes-march-25-2025) | 2025-03-25T14:28 | 2025-03-27T15:11 | 9.0 | after (+40h) | - |  |
| 2025-03-26 | LAC@NYK | RS | [Knicks vs Clippers Game Notes: March 26, 2025](https://www.nba.com/knicks/news/knicks-vs-clippers-game-notes-march-26-2025) | 2025-03-26T17:18 | 2025-03-27T15:10 | 6.2 | after (+16h) | - |  |
| 2025-03-28 | NYK@MIL | RS | [Knicks vs Bucks Game Notes: March 28, 2025](https://www.nba.com/knicks/news/knicks-vs-bucks-game-notes-march-28-2025) | 2025-03-27T20:55 | 2025-03-27T20:58 | 27.1 | before | - |  |
| 2025-03-30 | POR@NYK | RS | [Knicks vs Trail Blazers Game Notes: March 30, 2025](https://www.nba.com/knicks/news/knicks-vs-trail-blazers-game-notes-march-30-2025) | 2025-03-30T13:50 | 2025-03-31T13:33 | 8.2 | after (+16h) | - |  |
| 2025-04-01 | PHI@NYK | RS | [Knicks vs 76ers Game Notes: April 1, 2025](https://www.nba.com/knicks/news/knicks-vs-76ers-game-notes-april-2-2025) | 2025-03-31T20:10 | 2025-04-02T16:17 | 27.3 | after (+17h) | 20250401174048 |  |
| 2025-04-02 | NYK@CLE | RS | [Knicks vs Cavaliers Game Notes: April 2, 2025](https://www.nba.com/knicks/news/knicks-vs-cavaliers-game-notes-april-2-2025) | 2025-04-02T17:04 | 2025-04-02T18:17 | 5.9 | before | 20250402203317 |  |
| 2025-04-05 | NYK@ATL | RS | [Knicks vs Hawks Game Notes: April 5, 2025](https://www.nba.com/knicks/news/knicks-vs-hawks-game-notes-april-5-2025) | 2025-04-04T20:50 | 2025-04-07T15:58 | 22.2 | after (+45h) | - |  |
| 2025-04-06 | PHX@NYK | RS | [Knicks vs Suns Game Notes: April 6, 2025](https://www.nba.com/knicks/news/knicks-vs-suns-game-notes-april-6-2025) | 2025-04-06T16:24 | 2025-04-06T16:30 | 6.6 | before | - |  |
| 2025-04-08 | BOS@NYK | RS | [Knicks vs Celtics Game Notes: April 8, 2025](https://www.nba.com/knicks/news/knicks-vs-celtics-game-notes-april-8-2025) | 2025-04-08T13:54 | 2025-04-08T14:06 | 9.6 | before | - |  |
| 2025-04-10 | NYK@DET | RS | [Knicks vs Pistons Game Notes: April 10, 2025](https://www.nba.com/knicks/news/knicks-vs-pistons-game-notes-april-10-2025) | 2025-04-10T14:38 | 2025-04-10T14:41 | 8.4 | before | - |  |
| 2025-04-11 | CLE@NYK | RS | [Knicks vs Cavaliers Game Notes: April 11, 2025](https://www.nba.com/knicks/news/knicks-vs-cavaliers-game-notes-april-11-2025) | 2025-04-11T18:07 | 2025-04-11T18:07 | 5.4 | before | - |  |
| 2025-04-13 | NYK@BKN | RS | [Knicks vs Nets Game Notes: April 13, 2025](https://www.nba.com/knicks/news/knicks-vs-nets-game-notes-april-13-2025) | 2025-04-13T13:20 | 2025-04-13T13:20 | 3.7 | before | - |  |
| 2025-04-19 | DET@NYK | PO | [Knicks vs Pistons Round 1, Game 1 Notes: April 19, 2025](https://www.nba.com/knicks/news/knicks-vs-pistons-playoff-game-1-notes-april-19-2025) | 2025-04-19T00:08 | 2025-05-05T13:25 | 21.9 | after (+375h) | - |  |
| 2025-04-21 | DET@NYK | PO | [Knicks vs Pistons Round 1, Game 2 Notes: April 21, 2025](https://www.nba.com/knicks/news/knicks-vs-pistons-playoff-game-1-notes-april-19-2025-2) | 2025-04-21T13:41 | 2025-05-05T13:25 | 9.8 | after (+326h) | - |  |
| 2025-04-24 | NYK@DET | PO | [Knicks vs Pistons Round 1, Game 3 Notes: April 24, 2025](https://www.nba.com/knicks/news/knicks-vs-pistons-playoff-game-3-notes-april-24-2025) | 2025-04-23T18:43 | 2025-05-05T13:25 | 28.3 | after (+254h) | - |  |
| 2025-04-27 | NYK@DET | PO | [Knicks vs Pistons Round 1, Game 4 Notes: April 27, 2025](https://www.nba.com/knicks/news/knicks-vs-pistons-playoff-game-5-notes-april-27-2025) | 2025-04-27T13:09 | 2025-05-05T13:25 | 3.8 | after (+188h) | - |  |
| 2025-04-29 | DET@NYK | PO | [Knicks vs Pistons Round 1, Game 5 Notes: April 29, 2025](https://www.nba.com/knicks/news/knicks-vs-pistons-playoff-game-5-notes-april-27-2025-2) | 2025-04-28T18:26 | 2025-05-05T13:24 | 29.1 | after (+134h) | - |  |
| 2025-05-01 | NYK@DET | PO | [Knicks vs Pistons Round 1, Game 6 Notes: May 1, 2025](https://www.nba.com/knicks/news/knicks-vs-pistons-playoff-game-6-notes-may-1-2025) | 2025-04-30T18:26 | 2025-05-05T13:24 | 29.1 | after (+86h) | - |  |
| 2025-05-05 | NYK@BOS | PO | [Knicks vs Celtics Round 2, Game 1 Notes: May 5, 2025](https://www.nba.com/knicks/news/knicks-vs-celtics-round-2-game-1-notes-may-5-2025) | 2025-05-05T13:37 | 2025-05-09T17:20 | 9.4 | after (+90h) | - |  |
| 2025-05-07 | NYK@BOS | PO | [Knicks vs Celtics Round 2, Game 2 Notes: May 7, 2025](https://www.nba.com/knicks/news/knicks-vs-celtics-round-2-game-2-notes-may-7-2025) | 2025-05-06T20:02 | 2025-05-09T17:24 | 26.9 | after (+42h) | - |  |
| 2025-05-10 | BOS@NYK | PO | [Knicks vs Celtics Round 2, Game 3 Notes: May 10, 2025](https://www.nba.com/knicks/news/knicks-vs-celtics-round-2-game-3-notes-may-10-2025) | 2025-05-09T17:17 | 2025-05-09T17:18 | 26.2 | before | - |  |
| 2025-05-12 | BOS@NYK | PO | [Knicks vs Celtics Round 2, Game 4 Notes: May 12, 2025](https://www.nba.com/knicks/news/knicks-vs-celtics-round-2-game-4-notes-may-12-2025) | 2025-05-12T14:32 | 2025-05-12T14:32 | 9.0 | before | - |  |
| 2025-05-14 | NYK@BOS | PO | [Knicks vs Celtics Round 2, Game 5 Notes: May 14, 2025](https://www.nba.com/knicks/news/knicks-vs-celtics-round-2-game-5-notes-may-14-2025) | 2025-05-13T18:47 | 2025-05-13T18:47 | 28.2 | before | - |  |
| 2025-05-16 | BOS@NYK | PO | [Knicks vs Celtics Round 2, Game 6 Notes: May 16, 2025](https://www.nba.com/knicks/news/knicks-vs-celtics-round-2-game-6-notes-may-16-2025) | 2025-05-16T12:10 | 2025-05-16T12:10 | 11.8 | before | - |  |
| 2025-05-21 | IND@NYK | PO | [Knicks vs Pacers Eastern Conference Finals, Game 1 Notes: May 21, 2025](https://www.nba.com/knicks/news/knicks-vs-celtics-eastern-conference-finals-game-1-notes-may-21-2025) | 2025-05-21T13:53 | 2025-05-22T20:58 | 10.1 | after (+21h) | - |  |
| 2025-05-23 | IND@NYK | PO | [Knicks vs Pacers Eastern Conference Finals, Game 2 Notes: May 23, 2025](https://www.nba.com/knicks/news/knicks-vs-pacers-eastern-conference-finals-game-2-notes-may-23-2025) | 2025-05-22T21:31 | 2025-05-23T14:06 | 26.5 | before | - |  |
| 2025-05-25 | NYK@IND | PO | [Knicks vs Pacers Eastern Conference Finals, Game 3 Notes: May 25, 2025](https://www.nba.com/knicks/news/knicks-vs-pacers-eastern-conference-finals-game-3-notes-may-25-2025) | 2025-05-25T13:47 | 2025-05-25T13:47 | 10.2 | before | - | games-taxonomy wrong |
| 2025-05-27 | NYK@IND | PO | [Knicks vs Pacers Eastern Conference Finals, Game 4 Notes: May 27, 2025](https://www.nba.com/knicks/news/knicks-vs-pacers-eastern-conference-finals-game-4-notes-may-27-2025) | 2025-05-27T13:38 | 2025-05-27T13:40 | 10.3 | before | - |  |
| 2025-05-29 | IND@NYK | PO | [Knicks vs Pacers Eastern Conference Finals, Game 5 Notes: May 29, 2025](https://www.nba.com/knicks/news/knicks-vs-pacers-eastern-conference-finals-game-5-notes-may-29-2025) | 2025-05-29T13:56 | 2025-05-29T23:44 | 10.1 | after (+-0h) | - |  |
| 2025-05-31 | NYK@IND | PO | [Knicks vs Pacers Eastern Conference Finals, Game 6 Notes: May 31, 2025](https://www.nba.com/knicks/news/knicks-vs-pacers-eastern-conference-finals-game-6-notes-may-31-2025) | 2025-05-30T17:35 | 2025-05-30T17:35 | 30.4 | before | - |  |

#### 2025-26

| Game (ET) | Matchup | Type | Title | Published (UTC) | Modified (UTC) | Pub lead (h) | Mod vs cutoff | Wayback pre-cutoff | Flags |
|---|---|---|---|---|---|---:|---|---|---|
| 2025-10-22 | CLE@NYK | RS | [Knicks vs Cavaliers Game Notes: October 21, 2025](https://www.nba.com/knicks/news/knicks-vs-cavaliers-game-notes-october-21-2025) | 2025-10-21T23:59 | 2025-10-22T23:02 | 23.0 | after (+0h) | - | title date 2025-10-21 != game |
| 2025-10-24 | BOS@NYK | RS | [Knicks vs Celtics Game Notes: October 24, 2025](https://www.nba.com/knicks/news/knicks-vs-celtics-game-notes-october-24-2025) | 2025-10-24T13:46 | 2025-10-24T13:57 | 9.7 | before | post-only |  |
| 2025-10-26 | NYK@MIA | RS | [Knicks vs Heat Game Notes: October 26, 2025](https://www.nba.com/knicks/news/knicks-vs-heat-game-notes-october-26-2025) | 2025-10-25T22:43 | 2025-10-25T22:44 | 23.3 | before | post-only |  |
| 2025-10-28 | NYK@MIL | RS | [Knicks vs Bucks Game Notes: October 26, 2025](https://www.nba.com/knicks/news/knicks-vs-bucks-game-notes-october-26-2025) | 2025-10-27T19:36 | 2025-10-28T17:47 | 28.4 | before | - | title date 2025-10-26 != game; games-taxonomy wrong |
| 2025-10-31 | NYK@CHI | RS | [Knicks vs Bulls Game Notes: October 31, 2025](https://www.nba.com/knicks/news/knicks-vs-bulls-game-notes-october-31-2025) | 2025-10-30T18:21 | 2025-11-01T01:41 | 29.6 | after (+2h) | - |  |
| 2025-11-02 | CHI@NYK | RS | [Knicks vs Bulls Game Notes: November 2, 2025](https://www.nba.com/knicks/news/knicks-vs-bulls-game-notes-november-2-2025) | 2025-11-02T12:38 | 2025-11-03T00:12 | 11.4 | after (+0h) | - |  |
| 2025-11-03 | WAS@NYK | RS | [Knicks vs Wizards Game Notes: November 3, 2025](https://www.nba.com/knicks/news/knicks-vs-wizards-game-notes-november-3-2025) | 2025-11-03T17:10 | 2025-11-03T17:23 | 7.3 | before | - |  |
| 2025-11-05 | MIN@NYK | RS | [Knicks vs Timberwolves Game Notes: November 5, 2025](https://www.nba.com/knicks/news/knicks-vs-timberwolves-game-notes-november-5-2025) | 2025-11-04T22:05 | 2025-11-09T14:15 | 26.4 | after (+86h) | - |  |
| 2025-11-09 | BKN@NYK | RS | [Knicks vs Nets Game Notes: November 9, 2025](https://www.nba.com/knicks/news/knicks-vs-nets-game-notes-november-9-2025) | 2025-11-09T14:17 | 2025-11-09T14:25 | 8.7 | before | - |  |
| 2025-11-11 | MEM@NYK | RS | [Knicks vs Grizzlies Game Notes: November 11, 2025](https://www.nba.com/knicks/news/knicks-vs-grizzlies-game-notes-november-11-2025) | 2025-11-10T21:15 | 2025-11-10T21:30 | 27.2 | before | - |  |
| 2025-11-12 | ORL@NYK | RS | [Knicks vs Magic Game Notes: November 12, 2025](https://www.nba.com/knicks/news/knicks-vs-magic-game-notes-november-12-2025) | 2025-11-12T17:17 | 2025-11-18T23:08 | 6.7 | after (+143h) | - |  |
| 2025-11-14 | MIA@NYK | RS | [Knicks vs Heat Game Notes: November 14, 2025](https://www.nba.com/knicks/news/knicks-vs-heat-game-notes-november-14-2025) | 2025-11-14T14:31 | 2025-11-18T23:09 | 9.5 | after (+95h) | - |  |
| 2025-11-17 | NYK@MIA | RS | [Knicks vs Heat Game Notes: November 17, 2025](https://www.nba.com/knicks/news/knicks-vs-heat-game-notes-november-17-2025) | 2025-11-17T14:41 | 2025-11-18T23:09 | 9.8 | after (+23h) | - |  |
| 2025-11-19 | NYK@DAL | RS | [Knicks vs Mavericks Game Notes: November 19, 2025](https://www.nba.com/knicks/news/knicks-vs-mavericks-game-notes-november-19-2025) | 2025-11-18T23:08 | 2026-01-20T16:23 | 27.4 | after (+1478h) | - |  |
| 2025-11-22 | NYK@ORL | RS | [Knicks vs Magic Game Notes: November 22, 2025](https://www.nba.com/knicks/news/knicks-vs-magic-game-notes-november-22-2025) | 2025-11-22T16:50 | 2026-01-20T17:18 | 5.2 | after (+1411h) | 20251122173409 |  |
| 2025-11-24 | NYK@BKN | RS | [Knicks vs Nets Game Notes: November 24, 2025](https://www.nba.com/knicks/news/knicks-vs-nets-game-notes-november-24-2025) | 2025-11-24T15:08 | 2026-01-20T17:20 | 9.4 | after (+1361h) | 20251124161843 |  |
| 2025-11-26 | NYK@CHA | RS | [Knicks vs Hornets Game Notes: November 26, 2025](https://www.nba.com/knicks/news/knicks-vs-hornets-game-notes-november-26-2025) | 2025-11-26T15:43 | 2026-01-20T17:26 | 8.3 | after (+1313h) | - | games-taxonomy wrong |
| 2025-11-28 | MIL@NYK | RS | [Knicks vs Bucks Game Notes: November 28, 2025](https://www.nba.com/knicks/news/knicks-vs-bucks-game-notes-november-28-2025) | 2025-11-28T13:07 | 2026-01-20T17:39 | 11.4 | after (+1265h) | - | games-taxonomy wrong |
| 2025-11-30 | TOR@NYK | RS | [Knicks vs Raptors Game Notes: November 30, 2025](https://www.nba.com/knicks/news/knicks-vs-raptors-game-notes-november-30-2025) | 2025-11-30T15:23 | 2026-01-23T16:55 | 7.6 | after (+1290h) | - | games-taxonomy wrong |
| 2025-12-02 | NYK@BOS | RS | [Knicks vs Celtics Game Notes: December 02, 2025](https://www.nba.com/knicks/news/knicks-vs-celtics-game-notes-december-02-2025) | 2025-12-02T15:25 | 2026-01-23T16:59 | 9.6 | after (+1240h) | - |  |
| 2025-12-03 | CHA@NYK | RS | [Knicks vs Hornets Game Notes: December 03, 2025](https://www.nba.com/knicks/news/knicks-vs-hornets-game-notes-december-03-2025) | 2025-12-03T17:08 | 2026-01-23T17:02 | 7.3 | after (+1217h) | - | games-taxonomy wrong |
| 2025-12-05 | UTA@NYK | RS | [Knicks vs Jazz Game Notes: December 05, 2025](https://www.nba.com/knicks/news/knicks-vs-jazz-game-notes-december-05-2025) | 2025-12-05T14:30 | 2026-01-23T17:07 | 10.0 | after (+1169h) | - | games-taxonomy wrong |
| 2025-12-07 | ORL@NYK | RS | [Knicks vs Magic Game Notes: December 07, 2025](https://www.nba.com/knicks/news/knicks-vs-magic-game-notes-december-07-2025) | 2025-12-07T13:57 | 2026-01-23T17:08 | 3.0 | after (+1128h) | post-only | games-taxonomy wrong |
| 2025-12-09 | NYK@TOR | RS | [Knicks vs Raptors Game Notes: December 09, 2025](https://www.nba.com/knicks/news/knicks-vs-raptors-game-notes-december-09-2025) | 2025-12-09T14:58 | 2026-01-23T17:31 | 10.5 | after (+1072h) | post-only |  |
| 2025-12-13 | NYK@ORL | RS | [Knicks vs Magic Game Notes: December 13, 2025](https://www.nba.com/knicks/news/knicks-vs-magic-game-notes-december-13-2025) | 2025-12-13T14:43 | 2026-01-23T17:34 | 7.8 | after (+979h) | post-only |  |
| 2025-12-16 | SAS@NYK | Cup final | [Knicks vs Spurs Game Notes: December 16, 2025](https://www.nba.com/knicks/news/knicks-vs-spurs-game-notes-december-16-2025) | 2025-12-16T12:27 | 2026-01-23T19:03 | 13.1 | after (+906h) | 20251216130345 | games-taxonomy wrong |
| 2025-12-18 | NYK@IND | RS | [Knicks vs Pacers Game Notes: December 18, 2025](https://www.nba.com/knicks/news/knicks-vs-pacers-game-notes-december-18-2025) | 2025-12-18T15:48 | 2026-01-23T19:12 | 8.2 | after (+859h) | post-only |  |
| 2025-12-19 | PHI@NYK | RS | [Knicks vs 76ers Game Notes: December 19, 2025](https://www.nba.com/knicks/news/knicks-vs-76ers-game-notes-december-19-2025) | 2025-12-19T17:53 | 2026-01-23T19:14 | 6.1 | after (+835h) | post-only | games-taxonomy wrong |
| 2025-12-21 | MIA@NYK | RS | [Knicks vs Miami Heat Game Notes: December 21, 2025](https://www.nba.com/knicks/news/knicks-vs-miami-heat-game-notes-december-21-2025) | 2025-12-21T13:56 | 2026-01-23T19:15 | 9.1 | after (+788h) | post-only |  |
| 2025-12-23 | NYK@MIN | RS | [Knicks vs Timberwolves Game Notes: December 23, 2025](https://www.nba.com/knicks/news/knicks-vs-timberwolves-game-notes-december-23-2025) | 2025-12-23T14:52 | 2026-01-23T19:23 | 10.1 | after (+738h) | 20251223233353 |  |
| 2025-12-25 | CLE@NYK | RS | [Knicks vs Cavaliers Game Notes: December 25, 2025](https://www.nba.com/knicks/news/knicks-vs-cavaliers-game-notes-december-25-2025) | 2025-12-25T12:18 | 2026-01-23T19:29 | 4.7 | after (+698h) | post-only | games-taxonomy wrong |
| 2025-12-27 | NYK@ATL | RS | [Knicks vs Hawks Game Notes: December 27, 2025](https://www.nba.com/knicks/news/knicks-vs-hawks-game-notes-december-27-2025) | 2025-12-27T13:31 | 2026-01-23T19:33 | 11.5 | after (+643h) | 20251227152002 |  |
| 2025-12-29 | NYK@NOP | RS | [Knicks vs Pelicans Game Notes: December 29, 2025](https://www.nba.com/knicks/news/knicks-vs-pelicans-game-notes-december-29-2025) | 2025-12-29T14:39 | 2026-01-23T19:37 | 10.3 | after (+595h) | post-only |  |
| 2025-12-31 | NYK@SAS | RS | [Knicks vs Spurs Game Notes: December 31, 2025](https://www.nba.com/knicks/news/knicks-vs-spurs-game-notes-december-31-2025) | 2025-12-31T15:31 | 2026-01-23T19:39 | 8.5 | after (+548h) | 20251231160522 |  |
| 2026-01-02 | ATL@NYK | RS | [Knicks vs Hawks Game Notes: January 2, 2026](https://www.nba.com/knicks/news/knicks-vs-hawks-game-notes-january-2-2026) | 2026-01-02T14:03 | 2026-01-23T19:42 | 10.4 | after (+499h) | post-only |  |
| 2026-01-03 | PHI@NYK | RS | [Knicks vs 76ers Game Notes: January 3, 2026](https://www.nba.com/knicks/news/knicks-vs-76ers-game-notes-january-3-2026) | 2026-01-03T17:54 | 2026-01-23T19:48 | 6.6 | after (+475h) | post-only | games-taxonomy wrong |
| 2026-01-05 | NYK@DET | RS | [Knicks vs Pistons Game Notes: January 5, 2026](https://www.nba.com/knicks/news/knicks-vs-pistons-game-notes-january-5-2026) | 2026-01-05T14:46 | 2026-01-05T14:48 | 9.2 | before | post-only |  |
| 2026-01-07 | LAC@NYK | RS | [Knicks vs Clippers Game Notes: January 7, 2026](https://www.nba.com/knicks/news/knicks-vs-clippers-game-notes-january-7-2026) | 2026-01-06T17:45 | 2026-01-23T19:49 | 30.7 | after (+379h) | 20260107070608 |  |
| 2026-01-09 | NYK@PHX | RS | [Knicks vs Suns Game Notes: January 9, 2026](https://www.nba.com/knicks/news/knicks-vs-suns-game-notes-january-9-2026) | 2026-01-09T17:46 | 2026-01-23T19:53 | 8.2 | after (+330h) | - |  |
| 2026-01-11 | NYK@POR | RS | [Knicks vs Trailblazers Game Notes: January 11, 2026](https://www.nba.com/knicks/news/knicks-vs-trailblazers-game-notes-january-11-2026) | 2026-01-11T15:44 | 2026-01-23T19:59 | 7.3 | after (+285h) | - |  |
| 2026-01-14 | NYK@SAC | RS | [Knicks vs Kings Game Notes: January 14, 2026](https://www.nba.com/knicks/news/knicks-vs-kings-game-notes-january-14-2026) | 2026-01-13T20:37 | 2026-01-23T20:03 | 30.4 | after (+209h) | - | games-taxonomy wrong |
| 2026-01-15 | NYK@GSW | RS | [Knicks vs Warriors Game Notes: January 15, 2026](https://www.nba.com/knicks/news/knicks-vs-warriors-game-notes-january-15-2026) | 2026-01-15T17:17 | 2026-01-23T20:13 | 9.7 | after (+185h) | - | games-taxonomy wrong |
| 2026-01-17 | PHX@NYK | RS | [Knicks vs Suns Game Notes: January 17, 2026](https://www.nba.com/knicks/news/knicks-vs-suns-game-notes-january-16-2026) | 2026-01-16T17:37 | 2026-01-23T20:15 | 30.9 | after (+140h) | - |  |
| 2026-01-19 | DAL@NYK | RS | [Knicks vs Mavericks Game Notes: January 19, 2026](https://www.nba.com/knicks/news/knicks-vs-mavericks-game-notes-january-19-2026) | 2026-01-19T14:59 | 2026-01-23T20:16 | 7.0 | after (+94h) | - |  |
| 2026-01-21 | BKN@NYK | RS | [Knicks vs Nets Game Notes: January 21, 2026](https://www.nba.com/knicks/news/knicks-vs-nets-game-notes-january-21-2026) | 2026-01-21T14:26 | 2026-01-21T14:26 | 10.1 | before | - |  |
| 2026-01-24 | NYK@PHI | RS | [Knicks vs 76ers Game Notes: January 24, 2026](https://www.nba.com/knicks/news/knicks-vs-76ers-game-notes-january-24-2026) | 2026-01-23T21:24 | 2026-01-23T21:54 | 22.6 | before | - |  |
| 2026-01-27 | SAC@NYK | RS | [Knicks vs Kings Game Notes: January 27, 2026](https://www.nba.com/knicks/news/knicks-vs-kings-game-notes-january-27-2026) | 2026-01-26T20:38 | 2026-01-26T20:38 | 27.9 | before | - | games-taxonomy wrong |
| 2026-01-28 | NYK@TOR | RS | [Knicks vs Raptors Game Notes: January 28, 2026](https://www.nba.com/knicks/news/knicks-vs-raptors-game-notes-january-28-2026) | 2026-01-28T18:27 | 2026-01-28T18:27 | 6.0 | before | 20260128190245 | games-taxonomy wrong |
| 2026-01-30 | POR@NYK | RS | [Knicks vs Trail Blazers Game Notes: January 30, 2026](https://www.nba.com/knicks/news/knicks-vs-trail-blazers-game-notes-january-28-2026) | 2026-01-30T14:42 | 2026-01-30T17:38 | 9.8 | before | - |  |
| 2026-02-01 | LAL@NYK | RS | [Knicks vs Lakers Game Notes: February 1, 2026](https://www.nba.com/knicks/news/knicks-vs-lakers-game-notes-february-1-2026) | 2026-02-01T16:15 | 2026-02-01T16:26 | 7.7 | before | 20260201163226 | games-taxonomy wrong |
| 2026-02-03 | NYK@WAS | RS | [Knicks vs Wizards Game Notes: February 3, 2026](https://www.nba.com/knicks/news/knicks-vs-wizards-game-notes-february-3-2026) | 2026-02-02T20:41 | 2026-02-02T20:41 | 27.3 | before | - |  |
| 2026-02-04 | DEN@NYK | RS | [Knicks vs Nuggets Game Notes: February 4, 2026](https://www.nba.com/knicks/news/knicks-vs-nuggets-game-notes-february-4-2026) | 2026-02-04T17:04 | 2026-02-04T17:04 | 6.9 | before | - |  |
| 2026-02-06 | NYK@DET | RS | [Knicks vs Pistons Game Notes: February 5, 2026](https://www.nba.com/knicks/news/knicks-vs-pistons-game-notes-february-5-2026) | 2026-02-05T21:12 | 2026-02-05T21:12 | 27.3 | before | - | title date 2026-02-05 != game |
| 2026-02-08 | NYK@BOS | RS | [Knicks vs Celtics Game Notes: February 8, 2026](https://www.nba.com/knicks/news/knicks-vs-celtics-game-notes-february-8-2026) | 2026-02-08T13:53 | 2026-02-08T13:53 | 3.6 | before | - |  |
| 2026-02-10 | IND@NYK | RS | [Knicks vs Pacers Game Notes: February 10, 2026](https://www.nba.com/knicks/news/knicks-vs-pacers-game-notes-february-9-2026) | 2026-02-09T21:36 | 2026-02-09T21:36 | 26.9 | before | - |  |
| 2026-02-11 | NYK@PHI | RS | [Knicks vs 76ers Game Notes: February 11, 2026](https://www.nba.com/knicks/news/knicks-vs-76ers-game-notes-february-11-2026) | 2026-02-11T18:05 | 2026-02-11T18:05 | 6.4 | before | 20260211190059 |  |
| 2026-02-19 | DET@NYK | RS | [Knicks vs Pistons Game Notes: February 19, 2026](https://www.nba.com/knicks/news/knicks-vs-pistons-game-notes-february-19-2026) | 2026-02-19T16:12 | 2026-02-19T16:46 | 8.3 | before | 20260219165253 |  |
| 2026-02-21 | HOU@NYK | RS | [Knicks vs Rockets Game Notes: February 21, 2026](https://www.nba.com/knicks/news/knicks-vs-rockets-game-notes-february-21-2026) | 2026-02-21T13:53 | 2026-02-21T13:53 | 11.6 | before | 20260221143203 |  |
| 2026-02-22 | NYK@CHI | RS | [Knicks vs Bulls Game Notes: February 22, 2026](https://www.nba.com/knicks/news/knicks-vs-bulls-game-notes-february-22-2026) | 2026-02-22T17:07 | 2026-02-22T17:07 | 7.9 | before | 20260222181412 |  |
| 2026-02-24 | NYK@CLE | RS | [Knicks vs Cavaliers Game Notes: February 24, 2026](https://www.nba.com/knicks/news/knicks-vs-cavaliers-game-notes-february-24-2026) | 2026-02-23T20:46 | 2026-02-23T20:48 | 27.7 | before | post-only |  |
| 2026-02-27 | NYK@MIL | RS | [Knicks vs Bucks Game Notes: February 27, 2026](https://www.nba.com/knicks/news/knicks-vs-bucks-game-notes-february-27-2026) | 2026-02-26T19:24 | 2026-02-26T19:24 | 29.6 | before | 20260226200324 |  |
| 2026-03-01 | SAS@NYK | RS | [Knicks vs Spurs Game Notes: March 1, 2026](https://www.nba.com/knicks/news/knicks-vs-spurs-game-notes-march-1-2026) | 2026-03-01T14:32 | 2026-03-01T14:32 | 3.5 | before | 20260301153208 |  |
| 2026-03-03 | NYK@TOR | RS | [Knicks vs Raptors Game Notes: March 3, 2026](https://www.nba.com/knicks/news/knicks-vs-raptors-game-notes-march-3-2026) | 2026-03-03T12:36 | 2026-03-03T12:36 | 11.9 | before | - |  |
| 2026-03-04 | OKC@NYK | RS | [Knicks vs Thunder Game Notes: March 4, 2026](https://www.nba.com/knicks/news/knicks-vs-thunder-game-notes-march-3-2026) | 2026-03-04T17:42 | 2026-03-04T19:01 | 6.3 | before | - |  |
| 2026-03-06 | NYK@DEN | RS | [Knicks vs Nuggets Game Notes: March 6, 2026](https://www.nba.com/knicks/news/knicks-vs-nuggets-game-notes-march-6-2026) | 2026-03-06T14:18 | 2026-03-06T14:18 | 11.7 | before | post-only |  |
| 2026-03-08 | NYK@LAL | RS | [Knicks vs Lakers Game Notes: March 8, 2026](https://www.nba.com/knicks/news/knicks-vs-lakers-game-notes-march-8-2026) | 2026-03-08T14:07 | 2026-03-08T21:08 | 5.4 | after (+2h) | - |  |
| 2026-03-09 | NYK@LAC | RS | [Knicks vs Clippers Game Notes: March 9, 2026](https://www.nba.com/knicks/news/knicks-vs-clippers-game-notes-march-9-2026) | 2026-03-09T18:25 | 2026-03-09T18:25 | 7.6 | before | - |  |
| 2026-03-11 | NYK@UTA | RS | [Knicks vs Jazz Game Notes: March 11, 2026](https://www.nba.com/knicks/news/knicks-vs-jazz-game-notes-march-11-2026) | 2026-03-11T14:07 | 2026-03-11T14:07 | 10.9 | before | - |  |
| 2026-03-13 | NYK@IND | RS | [Knicks vs Pacers Game Notes: March 13, 2026](https://www.nba.com/knicks/news/knicks-vs-pacers-game-notes-march-13-2026) | 2026-03-13T15:41 | 2026-03-13T15:41 | 7.8 | before | - |  |
| 2026-03-15 | GSW@NYK | RS | [Knicks vs Warriors Game Notes: March 15, 2026](https://www.nba.com/knicks/news/knicks-vs-warriors-game-notes-march-15-2026) | 2026-03-15T14:08 | 2026-03-15T14:08 | 9.8 | before | - |  |
| 2026-03-17 | IND@NYK | RS | [Knicks vs Pacers Game Notes: March 17, 2026](https://www.nba.com/knicks/news/knicks-vs-pacers-game-notes-march-17-2026) | 2026-03-17T14:25 | 2026-03-17T14:25 | 9.1 | before | - |  |
| 2026-03-20 | NYK@BKN | RS | [Knicks vs Nets Game Notes: March 20, 2026](https://www.nba.com/knicks/news/knicks-vs-nets-game-notes-march-20-2026) | 2026-03-20T14:11 | 2026-03-20T14:11 | 9.3 | before | 20260320203239 |  |
| 2026-03-22 | WAS@NYK | RS | [Knicks vs Wizards Game Notes: March 22, 2026](https://www.nba.com/knicks/news/knicks-vs-wizards-game-notes-march-22-2026) | 2026-03-22T16:00 | 2026-03-22T16:00 | 7.5 | before | 20260322163205 |  |
| 2026-03-24 | NOP@NYK | RS | [Knicks vs Pelicans Game Notes: March 24, 2026](https://www.nba.com/knicks/news/knicks-vs-pelicans-game-notes-march-24-2026) | 2026-03-24T15:46 | 2026-03-24T15:46 | 7.7 | before | - |  |
| 2026-03-26 | NYK@CHA | RS | [Knicks vs Hornets Game Notes: March 26, 2026](https://www.nba.com/knicks/news/knicks-vs-hornets-game-notes-march-26-2026) | 2026-03-26T14:03 | 2026-05-12T17:27 | 8.9 | after (+1122h) | 20260326150212 |  |
| 2026-03-29 | NYK@OKC | RS | [Knicks vs Thunder Game Notes: March 29, 2026](https://www.nba.com/knicks/news/knicks-vs-thunder-game-notes-march-29-2026) | 2026-03-29T15:29 | 2026-05-12T17:27 | 8.0 | after (+1050h) | post-only | games-taxonomy wrong |
| 2026-03-31 | NYK@HOU | RS | [Knicks vs Rockets Game Notes: March 31, 2026](https://www.nba.com/knicks/news/knicks-vs-rockets-game-notes-march-31-2026) | 2026-03-31T13:39 | 2026-05-12T17:26 | 10.3 | after (+1001h) | 20260331143339 |  |
| 2026-04-01 | NYK@MEM | RS | [Knicks vs Grizzlies Game Notes: April 1, 2026](https://www.nba.com/knicks/news/knicks-vs-grizzlies-game-notes-april-1-2026) | 2026-04-01T17:18 | 2026-05-12T17:26 | 6.7 | after (+977h) | post-only |  |
| 2026-04-03 | CHI@NYK | RS | [Knicks vs Bulls Game Notes: April 3, 2026](https://www.nba.com/knicks/news/knicks-vs-bulls-game-notes-april-3-2026) | 2026-04-03T13:43 | 2026-05-12T17:25 | 9.8 | after (+930h) | 20260403150219 | games-taxonomy wrong |
| 2026-04-06 | NYK@ATL | RS | [Knicks vs Hawks Game Notes: April 6, 2026](https://www.nba.com/knicks/news/knicks-vs-hawks-game-notes-april-6-2026) | 2026-04-06T13:25 | 2026-05-12T17:25 | 9.6 | after (+858h) | 20260406140151 |  |
| 2026-04-09 | BOS@NYK | RS | [Knicks vs Celtics Game Notes: April 9, 2026](https://www.nba.com/knicks/news/knicks-vs-hawks-game-notes-april-9-2026) | 2026-04-09T13:54 | 2026-05-12T17:24 | 9.6 | after (+786h) | - | games-taxonomy wrong |
| 2026-04-10 | TOR@NYK | RS | [Knicks vs Raptors Game Notes: April 10, 2026](https://www.nba.com/knicks/news/knicks-vs-raptors-game-notes-april-10-2026) | 2026-04-10T15:51 | 2026-05-12T17:24 | 7.6 | after (+762h) | 20260410160511 |  |
| 2026-04-12 | CHA@NYK | RS | [Knicks vs Hornets Game Notes: April 12, 2026](https://www.nba.com/knicks/news/knicks-vs-hornets-game-notes-april-12-2026) | 2026-04-12T15:51 | 2026-05-12T17:23 | 6.1 | after (+715h) | post-only | games-taxonomy wrong |
| 2026-04-18 | ATL@NYK | PO | [Knicks vs Hawks Game Notes / April 18, 2026: Round 1 Game 1](https://www.nba.com/knicks/news/knicks-vs-hawks-game-notes-april-18-2026-round-1-game-1) | 2026-04-18T14:01 | 2026-05-12T17:23 | 8.0 | after (+571h) | 20260418160334 |  |
| 2026-04-20 | ATL@NYK | PO | [Knicks vs Hawks Game Notes / April 20, 2026: Round 1 Game 2](https://www.nba.com/knicks/news/knicks-vs-hawks-game-notes-april-20-2026-round-1-game-2) | 2026-04-20T13:25 | 2026-05-12T17:22 | 10.6 | after (+521h) | 20260420143248 |  |
| 2026-04-23 | NYK@ATL | PO | [Knicks vs Hawks Game Notes / April 23, 2026: Round 1 Game 3](https://www.nba.com/knicks/news/knicks-vs-hawks-game-notes-april-23-2026-round-1-game-3) | 2026-04-22T21:22 | 2026-05-12T17:22 | 25.6 | after (+450h) | - |  |
| 2026-04-25 | NYK@ATL | PO | [Knicks vs Hawks Game Notes / April 25, 2026: Round 1 Game 4](https://www.nba.com/knicks/news/knicks-vs-hawks-game-notes-april-25-2026-round-1-game-4) | 2026-04-24T20:57 | 2026-05-12T17:21 | 25.0 | after (+403h) | - |  |
| 2026-04-28 | ATL@NYK | PO | [Knicks vs Hawks Game Notes / April 28, 2026: Round 1 Game 5](https://www.nba.com/knicks/news/knicks-vs-hawks-game-notes-april-28-2026-round-1-game-5) | 2026-04-28T13:58 | 2026-05-12T17:20 | 10.0 | after (+329h) | 20260428165021 |  |
| 2026-04-30 | NYK@ATL | PO | [Knicks vs Hawks Game Notes / April 30, 2026: Round 1 Game 6](https://www.nba.com/knicks/news/knicks-vs-hawks-game-notes-april-30-2026-round-1-game-6) | 2026-04-29T18:35 | 2026-05-12T17:19 | 28.4 | after (+282h) | 20260430105319 |  |
| 2026-05-04 | PHI@NYK | PO | [Knicks vs 76ers Game Notes / May 04, 2026: Round 2 Game 1](https://www.nba.com/knicks/news/knicks-vs-76ers-game-notes-may-04-2026-round-2-game-1) | 2026-05-04T13:22 | 2026-05-12T17:17 | 10.6 | after (+185h) | 20260504150247 |  |
| 2026-05-06 | PHI@NYK | PO | [Knicks vs 76ers Game Notes / May 06, 2026: Round 2 Game 2](https://www.nba.com/knicks/news/knicks-vs-76ers-game-notes-may-06-2026-round-2-game-2) | 2026-05-05T16:49 | 2026-05-12T17:17 | 30.2 | after (+138h) | post-only |  |
| 2026-05-08 | NYK@PHI | PO | [Knicks vs 76ers Game Notes / May 08, 2026: Round 2 Game 3](https://www.nba.com/knicks/news/knicks-vs-76ers-game-notes-may-08-2026-round-2-game-3) | 2026-05-08T14:27 | 2026-05-12T17:15 | 8.5 | after (+90h) | post-only | games-taxonomy wrong |
| 2026-05-10 | NYK@PHI | PO | [Knicks vs 76ers Game Notes / May 10, 2026: Round 2 Game 4](https://www.nba.com/knicks/news/knicks-vs-76ers-game-notes-may-10-2026-round-2-game-4) | 2026-05-10T13:03 | 2026-05-18T16:25 | 6.4 | after (+189h) | 20260510140555 | games-taxonomy wrong |
| 2026-05-19 | CLE@NYK | PO | [Knicks vs Cavaliers Game Notes / May 19, 2026: Eastern Conference Finals Game 1](https://www.nba.com/knicks/news/knicks-vs-cavaliers-game-notes-may-19-2026-eastern-conference-finals-game-1) | 2026-05-18T16:34 | 2026-05-18T16:34 | 31.4 | before | 20260518173427 | games-taxonomy wrong |
| 2026-05-21 | CLE@NYK | PO | [Knicks vs Cavaliers Game Notes / May 21, 2026: Eastern Conference Finals Game 2](https://www.nba.com/knicks/news/knicks-vs-cavaliers-game-notes-may-21-2026-eastern-conference-finals-game-2) | 2026-05-20T18:43 | 2026-05-20T18:43 | 29.3 | before | 20260520193342 | games-taxonomy wrong |
| 2026-05-23 | NYK@CLE | PO | [Knicks vs Cavaliers Game Notes / May 23, 2026: Eastern Conference Finals Game 3](https://www.nba.com/knicks/news/knicks-vs-cavaliers-game-notes-may-23-2026-eastern-conference-finals-game-3) | 2026-05-23T14:41 | 2026-05-23T23:33 | 9.3 | after (+-0h) | post-only | games-taxonomy wrong |
| 2026-05-25 | NYK@CLE | PO | [Knicks vs Cavaliers Game Notes / May 25, 2026: Eastern Conference Finals Game 4](https://www.nba.com/knicks/news/knicks-vs-cavaliers-game-notes-may-25-2026-eastern-conference-finals-game-4) | 2026-05-25T13:26 | 2026-05-25T13:26 | 10.6 | before | post-only | games-taxonomy wrong |
| 2026-06-03 | NYK@SAS | PO | [Knicks vs Spurs Game Notes / June 3, 2026: Finals - Game 1](https://www.nba.com/knicks/news/knicks-vs-spurs-game-notes-june-3-2026-finals-game-1) | 2026-06-05T19:06 | 2026-06-05T19:11 | -42.6 | after (+43h) | post-only | PUBLISHED AFTER CUTOFF |
| 2026-06-05 | NYK@SAS | PO | [Knicks vs Spurs Game Notes / June 5, 2026: NBA Finals - Game 2](https://www.nba.com/knicks/news/knicks-vs-spurs-game-notes-june-5-2026-nba-finals-game-2) | 2026-06-05T19:07 | 2026-06-05T19:11 | 5.4 | before | post-only |  |
| 2026-06-08 | SAS@NYK | PO | [Knicks vs Spurs Game Notes / June 8, 2026: NBA Finals - Game 3](https://www.nba.com/knicks/news/knicks-vs-spurs-game-notes-june-8-2026-nba-finals-game-3) | 2026-06-08T14:28 | 2026-06-10T16:16 | 10.0 | after (+40h) | post-only |  |
| 2026-06-10 | SAS@NYK | PO | [Knicks vs Spurs Game Notes / June 10, 2026: NBA Finals - Game 4](https://www.nba.com/knicks/news/knicks-vs-spurs-game-notes-june-10-2026-nba-finals-game-4) | 2026-06-10T16:19 | 2026-06-10T16:19 | 8.2 | before | post-only |  |
| 2026-06-13 | NYK@SAS | PO | [Knicks vs Spurs Game Notes / June 13, 2026: NBA Finals - Game 5](https://www.nba.com/knicks/news/knicks-vs-spurs-game-notes-june-13-2026-nba-finals-game-5) | 2026-06-13T15:09 | 2026-06-13T15:10 | 9.3 | before | post-only |  |

#### 2026-27

| Game (ET) | Matchup | Type | Title | Published (UTC) | Modified (UTC) | Pub lead (h) | Mod vs cutoff | Wayback pre-cutoff | Flags |
|---|---|---|---|---|---|---:|---|---|---|
| 2026-10-20 | PHI@NYK | RS | [Knicks vs 76ers Game Notes / October 20, 2026](https://www.nba.com/knicks/news/knicks-vs-76ers-game-notes-october-20-2026) | 2026-10-23T20:00 | 2026-10-31T02:06 | -69.0 | after (+243h) | - | PUBLISHED AFTER CUTOFF |
| 2026-10-23 | NYK@BOS | RS | [Knicks vs Celtics Game Notes / October 23, 2026](https://www.nba.com/knicks/news/knicks-vs-celtics-game-notes-october-23-2026) | 2026-10-23T20:07 | 2026-10-31T02:05 | 2.9 | after (+171h) | - |  |
| 2026-10-25 | ORL@NYK | RS | [Knicks vs Magic Game Notes / October 25, 2026](https://www.nba.com/knicks/news/knicks-vs-magic-game-notes-october-25-2026) | 2026-10-26T01:00 | 2026-10-31T02:05 | -2.0 | after (+123h) | - | PUBLISHED AFTER CUTOFF |
| 2026-10-27 | DET@NYK | RS | [Knicks vs Pistons Game Notes / October 27, 2026](https://www.nba.com/knicks/news/knicks-vs-pistons-game-notes-october-27-2026) | 2026-10-27T15:44 | 2026-10-31T02:04 | 8.3 | after (+74h) | - |  |
| 2026-10-28 | NYK@CHI | RS | [Knicks vs Bulls Game Notes / October 28, 2026](https://www.nba.com/knicks/news/knicks-vs-bulls-game-notes-october-28-2026) | 2026-10-28T17:57 | 2026-10-31T02:04 | 6.0 | after (+50h) | - | games-taxonomy wrong |
| 2026-10-30 | NYK@PHI | RS | [Knicks vs 76ers Game Notes / October 30, 2026](https://www.nba.com/knicks/news/knicks-vs-76ers-game-notes-october-30-2026) | 2026-10-29T22:32 | 2026-10-31T02:03 | 24.9 | after (+3h) | - |  |

## Appendix C. Raw artifacts (scratchpad, not in repo)

* `raw/enum/knicks_category_game-notes.json`: metadata for all 170 posts.
* `raw/knicks_final.json`: posts mapped to games, with Wayback captures.
* `raw/blz/meta.jsonl`: Blazers `tag/preview` pages 12–40.
* `raw/sitemaps/*`: 30 team news sitemaps and indexes.
* `raw/probe_summary.jsonl`: the 60 category/tag probes.
* `raw/pdf_probe.json`: the 58 dated-PDF guesses.
* `raw/rolling_pdfs.txt`: HEAD results for the 30 rolling PDFs.
* `raw/cdx_knicks_news.json`, `raw/cdx_blazers_gn.json`, `raw/cdx_cle_page.json`: Wayback indexes.
* `raw/wb/*.html`: the archived and current pages behind section 4.
* `scripts/*.py`: probe and analysis scripts.
