# Bounded 2023–26 note expansion

This collection uses `config/bam_notes_expansion.json` and the new
`scripts/bam_collect_notes.py`. It never changes the prior collectors, settings,
raw artifacts, 256-game snapshot, or completed pilot results. Raw bytes and every
HTTP attempt go under `data/raw/bam_notes_expansion/v1/`; processed candidates,
content audits and the additional approved PDF manifest go under
`data/processed/bam_notes_expansion/v1/`.

Both Claude discovery reports are durable at
`data/provenance/claude_archive_research/20261101T2212Z/`. They are research leads,
not verified inputs. In particular, the web-archive report's roughly 668-game
union is an index-matched estimate relative to 404 source-verified games. The
current model-approved cohort is only 256 games. These counts are different
stages of validation and must not be conflated.

1. Recheck already downloaded, source-verified versions omitted from the approved
   manifest. Reuse strict date, season, matchup, prior-record/game-number and
   previously reviewed publisher-template checks. Try plain and layout extraction
   independently. A formatting repair can pass only when every check passes;
   unreviewed templates, wrong records, wrong venues and unreadable pages remain
   quarantined. Prior explicit source exclusions remain in force.
2. Retry at most 24 current-season PDF candidates from the old failed replays,
   prioritizing new distinct games and reviewed templates. Archive requests use
   one worker and a 12-second gap. Every successful body needs an exact replay
   URL, matching Memento timestamp, pre-cutoff capture, PDF signature and SHA-256.
   Failed 429/503 or HTTP-200 block pages stop further archive traffic in the pass.
3. Query one master Wayback prefix index including query-string variants and
   revisits. This addresses the failed Philadelphia index and Lakers query
   variants without mutating the old collector. Drop cross-team placeholder
   digests and stale versions. Candidate windows end at tipoff minus 60 minutes,
   begin no more than 36 hours earlier and after the previous scheduled tip plus
   2.5 hours. These windows are filters, not approval: header and content checks
   remain mandatory. Preserve latest pre-cutoff versions and all provenance.
4. Parse the exact Knicks post list from the durable report and join by game date
   and both teams against the verified regular-season history. Never trust
   `taxonomy.games`, slug dates or archive capture counts as game identity.
   Retrieve at most four actual pre-cutoff HTML replays in this first pass.
   Extract only article-owned blocks from the archived page object, retain CMS
   published/modified evidence and reject metadata later than cutoff. These HTML
   candidates stay separate until an explicit HTML snapshot adapter and content
   review are complete; they are never passed to a PDF reader or silently promoted
   through the PDF approval path.

Knicks current pages modified after cutoff remain rejected, including bulk
housekeeping edits; an actual archived pre-cutoff version is required. CMS-clean
Knicks/Portland pages are a separate, weaker publisher-clock evidence tier and
are not included in this strict initial archived-source pass. Portland's roughly
123 CMS-clean mapped articles and nine pre-cutoff Wayback captures warrant the
next HTML pass, after source-kind integration. Broad preview scraping is lower
priority because it introduces additional content/timing validation work.

The highest-yield later PDF queues are the report's Philadelphia and Lakers
candidates and the old throttled replay failures. Philadelphia front pages may
need visual OCR/transcription; archived bytes alone do not establish readable
content or full-packet quality. Arquivo.pt can add a small number of 2025–26
candidates after a separate archive-proof adapter. Common Crawl adds no reported
2023–26 documents; its historical 2016–19 records and the roughly 1,980 earlier
Wayback team-game candidates remain a separate future project requiring matching
numerical history. No 2012–22 collection starts here.

The new manifest preserves all distinct approved packet hashes, including a
second team's packet or additional pre-cutoff source versions. The next immutable
BAM snapshot is created only after root freezes the expanded manifest. Existing
train/dev/test boundaries and exploratory status persist until an explicitly
frozen new evaluation protocol; adding historical notes cannot manufacture a
fresh final holdout.

## Official team records versus the frozen numerical history

The new collector independently identified three Cup championship events in the
cached ESPN history: `espn:401607495` (2023-12-09), `espn:401734908`
(2024-12-17), and `espn:401809839` (2025-12-16). Each cached competition explicitly
has type `39` / `CC` and a championship headline. NBA rules exclude these finals
from regular-season standings and statistics ([official NBA rules](https://www.nba.com/news/nba-cup-101)).

Only the new collector's official prior-record/game-number validation excludes
these three pinned events. It still uses strict earlier-date results and checks
the raw scoreboard SHA, event ID, date, sides and championship classification
before applying an exclusion. Actual prior Cup finals remain in the schedule
used for capture timing/rest windows. Regression tests cover both finalists
before, on and after each final, and reject excluding a normal regular-season
match by an arbitrary ID list.

The original booster, normalized feature history, frozen inputs and earlier
results remain unchanged. Their prior-game features include these real prior Cup
final games, so those features are not strictly regular-season-only. This is a
scope/description issue rather than a target-outcome timestamp leak. Preserving
the original booster is deliberate for the controlled comparison; a separate
regular-season-only numerical baseline would be a future explicit experiment.

Pinning and raw-event provenance are recorded in
`data/provenance/claude_archive_research/20261101T2212Z/nba_cup_record_policy.json`.
Previously rejected Lakers/other finalist packets are re-evaluated with official
records; dates, seasons, venues, publication proof and all other content gates
are not relaxed.

## OCR feasibility and remaining HTML leads

The local Apple Vision probe for the scanned Milwaukee 2023-10-26 front page is
preserved in `ocr_feasibility/MIL_20231026/`. Raw and decoded PDFs, the rendered
page, engine/build details, executable Swift source, observation bounding boxes,
confidence values and derived text have independent hashes. No OCR source was
approved. Although the header is legible, all 460 observations reported confidence
1.0, including a visible `FG%` → `F6%` substitution. Table cells also fragment into
separate observations. Confidence alone therefore cannot approve numeric facts.
Full-packet OCR requires a separately audited page-rendering and column-recovery
process, representative visual checks, and explicit numeric-field verification.
This work does not silently substitute OCR text into the initial or expanded
snapshots.

The team-report Blazers index and article metadata were copied into durable
`data/provenance/claude_archive_research/20261101T2212Z/team_leads/` with original
paths and hashes. Offline title and publication-window matching produces 121
current candidate articles, nine with pre-cutoff Wayback leads, in
`data/processed/bam_notes_expansion/v1/por_candidate_leads.jsonl`. Publisher game
tags are not trusted for schedule matching. Collection-only `--stage por`
preserves exact replay bytes and CDX SHA-1 proof; it never approves an article.
The current Knicks parser requires a date in the title, while these Portland
titles generally omit it. They require a publisher-specific header/date adapter
and full-article review before admission; CMS timestamps alone remain a weaker
quarantined tier.

After the single replay drain finishes, `--stage recheck --offline` now reassesses
all cached strict PDF candidates against the current reviewed templates and
official-record policy. This applies the new Washington and readable Milwaukee
template proofs without redownloading them. No concurrent manifest writers
should be run. A final expanded corpus must use copied, hashed manifest freezes,
then `bam_data_html.py`, rather than mutable collection-progress files.

Readable Warriors packets receive one narrow header adapter in
`scripts/bam_pdf_notes.py`. Their literal nickname matchup line, both official
prior records, a full weekday/month/day/year line, exact publisher game number
and home/road game label must all agree independently. Only then can the old
checker's full-city-name and missing-independent-record failures be resolved;
all other failures remain. The original source text is never rewritten. Visual
evidence is pinned in the expansion configuration, and adapter code hashes are
retained with the approval checks. Regression cases reject wrong opponents,
records, dates, weekdays, numbers, routing and duplicate/ambiguous headers.

Every expanded PDF approval now has an exact content-addressed review
configuration under `data/raw/bam_notes_expansion/v1/review_config_versions/`.
Earlier versions were recovered against the hashes already stored in approved
records; the recovery audit records exact matches. Future passes preserve these
versions and immutable pass summaries directly. The expanded snapshot builder
pins these review files in its provenance. The initial snapshot and its hashes
remain unchanged.
