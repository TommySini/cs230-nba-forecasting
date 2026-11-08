# Provenance review of game-notes v2

Reviewed canonical commit `723c8b0743ddc7acd42df60b4023ed8c969dc6cb`
read-only. The canonical checkout was clean at inspection. No collector, dataset,
model, cloud job or paid API call was run or changed. The frozen 424-game BAM
snapshot remains separate. Counts below refer to the local v2 manifests inspected,
not an eligibility endorsement of all 1,997 games.

**Recommendation:** preserve the expansion as a valuable candidate corpus, but do
not import its `approved` flag as strict BAM pregame approval. Preserve the two
provenance tiers, exclude or recover pre-cutoff versions of the 80 HTML documents
explicitly modified after cutoff, and bind derived text to canonical source
bytes before building a new immutable experiment snapshot.

## Findings

### 1. Eighty approved HTML documents have post-cutoff modifications

`collect_notes_v2.py:474–493` accepts HTML when `published_utc <= cutoff` but
records `modified_after_cutoff` without making it a rejection. The validator
checks publication only (`validate_notes_v2.py:28–31`). Of 252 approved HTML
documents, **80 explicitly have `modified_after_cutoff: true`**.

Examples from metadata alone:

| Game | Publisher | Published UTC | Modified UTC | Cutoff UTC |
| --- | --- | --- | --- | --- |
| `espn:401585340` | POR | 2024-02-04 16:53:04 | 2024-02-05 01:30:08 | 2024-02-05 00:30:00 |
| `espn:401705095` | NYK | 2025-01-10 14:25:12 | 2025-01-15 15:29:22 | 2025-01-10 23:30:00 |
| `espn:401705105` | NYK | 2025-01-12 14:38:32 | 2025-01-16 14:46:28 | 2025-01-12 19:00:00 |

These may be harmless edits, but the metadata does not establish that the fetched
revision existed at prediction time. An original publication timestamp plus a
negative result-pattern screen cannot establish that. The collector explicitly
documents a relaxed publisher-timestamp policy; this review does not silently
reinterpret that policy as archive proof. Under the frozen strict BAM version
policy, these revisions need an independently verified earlier capture or must
remain outside prediction inputs. Even publisher revisions modified before
cutoff retain a weaker evidentiary tier than archived bytes.

### 2. Derived text is not bound to the verified raw source

The validator hashes PDF bytes or a whole HTML API response, but then independently
reads `text_path` without an expected text hash or reconstruction check
(`validate_notes_v2.py:30–44`). It does not rerun header matching. `extract` also
trusts any existing text file (`collect_notes_v2.py:401–405`). Consequently an old,
incorrect or accidentally replaced text file can retain a valid raw-file checksum
and pass the current validator if the limited result screen finds nothing.

HTML has a related structural risk: text paths use only game ID and publisher,
and each candidate writes that path before `choose` selects a document
(`collect_notes_v2.py:484–486,498–506`). Multiple articles for one team-game could
therefore overwrite the selected article's text. This is a code-path risk; the
separate data audit owns whether actual collisions occur. API page SHA alone does
not identify the selected article body within that response.

A new snapshot should bind raw SHA, actual replay URL/timestamp or publisher
revision metadata, article identity, canonical extracted page/section lines,
extraction settings and derived text/page SHA. Reconstruction must match these
bindings; successful hashing of one layer does not verify another layer.

### 3. Header matching accepts an explicitly wrong numeric year

In an in-memory synthetic regression, a 2025-01-01 game with opening text
`BOSTON CELTICS vs NEW YORK KNICKS 01/01/2023` plus enough ordinary words returns
no header rejection and `header_date: true`. The month/day regex can match the
prefix while its optional year is omitted; the contradictory-year check only
handles spelled-out months (`collect_notes_v2.py:355–363`). Printed records are
optional, so they do not necessarily repair this association.

This does not prove a particular approved document is mismatched. It establishes
that the advertised exact game/date header check is incomplete. Explicit years
must agree for numeric and textual dates, and a date plus two team mentions
somewhere in 6,000 characters should not be treated as an independently reviewed
masthead. The existing tests do not cover the numeric wrong-year case.

### 4. The postgame screen is a diagnostic, not absence-of-leakage proof

The screen matches selected score, record and result-sentence formats. With a
synthetic BOS–NYK game ending 112–105, each of these returns no flags:

```
FINAL
BOS 112
NYK 105
```

```
The Celtics won 112 to 105 against the Knicks.
Boston prevailed over New York in overtime.
```

The validator still prints “no game results in text” and “evidence predates every
cutoff.” Its actual guarantee is narrower: stored raw hashes match, specified
timestamp fields pass their tier-specific checks, and the implemented patterns
did not fire. Unrecognized result text, revised injuries or later statistics are
not ruled out. These synthetic failures are not an allegation that the approved
real texts contain those outcomes.

The screen also uses the realized game result to decide exclusion. That is
appropriate as an offline contamination diagnostic, never a forecast input;
record any exclusions as outcome-aware curation and avoid presenting the resulting
sample as prospective, outcome-blind source selection. Prefer temporal source
eligibility independent of score-pattern coincidences.

### 5. Full-document extraction does not certify readable full-page coverage

All PDF pages are traversed, but `page.extract_text() or ""` retains empty pages
without extraction-quality flags (`collect_notes_v2.py:392–398`). Header and word
count tests cannot establish readability of later image-only tables. The new
snapshot needs the same explicit unreadable-page/undecoded-numeric coverage audit
as BAM v2; do not turn an empty page into a claim that nothing relevant was there.

## What is already sound, and remaining boundaries

Archive matching uses the **full UTC cutoff**, not a same-calendar-day assertion.
Replay admission checks actual final replay time against `Memento-Datetime` and
requires both at or before tipoff minus 60 minutes. This rejects post-cutoff
redirected captures. In all **1,634** actual approved archive records inspected,
the replay's original URL also matched the requested original modulo HTTP/HTTPS.
The code does not itself require that URL equality, so it should remain an
explicit identity check in the eventual snapshot loader.

Live dated PDFs require `Last-Modified <= cutoff` and preserve raw SHA. This is
publisher evidence rather than an independent archive capture. The overall
manifest contains 1,634 archive-proven and 510 site-timestamped documents, with
the latter comprising 258 live PDFs and 252 HTML posts. Keep these distinctions
in analyses and eligibility counts.

The history source collector filters completed ESPN regular-season events
(`collect_pilot.py:89–93`). V2's own `load_history` only filters team membership;
it relies on that upstream contract and does not independently validate season
type. Any postseason extension must be explicit: playoffs reset/report records
differently and cannot simply reuse regular-season record/game-number matching.
The separate data audit owns actual schedule coverage and duplicate counts.

The validator enforces one chosen document per `(game_id, team)` and checks no
outcome keys at the document record's top level. This is useful, but neither a
recursive model-input schema validation nor source-byte uniqueness across games.
Never feed its whole raw API responses to an LM; materialize only the selected,
bound article content and canonical safe source metadata.

## Historical model eligibility

Older games can expand TRAIN research supervision. Under the agreed BAM-style
criterion, every component that influences held-out predictions—including
selector, partial synthesizer, research student, repair/review critic and
forecaster—must have an exact pinned identity and a documented knowledge cutoff
before DEV/TEST. Release before DEV/TEST is a conservative sufficient alternative
when a reliable knowledge cutoff is unavailable; later release alone is not an
automatic exclusion. Thus the reported June 1, 2024 knowledge cutoff for both
gpt-oss models makes them BAM-style candidates for 2024–25 DEV despite their
August 2025 release, without proving absence of game-level memory. Document any
later instruction tuning or distillation and the scope of the cutoff evidence
explicitly; quantization does not erase later training in a derived checkpoint.

A modern teacher may produce TRAIN-only postmortems, gold and gold review, with
the existing separate hindsight store and output checks. That is retrospective
distillation, not a claim that every training explanation was produced by a
historically available model. Fit each research fold only on earlier games; do
not apply API critics without a qualifying documented knowledge cutoff to
held-out report acceptance and call the complete forecasting pipeline historical.

The committed memorization probe is explicitly preliminary (400 games from
2023–24, 92 from 2024–25). Low exact-score recall and weak winner discrimination
cannot prove absence of pretraining contamination, particularly with different
prompts or contextual cues. It is not grounds to move DEV/TEST before the
documented knowledge cutoff. The probe also uses hindsight season strength only as a diagnostic
comparator; it must remain outside forecast features. No model was run for this
review, and no new model choice is inferred from that probe.

## Read-only validation performed

- The nine canonical `test_notes_v2.py` tests passed.
- The seven canonical `test_postgame_screen.py` tests passed.
- `validate_notes_v2.py` passed all 2,144 documents / 1,997 games under its current
  rules; the limitations above explain why this is not strict BAM acceptance.
- The wrong numeric-year and three postgame-format counterexamples were evaluated
  in memory; no source files or frozen artifacts were changed.
- No collection/rebuild, model inference, paid service or cloud job was launched.
