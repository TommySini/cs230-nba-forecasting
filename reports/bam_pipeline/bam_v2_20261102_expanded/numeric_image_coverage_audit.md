# Numeric image coverage audit

Read-only inspection of the final expanded snapshot. No snapshot, canonical text, worker, collection configuration or transcription was changed. This report audits generated request payloads; it does not claim a model has already emitted an incorrect disposition.

## Exact pages

| Game / publisher | Physical PDF page | Source ID | Canonical text |
|---|---:|---|---|
| espn:401585678 / BKN | 16 | `sha256:bbf36c10626ef03cb9a08c3ef074feb8123a118a51f254b6ac5e7e88907286c7` | Cumulative-stats title at line1 and footer at line70; no table cells. |
| espn:401703421 / DET | 24 | `sha256:3e9f6bb232643baef19bcb96016d6332927836e4f0167b105508f85226f64177` | One footer/branding line; no player bio or table cells. |

Both sources are active, `source_kind=pdf`, `pagination=pdf_pages`. Both pages have `method=pypdf_layout` and `flags=[sparse_or_empty_text_needs_visual_review]`. The exact full source metadata, extraction text/checksums, nonempty lines and request IDs are in `image_page_request_trace.json`.

The sidecar binds snapshot manifest SHA `b0e6d0265871448580683cb6dcf4c439751857c1e0dc4493a3108a67a68f27b8`. Its SHA is `1cba9d977f8bb03d8fc6f7651c2c9df80aa4100dbeae401ec1715c0411643200`. Each page binds the original raw source SHA, exact physical page, exact text SHA and reviewed render SHA. Text SHA uses SHA256 over `page.text.encode("utf-8")`, preserving original whitespace and newlines.

## Observed path into research

1. `load_snapshot_game` retains the page flags, method, original text representations and lines. The images themselves are not model inputs.
2. `build_selection_requests` emits one request for each of these pages, including all original canonical lines. `_page_metadata` forwards page/page-kind but drops flags and extraction method. Neither source currently supplies a numeric-image source warning.
3. Table decoding produces no matching `undecoded_tables` entry for either image page because the numeric cells are absent from the text. Brooklyn has no matching failure; Detroit has an unrelated decoder failure on page1, which does not identify the page24 gap.
4. Selection permits arbitrary text in `disposition`; it has no mandatory extraction-gap unknown. There is no deterministic instruction saying these pages have no game-relevant information, but the sparse visible text can invite that unsupported inference.
5. `build_module_requests` preserves selected findings and reviewed-chunk counts/hashes, but discards selection dispositions. `_evidence_for` retains pages linked to findings; a page yielding no findings can disappear from later evidence. No deterministic numeric-image coverage gap currently survives into module unknowns or the reviewed-chunk count qualification.

## Minimal handling for the next protocol

- Preserve the immutable snapshot. Load and hash-pin `page_quality_sidecar_v1.json` in the new protocol, validate manifest/source/page/text/render hashes, and attach its matching entry to an in-memory `source.pages[].coverage_audit` after source validation. Bind this derived audit state into request/artifact hashes.
- Forward existing extraction flags/method for all pages. A sparse flag means extraction uncertainty until visually classified; empty text alone cannot establish a blank page or low relevance. The `layout_omitted_text_recovered_from_plain` flag is a resolved fallback, not an automatic missing-content claim.
- Forward these two trusted audit entries into selection and all downstream module/merge/review coverage payloads even when they yield no findings. Keep the mandatory unknown/coverage note that substantive numeric image content is untranscribed, so numerical detail, relevance and booster overlap remain unknown. Qualify reviewed-chunk counts as review of available extracted text, not complete visual/numeric coverage.
- Preserve the mandatory unknown deterministically or mechanically validate its presence. Do not rely on a free-form selection disposition to survive aggregation.
- Treat coverage audits as non-citable preprocessing metadata. They authorize an unknown, never numerical values, player statistics, decoded-table IDs or factual basketball findings. No image-cell number can be supported by the title/footer lines.

## Earlier flagged-page review

The initial review covered 15 flagged pages, not 15 empty pages: four Detroit pages were intentional branded/header/footer blanks; eleven Boston pages were substantive, legible numeric tables with extracted text present. The Boston flag concerns nonsemantic U+0000 marks at section boundaries, not missing numeric table bodies. Those eleven pages still require valid table decoding and season/entity/window interpretation; visual legibility is not a decoder certification. No image-only numeric gap was found among the original fifteen.

Across the expanded snapshot there are nine sparse flags: six visual blanks, these two image-only numeric tables, and one Minnesota headshot roster with untranscribed image text. Eleven additional Boston suspicious-glyph flags bring the reviewed visual-flag total to twenty. Headshots do not establish lineup roles or availability.

## Transcription assessment

Complete, accurate, independently double-checked transcription of both pages is not credible within15 minutes. Brooklyn contains a dense cumulative player table plus team/opponent totals, many paired fields and percentages. Detroit mixes multiple player profiles, recent-game tables, season/career/high columns and historical NBA/G-League prose; its top profile continues from the preceding page, so identity verification also needs that context. Hundreds of numeric cells and their row/column/league/season associations require deliberate verification.

No transcription or OCR admission was attempted. These remain explicit coverage gaps. A future transcription must preserve the raw PDF/render hashes, capture every header/row/denominator and applicable continuation context, independently compare every numeric field against the rendered page, and create a new version before values can become citation-supported evidence.
