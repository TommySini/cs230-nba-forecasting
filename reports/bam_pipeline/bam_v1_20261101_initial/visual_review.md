# Flagged PDF page visual review

Reviewed all 15 flagged pages from the frozen 256-game snapshot. This is an outcome-blind layout and encoding audit, including one test packet; no outcome labels, postgame sources or target-game results were opened. No immutable snapshot or global extraction override was changed.

Poppler is unavailable in the repository environment. Pages were rendered at 1,200 pixels wide with macOS PDFKit/AppKit using the exact decoded archived bytes. The page PNGs and their SHA-256 hashes are retained in `visual_review_images/` and `visual_review_proposed_overrides.json`.

## Findings

| Publisher / packet date | PDF pages | Finding | Action |
| --- | --- | --- | --- |
| DET / 2023-10-25 | 32 | Blank body with game-note branding/header/footer. | Preserve as intentional blank; no OCR or transcription. |
| DET / 2023-12-13 | 34 | Blank body with footer. | Preserve as intentional blank; no OCR or transcription. |
| DET / 2024-01-15 | 32 | Blank body with footer. | Preserve as intentional blank; no OCR or transcription. |
| DET / 2024-03-15 | 33 | Blank body with game-note branding/header/footer. | Preserve as intentional blank; no OCR or transcription. |
| BOS / 2026-02-11 | 36, 39, 43, 46–53 | Complete, legible tables. Extracted U+0000 corresponds to a small hollow-box mark at a section boundary; no surrounding text appears lost. | Optional NUL-only cleanup in a future derived view; preserve all line breaks. |

The 11 Boston pages contain 17 NUL characters. The proposed JSON includes complete replacement text, original/replacement text hashes and rendered-image hashes. It removes only U+0000 and leaves every newline, number, label, row and ordering unchanged. It is not applied, does not certify table-column decoding, and must not silently replace citation text in the current frozen run. Original raw PDFs and canonical page/line text remain authoritative.

Dense basketball tables still require explicit heading/season/column interpretation. A clean render is not proof that a downstream model reads totals, averages, career highs or traded-player rows correctly. No numerical accuracy or forecasting choices were assessed from the test pages.

## Per-page evidence

- DET 2023-10-25 page 32: `intentional_blank_body`; [render](visual_review_images/DET_2023-10-25_p32.png).
- DET 2023-12-13 page 34: `intentional_blank_body`; [render](visual_review_images/DET_2023-12-13_p34.png).
- DET 2024-01-15 page 32: `intentional_blank_body`; [render](visual_review_images/DET_2024-01-15_p32.png).
- DET 2024-03-15 page 33: `intentional_blank_body`; [render](visual_review_images/DET_2024-03-15_p33.png).
- BOS 2026-02-11 page 36: `nonsemantic_null_glyph_at_section_boundary`; [render](visual_review_images/BOS_2026-02-11_p36.png).
- BOS 2026-02-11 page 39: `nonsemantic_null_glyph_at_section_boundary`; [render](visual_review_images/BOS_2026-02-11_p39.png).
- BOS 2026-02-11 page 43: `nonsemantic_null_glyph_at_section_boundary`; [render](visual_review_images/BOS_2026-02-11_p43.png).
- BOS 2026-02-11 page 46: `nonsemantic_null_glyph_at_section_boundary`; [render](visual_review_images/BOS_2026-02-11_p46.png).
- BOS 2026-02-11 page 47: `nonsemantic_null_glyph_at_section_boundary`; [render](visual_review_images/BOS_2026-02-11_p47.png).
- BOS 2026-02-11 page 48: `nonsemantic_null_glyph_at_section_boundary`; [render](visual_review_images/BOS_2026-02-11_p48.png).
- BOS 2026-02-11 page 49: `nonsemantic_null_glyph_at_section_boundary`; [render](visual_review_images/BOS_2026-02-11_p49.png).
- BOS 2026-02-11 page 50: `nonsemantic_null_glyph_at_section_boundary`; [render](visual_review_images/BOS_2026-02-11_p50.png).
- BOS 2026-02-11 page 51: `nonsemantic_null_glyph_at_section_boundary`; [render](visual_review_images/BOS_2026-02-11_p51.png).
- BOS 2026-02-11 page 52: `nonsemantic_null_glyph_at_section_boundary`; [render](visual_review_images/BOS_2026-02-11_p52.png).
- BOS 2026-02-11 page 53: `nonsemantic_null_glyph_at_section_boundary`; [render](visual_review_images/BOS_2026-02-11_p53.png).
