# Research the complete game-note packet before forecasting

The next system processes every page of the verified pregame PDF. The completed
first-page experiment remains a frozen pilot result; its scores do not measure
this new design.

The first-page restriction was an implementation shortcut: its templates had
been reviewed, and the local training run used a 3,072-token input budget. It was
not based on evidence that later pages lacked predictive value. The new corpus
removes that restriction.

## Evidence workflow

1. Verify the complete archived PDF's checksum and capture time against the
   game's cutoff. Retain its matchup, publisher and original URL.
2. Extract every page in both layout-preserving and plain-text forms. Keep the
   original PDF for images, difficult tables and visual checks. Flag sparse,
   unreadable or corrupted extraction instead of silently treating it as no news.
3. Read every page through bounded chunks. Every selected full-page text character appears
   exactly once in the chunk inventory. There is no packet-level page or length
   truncation. Processing a long packet in sections is distinct from omitting
   sections. Both original text extraction variants remain available; a fallback
   selects plain text when the layout parser omits text, and reviewed visual
   transcriptions can recover garbled fonts. Every chunk must have a research result, including explicit
   no-relevant-evidence or needs-visual-review results.
4. Extract evidence about availability, roles and lineup changes, workload,
   travel/rest, recent form, matchup statistics and transactions. Preserve dates,
   numeric denominators, comparison periods and uncertainty. A previous game's
   starters are not tonight's confirmed lineup. Historical injury absences are
   not current availability. Promotional streaks are not automatically predictive.
5. Merge the chunk findings into one game report with page citations, exact
   supporting quotes, duplicate/conflicting claims, and missing information.
   Distinguish source facts from basketball interpretations. Check quotations
   against the frozen page text, then review whether the claims actually follow
   from them. Quote matching alone does not prove that.
6. Only then give the evidence report and original boosting probability to the
   forecasting model. Keep the archived packet and all evidence records available
   for inspection. Do not invent a percentage adjustment directly from an
   unsupported basketball explanation.

The evidence researcher does not receive the target outcome or boosting
probability. This keeps the extraction stage focused on the source. The
forecaster receives the boosting probability later. Both stages must use only
pre-cutoff evidence; the chosen research and forecast checkpoints must precede
their evaluation games. Generated reports should be pinned before training or
scoring, and the next final test must be a fresh held-out period.

This follows BAM's separation of evidence gathering from probability forecasting,
without claiming to reproduce its complete architecture or results:
[BAM, section 4](https://arxiv.org/html/2607.09921v1#S4).

## What Qwen's numerical representation meant in the pilot

The input was text containing the teams, boosting probability and first-page
excerpt. Tokenization divided it into word pieces, numbers and punctuation and
assigned each piece an integer ID. These IDs are labels, not measurements of
basketball meaning.

Qwen looked up a learned vector of 896 numbers for each token. Its 24 transformer
layers repeatedly updated those vectors using attention to earlier tokens,
position information and nonlinear transformations. This allows the same word
to acquire different representations in different sentences. The dimensions
are learned, distributed features; there is no predefined dimension called
"injury severity" or "home advantage."

The sequence-classification head used the final non-padding token's contextual
vector. A learned weighted sum of its 896 components produced a scalar
correction, applied as:

`p_hybrid = sigmoid(logit(p_boost) + correction)`

The head started at zero. Outcome supervision updated the head and small LoRA
adapters, while the pretrained backbone weights stayed fixed. This computation
does not itself produce a written evidence report or establish which passage
caused an adjustment. The new research stage makes source interpretation
inspectable before it reaches that forecast.

## Current implementation boundary

`scripts/prepare_full_game_notes.py` prepares and validates the complete text and
research requests for the same 107 packets used in the historical pilot. Its
source, page and character coverage checks reject missing pages or truncated
chunks; its citation validator checks exact quotes and source identity.

```bash
.venv/bin/python scripts/prepare_full_game_notes.py
.venv/bin/python scripts/prepare_full_game_notes.py --validate
```

The resulting [index](index.md) links every PDF and its complete extracted text.
Corpus preparation is not an automated semantic review of all packets. The
research requests are prepared prompts, not completed model calls. Model-backed
extraction/merging, visual-content resolution, report-quality evaluation and
forecaster retraining are subsequent work. No new Brier score is claimed.

Before a new forecasting run, evaluate whether the research reports preserve
important later-page facts, handle units/time scopes correctly, retain conflicts,
and add evidence beyond the booster's existing 18 statistical features. Keep
probability calibration and text contribution distinguishable in development
diagnostics. Do not choose the new design using the old 14 test outcomes.
