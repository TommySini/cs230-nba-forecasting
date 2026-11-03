# Independent review of the first teacher/reviewer benchmark

Neither initial candidate is ready to certify training reports. On the fixed 40
training-only cases, both frequently accepted claims contradicted by the supplied
source. Model size did not resolve this failure.

| Candidate | Strict valid JSON | Strict correct / 40 | Complete-fence diagnostic correct / 40 | False acceptances / 16 negatives after fence normalization |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-14B | 39 | 31 | 31 | 9 / 16 |
| Qwen2.5-72B-AWQ | 12 | 11 | 30 | 10 / 16 |

The secondary diagnostic removes only one complete Markdown code fence. Original
responses and strict scores remain unchanged. Both candidates accepted all 24
positive cases after that transport normalization; their main semantic weakness
was rejecting bad claims, not finding support for valid ones. The 72B model's
strict result also exposes substantial formatting noncompliance. Per-request
runtime was not reported in the score inputs and remains unknown there.

Independent checks of the failure evidence confirm the benchmark labels:

- **Wrong table column:** the December 1 Toronto row for Bam Adebayo has FG = 5
  and PTS = 13. Both models accepted “5 points”; their own explanations mention
  13 points. The 72B answer rationalized that 13 included 5.
- **Ignoring a newer row:** Highsmith's November 24 Dallas entry is followed by
  a December 1 Toronto entry. Both accepted the older entry as his latest game.
- **Past versus current status:** previous DNP-CD/NWT entries cannot establish
  tonight's coach-decision absence. Dru Smith's November 2023 ACL note explicitly
  describes the previous season and cannot establish a December 2024 absence.
- **Wrong player in a multicolumn page:** the 52-game made-three streak belongs
  to Tyler Herro, whose #14 section spans page 1 lines 48–52. Duncan Robinson's
  #55 section is at lines 43–46. Both attributed the 52-game streak to Robinson.
- **Wrong player across a section:** the Brooklyn page's KEON heading assigns
  the 14-steal stretch to Keon Johnson. The 72B model accepted Cam Johnson.
- **Scope and logic:** preseason averages were promoted to regular-season
  production; an ended home streak was described as active; a January 2024 recap
  was used as current February 2025 evidence; and a conditional scoring record
  was mistaken for the current scoring leader.
- **Incomplete original citation:** a model could find the missing continuation
  sentence elsewhere on the full page, but that does not make the candidate's
  original citation complete. The benchmark requires `unclear` plus repair
  citations for that case.

The common verdict-before-evidence output order may encourage premature assent;
this is a hypothesis, not an established explanation. Version 2 preserves every
claim, source page, candidate citation and label, while requesting source evidence,
six explicit attribute comparisons, contradictions, and a final verdict in that
order. It also checks quoted evidence against actual source lines and rejects a
supported verdict when the structured comparison itself records a contradiction.
This catches inconsistent outputs; it does not certify that the model identified
all contradictions correctly. New-model/prompt runs remain training-only model
selection work, not heldout forecast evaluation.

A separate mechanical-gate audit found an additional coverage problem: only 9 of
20 manually source-checked positive reference claims were admitted. Fixing mixed
prose/table citation false exclusions and explicit current-season absence wording
raised that to 13 of 20. The remaining seven need conservative decoding of named
player-season averages or named team-season comparison tables. They must remain
unknown until decoded, not silently treated as absent basketball evidence. The
support benchmark and full-report coverage therefore measure different things.

The full-game manual review rubric remains pending actual module/postmortem/gold
outputs. Passing isolated claim checks alone is insufficient to approve research
training or assert broad factual accuracy.
