# Qwen3 V2 independent adjudication

The source-first V2 prompt did not improve the two remaining Qwen3 semantic failures. Its strict score is also dominated by the fragile requirement that the model reproduce exact quotations from layout-preserving PDF lines. These are training-only diagnostic cases; this result does not establish full-report quality or held-out forecasting performance.

Inputs: `data/probes/bam_cloud/20261101_reasoning_quality/qwen3_v2_result.json`; frozen benchmark `data/processed/bam_quality/teacher_reference_v2`, hash `e59a77b6bfde4362114b923f1b809bd96f6692bcb14f4adcdd030598a32d21b7`. Model: `Qwen/Qwen3-32B-AWQ@0499c3ac83fdef8810b907a23894ba91e95eddd8`. Machine-readable independent result: `qwen3_v2_independent_score.json`.

| Measure | Result |
| --- | ---: |
| Valid responses | 22/40 |
| Strict binary support accuracy | 21/40 (52.5%) |
| Admitted false accepts | 0/16 |
| Attempted false accepts before validation | 2/16 |
| Supported references excluded, including invalid outputs | 12/24 |
| Invalid quotation/source-line alignment | 17 |
| Invalid supported verdict inconsistent with its own evidence matrix | 1 |
| Raw verdict-only binary diagnostic, ignoring validation | 37/40 (92.5%) |
| Complete generation batch time, counted once | 116.361 seconds |

Zero admitted false accepts is not evidence of semantic reliability: both incorrect supported decisions were blocked by quotation validation. V1 had the same two semantic false accepts and a 38/40 raw binary score. The raw diagnostic is not an alternative acceptance rule.

- **legacy-10, incorrect player:** Qwen3 attributes the 52-game made-three streak to Duncan Robinson again. The packet's page 1 player-card layout places that streak with Tyler Herro (#14); Robinson's 69- and 57-game streaks are historical comparisons. The response cites lines 48–49 but supplies text spanning farther lines and asserts that all entity checks match. This is a substantive layout/entity error, with an additional citation failure.
- **contrast-08, original citation incomplete:** Qwen3 again accepts the combined Miami 108.4-points-allowed and 15-game claim while citing only page 3 line 7. The 15-game information continues on line 8, which the candidate did not cite. The generated quotation silently joins the following sentence while naming only line 7. The model marks citation completeness not applicable. This is both a citation-scope reasoning failure and a copying failure.
- **diverse-20, unnecessary exclusion:** The model recognizes Brooklyn's 23 makes and 61 attempts against Oklahoma City, but demands the supplementary chart at page 7 line 63. The original lines 1–2 and 4–5 already provide the narrative's game context and attempt count. The supported reference remains supported; requiring redundant chart evidence is an unjustified exclusion.
- **legacy-07, self-inconsistent matrix:** The source explicitly says Miami is on the second game of a road back-to-back after Toronto the previous night (page 4 line 9). The response correctly supports the claim but labels citation completeness incomplete while saying the candidate includes that key line. The consistency validator correctly rejects that contradictory response; the underlying fact is supported.
- **diverse-05, representative quotation fragility:** The response reads Michael Porter Jr.'s season averages correctly, but reconstructs prose and inserts the player's name into a quotation rather than reproducing the exact line. That reconstruction is not faithful quotation, even where the substantive fact is correct. Other invalid outputs require case-specific line review; they must not all be characterized as harmless whitespace errors.

The completed V3 protocol keeps the same claims, sources and labels, requires source line IDs plus explicit source readings, and lets code materialize canonical quotations. This removes the copying requirement without treating copied source text as proof of entailment. V3 must still be judged on original-citation completeness, entity attribution, numeric values and units, date/season scope, and unsupported exclusions. Its results and the three complete training-game reports remain separate required quality checks. No frozen implementation or benchmark inputs were changed for this adjudication.
