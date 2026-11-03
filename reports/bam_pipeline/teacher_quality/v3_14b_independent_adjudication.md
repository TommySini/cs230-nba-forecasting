# V3 14B independent source audit

All eight scored false exclusions are confirmed against the original candidate citations and source lines in the frozen V3 requests. The result is 40/40 valid, 32/40 correct, 0/16 false accepts and 8/24 supported facts excluded. Complete generation batch time is 57.092 seconds. Zero observed false accepts on these 16 purposive training cases does not establish general reliability.

| Case | Source-grounded adjudication |
| --- | --- |
| legacy-05 | The candidate explicitly includes MIA page11 line73, showing Herro36:49 atToronto12/01/24, plus player/header context. Saying no exact supporting lines were supplied is false. |
| legacy-06 | The candidate explicitly includes MIA page11 line24, showing Butler34:25 atToronto12/01/24, plus header context. The alleged missing citation is present. |
| legacy-07 | MIA page4 lines9–10 explicitly state second road back-to-back after Toronto last night. Both are in the candidate's citation; a larger citation set is not incomplete merely because it contains additional context. |
| diverse-09 | MEM page5 line9 contains both Iisalo's March28 appointment and his prior lead-assistant role. Candidate cites lines1,9. No additional line is missing. |
| diverse-10 | BOS page1 lines12,14,16,17 identify Tatum, season, columns and1/1games with34PPG/11RPG. The phrase that the one-game sample matters is a cautious interpretation of the explicit sample size, not an unsupported new statistic or causal claim. |
| diverse-17 | HOU page1 lines6–9,12–13 support Durant's two prior personal absences and Nov26inactive status. The caveat that these do not establish tonight's status correctly limits inference. Extra season/column headers do not make the citation incomplete. |
| diverse-18 | TOR page5 lines1,2,4,7,8 establish injury table/season, Barrett and the two distinct absence windows. QUESTIONABLE aligns with the December illness row. Season/title headers are useful context, not evidence of incompleteness. The model's source reading also incorrectly repeats QUESTIONABLE onto the older October shoulder row, though that is not the candidate claim. |
| diverse-20 | BKN page7 lines1–2,4–5 describe the prior Wednesday game and historical comparisons. The candidate never claims the game occurred on the forecast date Feb28; the model invents that assertion and contradicts it with Feb26. The cited lines already contain the historical comparisons it says are absent. |

Six exclusions invoke incorrect citation-incompleteness claims; two also mishandle uncertainty or date scope. The broader pattern is substantial false exclusion of grounded evidence even after quote-copying fragility was removed. Keep the fixed benchmark unchanged and compare the two remaining V3 models before choosing a reviewer. Full-game coverage and interpretation still need manual review.

Inputs: `data/probes/bam_cloud/20261101_quality_v3/14b_result.json`, `14b_score.json`, and `data/processed/bam_quality/teacher_reference_v3/requests.jsonl`. Model `Qwen/Qwen2.5-14B-Instruct@cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`; benchmark `7058d7bc02b7b0bc27b8b4aa9cb750d93c7fa8dc446b55d053f054451b0eae4a`. No code, labels or benchmark requests changed.
