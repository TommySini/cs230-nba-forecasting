# V3 72B independent source audit

Independent scoring matches the reported result: 40 valid responses, 29/40 correct, one false accept among16negative cases, and10false exclusions among24supported cases. Generation batch time is126.567seconds. These are purposive training cases and do not estimate broad reliability.

The false accept is **contrast-03**. Boston page6 line67 explicitly says the20home wins were followed by a102–100loss toDenver onJan19. The model copies that loss into its source reading but then says the same sentence supports an active unbeaten20-game streak as ofJan29. Every attribute is marked match. This is a clear semantic failure despite fully valid structured output and canonical source text.

All10exclusions are also reviewer errors. Eight primarily assert that a citation is incomplete when it contains the supporting lines or includes useful extra context:

| Case | Supporting lines already in original citation |
| --- | --- |
| legacy-04 | MIAp10L26,Adebayo38:36; player andMINcolumn headers also included. |
| legacy-05 | MIAp11L73,Herro36:49. Reviewer expressly acknowledges the line is included. |
| legacy-06 | MIAp11L24,Butler34:25. Reviewer expressly acknowledges the line is included. |
| legacy-07 | MIAp4L9–10,roadback-to-back. Reviewer falsely says both are omitted. |
| diverse-01 | DETp1L13preseasonheader,L15player,L19columns,L20fourgames11.8PPG9.3RPG. The allegedly unnecessaryL13is essential season/scope context. |
| diverse-06 | DENp3L6season/columns,L7Dallas110.2PPG115.9allowed. Reviewer falsely saysL7is omitted from[5,6,7]. |
| diverse-11 | BOSp1L12season,L14player,L15columns,L17Tatum43games26.8/8.4/4.3. All allegedly missing lines are supplied. |
| diverse-12 | BOSp1L12season,L14player,L15columns,L16Tatum50games26.8/8.6/5.6. All allegedly missing lines are supplied. |

The remaining two additionally mishandle supported limitations: **diverse-10** rejects the one-game-sample caveat despite the explicit1/1games row; **diverse-17** rejects the statement that prior Durant personal absences do not establish tonight's status, explaining that the source supplies no upcoming-game status. That absence is precisely why the caveat is appropriate. Both also misuse citation completeness.

This result does not support choosing72B solely because it is larger. Do not repair these scores by relabeling supported cases or stripping context. Keep all benchmark versions frozen. For the live three-game pipeline, inspect actual reviewer rejections and surviving evidence before deciding a production remedy or review model. A reviewer may correctly reject dangerous claims yet still remove useful supported context; the complete reports must be assessed for both errors.

Inputs: `data/probes/bam_cloud/20261101_quality_v3/72b_result.json` and frozen V3requests. Score: `v3_72b_independent_score.json`. Model `Qwen/Qwen2.5-72B-Instruct-AWQ@035adeffb861e8ee98344e2c7d8f50fe585e373d`; benchmark `7058d7bc02b7b0bc27b8b4aa9cb750d93c7fa8dc446b55d053f054451b0eae4a`. No implementation or benchmark edits.
