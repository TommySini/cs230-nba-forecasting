# Preserved reference selection: actual-output diagnostic

This is a local diagnostic of the stopped `bam_reference_v2_20261101` selection batch, not a completed pipeline stage, semantic approval, gold generation, training or evaluation. The original 121 requests and raw responses were preserved. No target-game outcome stores were opened. The source-only checklist supplies expectations independently of these model outputs.

The exact raw result is `data/probes/bam_cloud/20261101_reference_aborted/selection_result.json`, SHA256 `9cc657a248b5381a5482968f7765bab32e6ee8d4df80b469e8048ab21e21efb8`. Model: Qwen2.5-14B-Instruct, revision `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`. The 120 complete responses are mechanically ingestible at the page level but many individual findings are rejected. The remaining response is truncated; its raw partial text was retained despite the transport's empty final `text` field.

| Packet | Requests | Original mechanically retained / rejected findings | After code-owned IDs and canonical table injection: retained / rejected proposals | Canonical tables after fix |
| --- | ---: | ---: | ---: | ---: |
| DEN, Jan 27 2024 | 23 | 27 / 60 | 44 / 50 | 9 |
| CLE, Dec 27 2024 | 52, one truncated | 47 / 107 | 74 / 89 | 9 |
| DET, Feb 28 2025 | 46 | 39 / 103 | 63 / 86 | 7 |

The before/after totals are 113 versus 181 retained findings and 270 versus 225 rejected model proposals, excluding unparseable truncated material. Forty-five false exclusions caused solely by repeated placeholder IDs are removed. All 25 supported decoder facts now reach evidence, versus only two previously. Retained counts are not factual-accuracy counts.

Both runs have separate diagnostic namespaces and recorded validator hashes:

- Before: `selection_v2_diagnostic_summary.json`, validator `1f56ef1f19c072836ad577476747f43059dcb43514f1f8a7bf63776ded76b3ff`, artifacts under `data/probes/bam_diagnostics/reference_v2_selection_compact_validator`.
- After: `selection_v2_code_owned_diagnostic_summary.json`, validator `f129c39474c9d59d8aa141607bb99842c09abe3fd69f02435a6e0423622f2d8d`, artifacts under `data/probes/bam_diagnostics/reference_v2_selection_code_owned_evidence`.

The fixes assign bookkeeping IDs in code and inject exact decoded facts with neutral interpretation, preserving raw model proposals and all rejections. They do not relax numerical/citation checks or certify paraphrases. Exact canonical table proposals with model-authored interpretations are retained in the table audit rather than letting the interpretation inherit decoder authority. Undecoded tables remain unresolved.

## Evidence coverage against the source-only checklist

These are targeted observations, not an exhaustive accuracy score or a requirement to repeat every source fact. “Partial” means useful evidence survives but the necessary scope, countercontext or related observation is missing or flawed. The fourth MIA game is not in this old batch.

| Packet / checklist item | Observed selection behavior |
| --- | --- |
| CLE 01, Strus return | Dec20 debut after absence retained; Dec23 participation is not captured cleanly. Unsupported current-available proposal rejected. |
| CLE 02, Okoro | Last three injury absences retained; recent shoulder cause is not preserved in that claim. Historical injuries are sometimes incorrectly proposed as current. |
| CLE 03, Merrill role | Prior start/performance appears in truncated first-page response, where interpretation wrongly calls it bench production. Canonical 26 games / two starts now retained but does not substitute for Dec23 role detail. |
| CLE 04, current injury information | Current status remains uncertain. No faithful capture of the external injury-link limitation; several retained claims nevertheless infer availability from performance. |
| CLE 05, road trip/rest | Correct upcoming four-game trip/rest context is not captured. A retained March travel claim incorrectly suggests fatigue for the December game. |
| CLE 06, Dec5 Denver matchup | Earlier 126–114 matchup result is not substantively selected. |
| CLE 07, shooting windows | General consecutive 20-plus-three games retained; precise recent versus season window comparison is incomplete. Canonical season rates now survive. |
| CLE 08, team season baselines | Both teams' canonical season tables now injected; original proposals were rejected for changing the decoded fact. |
| CLE 09, bench depth | LeVert/Jerome detail appears in truncated output; other retained bench observations are generic and do not replace the exact recent windows. |
| CLE 10, next opponent | Golden State table facts now survive with original page context. Final module must identify that page as the next-opponent section, not Denver evidence. |
| DEN 01, Gordon | Jan25 minutes proposal rejected for combining paraphrased table values; canonical season baseline now survives. Dec28 Watson-start narrative is retained after ID fix and must remain historical. |
| DEN 02, availability limit | Link-only injury limitation not selected faithfully; an incorrect Jokic-expected-to-play inference survives mechanics. |
| DEN 03, recent trip | Jan25 Knicks loss and completed five-game trip captured on page8. Other retained future schedules are wrongly presented as recent workload. |
| DEN 04, Murray/Porter windows | Current season baselines now injected. Jan25/Jan23 individual contrasts from page1 are rejected combined paraphrases, leaving weaker older examples. |
| DEN 05, Jokic recent production | Recent Indiana production selected, but fact_date incorrectly assigned Jan20 instead of the schedule's Jan23. |
| DEN 06, Jan16 Philadelphia meeting | Prior meeting result/Embiid context is not captured as a usable comparison. |
| DEN 07, opponent tables | Philadelphia/Denver canonical tables retained; next-opponent Milwaukee tables now also present and require relevance filtering with headings. |
| DEN 08, old streak | Jan1 14-game streak selected, initially lost only to duplicate ID. Date is Jan1, but phrasing/interpretation risks treating it as current. |
| DEN 09, date inconsistency | Jan19 schedule versus Jan20 Boston recap discrepancy is not recognized. |
| DET 01, Ivey | Explicit OUT/fracture retained, reproducing source typo “Jadey”; roster confirms Jaden. |
| DET 02, Fontecchio | Two recent absences retained, but interpretation unjustifiably suggests continued inactivity. Incorrect current combined toe/finger absence proposal rejected. |
| DET 03, streak window | Eight wins and recent statistics captured, but first-page 105.9 defensive rating mislabeled season-to-date. Page6 preserves the correct streak window. |
| DET 04, paint comparison | Comparative paint evidence retained; 14-versus-15-game inconsistency is not recognized. Some duplicate-ID losses now restored without resolving that conflict. |
| DET 05, Beasley | Strong season/bench shooting selected on later pages; exact Feb26 six-three/26-point performance not substantively captured. |
| DET 06, Boston absences | Brown/Kornet opposing context from the prior Boston win is omitted. |
| DET 07, Cunningham windows | Canonical season baseline now injected; recent 24-game contrast remains missing. |
| DET 08, current schedule | Feb28 as first night before Mar1 Brooklyn is not captured correctly. |
| DET 09, old Denver absences | Model proposes Dec28 Denver inactive list as current; mechanical current-status check rejects it. |
| DET 10, Stewart | Blocks/rim/bench evidence selected, with relevant uncertainty; some descriptive interpretations remain ungrounded, such as treating .600 field-goal shooting as weak. |

## Specific retained errors requiring semantic/manual review

- **Wrong unit:** DET page38 line36 says `TOTALS 3,044 ...`; selection calls 3,044 an average. Other totals on that page receive the same treatment. The number-presence guard cannot establish the unit and the current dense-row detector does not quarantine this row. This is not a valid decoded table fact.
- **Wrong comparator:** DEN page3 canonical rows show Denver12.8 turnovers and Philadelphia12.0. The former model interpretation claims Denver has fewer turnovers and that Philadelphia defense is absent. Canonical table injection now keeps exact figures with neutral interpretation; the original incorrect interpretation remains in the table-proposal audit.
- **Availability from silence:** DEN page7 seasonal accolades produce “Jokic is expected to play,” with an uncertainty statement admitting no current status is supplied. A generic status-keyword match somewhere in a long citation is insufficient. CLE likewise infers Garland availability from performance.
- **False chronology:** CLE page36 conditional-record table produces a purported back-to-back with Dec23 before Dec21 and a Philadelphia loss, instead of a faithful reading of row headers and dates. It passes mechanics; it must not reach a final report.
- **Future workload used as present fatigue:** CLE page42 March schedule and DEN page1 March/April schedule are used as recent workload for December/January forecasts. The schedules themselves are pregame-known, but that temporal interpretation is wrong.
- **Invented year:** CLE page6 2024 season milestones acquire 2023 fact dates. An explicit date that is merely before cutoff is not necessarily source-supported.
- **Interpretation overreach:** DET “30 points, five rebounds, five assists” is described as a triple-double. Other claims infer no fatigue, confirmed health or stable workload from participation/performance alone.

The first-page CLE truncation is request `e394bcf322f921238f307b05136a245f998d03f1b9a531c319eccd4b8a6d9baa`, page1/chunk1, prompt5,371 tokens and output3,000. It repeats facts across roles, assigns unrelated decoded-table IDs to prose, omits continuation citations for LeVert, and calls Merrill bench production despite the start. The bounded 6,000/8,192 repair budget addresses truncation only; these semantic/citation defects still need repair/review. The partial output is not admitted or synthetically completed.

Complete page processing is useful, but it is not coverage of salient evidence. The next complete-report audit must compare final modules against these source expectations, inspect retained model interpretations and rejections, and preserve missing opponent/status limitations. No training-quality claim follows from the larger mechanically retained count.
