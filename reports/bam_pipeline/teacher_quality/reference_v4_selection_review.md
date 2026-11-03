# Reference v4 selection review

Selection completed for all 136 source chunks in `bam_reference_v4_20261102`. This review concerns selected evidence only. No module or gold report is approved by this document.

| Game | Chunks | Retained findings | Canonical table findings | Retained rejected proposals |
|---|---:|---:|---:|---:|
| MIA, espn:401704882 | 15 | 49 | 27 | 5 |
| DET, espn:401705416 | 46 | 89 | 7 | 60 |
| DEN, espn:401585272 | 23 | 37 | 9 | 37 |
| CLE, espn:401704990 | 52 | 77 | 9 | 86 |

Canonical table injection supplies useful Miami evidence: Adebayo's latest 38:36 minutes and 13 points, Herro's 36:49, Butler's 34:25, Richardson's prior-game NWT status, and Highsmith's latest 27:01. Explicit page-four prose correctly assigns the 52 regular-season and 57 combined regular/postseason made-three streaks to Herro. The earlier neighboring-player-card trap is therefore avoidable with the available evidence.

Eight manually checked defects remain in final selected proposals: four Detroit season totals called averages (including PITP mislabeled as transition), Cunningham's inclusion thresholds called exact averages, Cleveland's seventh-ranked defense called second, Miami's 109.9 opponent PPG called higher than 110.8, and the Miami Boston-visit record missing its sentence continuation in the original citation. Exact artifact IDs, candidate IDs and citations are recorded in `reference_v4_initial_selection_issues.json`. Mechanical acceptance did not establish their correctness; downstream review must correct or reject these claims.

Source-context coverage is better than a finding-only inspection suggests. Expanded citation neighborhoods expose Denver's latest Murray/Porter/Jokic rows and Detroit's Brown/Kornet countercontext and current back-to-back schedule even without corresponding selected summaries. However, five important Cleveland checklist subjects have no reference lines in the final context union: the external injury-report limitation, four-game road trip, prior Denver meeting, recent three-point windows, and LeVert's eight-of-ten scoring stretch. A reviewer receiving only these module inputs cannot assess those omitted passages. `reference_v4_final_selection_context_coverage.json` records exact line overlap for all 40 checks; line presence itself is not semantic coverage.

The three truncated initial responses completed with expanded output budgets. Among 63 other chains whose first repair was byte-identical to the initial response, 36 second repairs remained byte-identical and 27 changed text. No previously retained fact string was lost. Only one new fact string was retained: a statement that a future February schedule contains no current/recent game information. Six chains reduced rejected proposals, mostly by removing them. This supports stopping repeated unchanged partial selections early, while retaining raw proposals and rejection reasons; it does not prove all repairs are useless. `reference_v4_second_repair_utility.json` preserves each chain.

An initial concern about repair deleting Ivey's current status was a reading error: the status was in Detroit page-one chunk zero, while the inspected empty repair was chunk one. Exact parent comparisons found no such deletion. The diagnostic is explicitly withdrawn in `reference_v4_selection_repair_evidence_loss.json`.

Next gate: inspect actual module synthesis, independent review decisions and final gold reports against the source checklist. Do not infer end-to-end research quality from chunk completion or mechanical validity.
