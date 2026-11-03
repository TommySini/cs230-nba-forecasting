# Actual partial-report review: quality gate fails

The first Qwen3 review batch completed 34 responses. Among the 33 mechanically valid reviews, 187 decisions were supported, three unclear and one unsupported. These counts describe model decisions, not factual accuracy. One review needed repair for an unknown source ID.

Targeted independent checking confirmed eleven false-supported claims or date fields among the twelve deliberately selected checks in `reference_v4_first_semantic_review_adjudication.json`. This is a diagnostic sample of known concerns, not a random accuracy estimate.

- The reviewer correctly rejected Cleveland's claimed second-ranked defense: page seven explicitly says seventh.
- It accepted four Detroit totals as averages. Its own source readings call the numbers totals, but its verdicts support the candidate's averages. The 3,044 PITP claim additionally mislabels paint scoring as transition scoring.
- It accepted Cunningham's 20/6/8 inclusion thresholds as exact averages. The cited prose says “at least,” and the same page explicitly states 25.7 PPG.
- It accepted six Cleveland milestone `fact_date` values in 2023. The milestone prose is yearless; the current-season schedule explicitly places the corresponding dated opponents in 2024. The review reasons addressed milestones but missed their date metadata.

The model-supported labels therefore cannot certify these partial reports. Continue the frozen reference only as diagnosis; no training-quality claim follows from an approved artifact status.

The run subsequently stopped during module repair because request `ac700202397753237c56e141665c77bfb75cf0519e8dc79683902c490f8c2fe6` exceeded its fixed context budget. Before that stop, seven of the eight original sentinel errors occurred in automatically approved artifacts; Cleveland's wrong defensive rank was the exception. The approved MIA matchup also retained the wrong `109.9 > 110.8` comparison and the original Boston-visit citation containing only page-three line seven, omitting line eight's continuation. All six unsupported 2023 milestone dates survived in an approved Cleveland partial. No gold report or training export completed. These thirteen checked defects establish a failed quality gate independently of the context-limit blocker; they are not an estimate of the overall error rate.

## Narrow source-based follow-ups

1. **Dense labeled rows.** The existing `numeric_row` helper does not count comma-formatted integers such as `3,044` and requires six numeric tokens. Detroit's five-value `TOTALS` row bypasses the table gate. A BAM-local recognizer should normalize thousands separators and recognize explicit totals/opponents/averages rows in table context. Unsupported layouts remain audited unknowns until decoded; code must not invent denominators or means. Keep the older frozen decoder unchanged.
2. **Date evidence.** `fact_date` currently has a parse/cutoff check but no source-entailment requirement. Require an explicit cited date or a supported combination of cited month/day and year/season context. If the year is unresolved, repair to null while preserving the useful dated prose. Do not treat the prediction year or previous year as evidence by themselves. Tests should cover calendar-year season boundaries, historical recaps and honestly unspecified years.
3. **Thresholds.** Where the same entity, metric and value tuple is qualified by “at least,” “at most,” or a similar comparator, require that qualifier or independent exact-rate evidence. Match the actual clause and units, not any threshold anywhere in a large citation. Include positive controls with independently supported exact rates; a broad numeric ban would recreate false exclusions.

These are proposed changes for a later coordinated protocol, not edits to the running reference. They address observed general failure classes rather than game-specific blacklists.

## Why the single-claim result did not transfer

The earlier Qwen3 V1 benchmark used one claim per request, with median 4,213.5 input tokens. This actual review batch used a median 11,946 (range 3,063–14,552), with multiple findings, interpretations, counterevidence and a whole-report judgment. The failed sentinel requests contained five to nine findings and 8,461–12,612 input tokens.

All 34 responses stopped normally. They used 758–3,268 output tokens against an 8,596-token combined answer/reasoning budget. Consequently, output exhaustion is not an evidenced cause. The observable failures fit missed qualifiers/metadata and shallow number matching under multiple decisions; context dilution is a hypothesis, not an established causal result.

A bounded diagnostic can isolate actual candidate claims while preserving complete relevant source context, including headers, continuation lines and opposing evidence. Reuse both the actual bad candidates and valid neighboring facts. Compare the current reviewer and any independently selected historically eligible alternative on those identical inputs. Keep whole-report coverage review as a separate requirement; individual support checks cannot certify omitted evidence. Do not create another artificial benchmark or prompt grid.
