# What the full packet adds: Miami at Boston, December 2, 2024

This is a worked evidence review of the shortest packet in the historical training split. I read the extracted text of all 15 pages and visually checked pages 3, 10, 11 and 15. It was authored by the assistant to demonstrate the research stage; it is not a new forecasting result or a completed automated research run. The target outcome was not used.

[Original PDF](</Users/tommasosini/.superset/worktrees/cs230-nba-forecasting/read-then-continue-dataset-09962af6/data/raw/archive_versions/3395d9409a40b03c06437707.pdf>) · [Complete extracted text](</Users/tommasosini/.superset/worktrees/cs230-nba-forecasting/read-then-continue-dataset-09962af6/data/processed/full_game_notes/packets/2024-12-02_MIA-at-BOS_401704882/full_text.txt>) · [Structured evidence and page-review record](</Users/tommasosini/.superset/worktrees/cs230-nba-forecasting/read-then-continue-dataset-09962af6/data/processed/full_game_notes/example_research.json>)

The later pages provide player-level information omitted by the old first-page model. For example, Adebayo played **38:36**, Herro **36:49**, and Butler **34:25** the previous night (pages 10–11). Page 4 establishes the road back-to-back. The booster already knows rest days and back-to-back status; it does not have these player workloads. A fatigue effect is a hypothesis, not a measured probability adjustment.

| Evidence | Source pages | What it contributes / limits |
| --- | --- | --- |
| Miami is on the second night of a road back-to-back after playing in Toronto the previous night. | 4 | The back-to-back flag and rest days are already baseline inputs; the named travel route supplies extra context. Possible workload context; this alone is not new evidence beyond the baseline rest feature. |
| On December 1 at Toronto, Bam Adebayo played 38:36, Tyler Herro 36:49, and Jimmy Butler 34:25. | 10, 11 | Individual player minutes are absent from the baseline 18 features. Combining player minutes with the current back-to-back identifies potential fatigue exposure. The direction and size of any predictive effect remain hypotheses. |
| The packet says Terry Rozier played all but five seconds of each of the preceding four fourth quarters. | 4 | Late-game player usage is absent from the baseline. Evidence about recent closing usage. It does not establish his starting status or guarantee the same role tonight. |
| The packet lists Miami/Boston scoring averages of 111.3/120.7 points per game and three-point percentages of .378/.376. | 3 | Season scoring and shooting profiles are additional numerical inputs; they are not a pure narrative effect. Potential context for contrasting scoring profiles. These are per-game figures, not pace-adjusted offensive ratings or win probabilities. |
| Nikola Jovic and Josh Richardson have NWT entries for December 1 at Toronto in their recent-game logs. | 13, 14 | Player-specific recent participation is absent from the baseline. Preserve the literal source status; these entries do not establish availability for December 2. |
| Dru Smith's season-ending ACL injury description explicitly refers to the previous season and November 22, 2023. | 15 | Background information only; not a current injury flag. Do not turn this sentence into an assertion that Smith is ruled out for this game. |
| The packet gives Spoelstra 760 regular-season wins in one section and a 759-536 regular-season coaching record in another. | 5, 9 | Quality-control finding, not a forecasting feature. Retain the conflict; source verification does not guarantee that all sections were updated consistently. |

## Source quality and missing information

The first-page status report is TBD; a last-game starter list does not confirm the next lineup. Pages 13–14 record prior-game NWT entries without establishing tonight's status. Page 15 describes Dru Smith's injury in the **previous season**, which must not be converted into a current ruled-out flag.

The packet also contains stale or inconsistent figures, extensive personal biographies, and retrospective conditional records. Complete ingestion lets the researcher identify and contextualize these passages; it does not make every sentence useful or reliable.

The numerical tables add statistical information as well as narrative. Any forecasting gain from this report would need to be described accordingly. The report does not assign a win probability or infer a causal effect from small lineup/conditional-record samples.

## Every page accounted for

| Page | Research disposition |
| ---: | --- |
| 1 | Read header, previous-game starters, status-report placeholder, transactions and schedule. Current lineup/availability is not confirmed by these fields. |
| 2 | Read roster and staff directory; useful for entity resolution, not a strong standalone game signal. |
| 3 | Read current-opponent comparison and separate next-opponent section. Do not assign the Lakers section to the Boston matchup. |
| 4 | Read all game notes; retained back-to-back context and closing usage. Distinguished performance windows from franchise milestones. |
| 5 | Read streaks and franchise/coach notes; noted an inconsistent coach win count. Sellout streaks and historical milestones were not treated as validated predictive features. |
| 6 | Read offensive-leader and charges tables; cumulative descriptive context with visually complex columns. |
| 7 | Read conditional win/loss tables and starting-lineup records. These small, retrospective subsets do not establish causal effects or reveal tonight's in-game statistics. |
| 8 | Read historical record lists. The triple-double date (Nov 30) differs from the Nov 29 player log on page 10; unresolved source inconsistency. |
| 9 | Read coach biography and record; retained the win-count conflict against page 5, not a causal coaching advantage. |
| 10 | Read Adebayo, Bryant and Burks profiles/logs; retained player workload and distinguished DNP-CD from an explicit injury. |
| 11 | Read Butler, Christopher and Herro profiles/logs; retained individual workload and distinguished G League entries from current availability. |
| 12 | Read Highsmith, Jaquez and Johnson profiles/logs; recent participation remains historical evidence, not a confirmed current rotation. |
| 13 | Read Jovic, Larsson and Love profiles/logs; retained the prior NWT entry with its date. |
| 14 | Read Richardson, Robinson and Rozier profiles/logs; retained the prior NWT entry and cross-checked recent usage context. |
| 15 | Read Smith and Ware profiles/logs; explicitly classified the ACL sentence as prior-season history. |

## Evidence checks

All supporting quotations in the structured file match their cited page in the frozen PDF extraction and carry the packet SHA-256. This verifies quotation provenance, not every interpretation. Only one example has received this semantic review; the rest of the corpus currently has full extraction and prepared research prompts.
