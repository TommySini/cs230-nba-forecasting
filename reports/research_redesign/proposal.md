# Proposed research and forecasting redesign

Design discussion, November 1, 2026. This is a proposal, not an implemented or evaluated model. The completed full-note experiment and its frozen artifacts remain unchanged.

## Recommendation

Keep the original gradient booster as the statistical forecast. Directly train a research model to produce grounded, useful module reports using reviewed hindsight-guided targets. Then train the downstream probability mapping on reports that the research model actually generates. Improving research is a first-class training objective, not merely an auxiliary explanation loss on the final predictor.

The relevant distinction in [the Balyasny paper](https://arxiv.org/html/2607.09921v1) is between hindsight-guided report targets (§4.2.1) and market-smoothed outcome targets (§4.2.2). It fine-tunes the forecasting LLM; the specialist research agents are described as prompted GPT-4.1/GPT-5 agents (§4.1.1). Direct research-module fine-tuning and the numerical head below are deliberate NBA adaptations.

## Terms and responsibilities

| Component | Receives | Produces | Proposed learning signal |
| --- | --- | --- | --- |
| Research model | Archived pregame sources and a module instruction | Cited findings, basketball interpretation, counterevidence and unknowns | Reviewed gold module reports |
| Report reader / encoder | Generated research brief and booster probability | A numerical representation of that input | Initially frozen for a controlled comparison; forecasting LoRA is the next comparison |
| Probability head | Reader representation and booster probability | One correction and the final home-win probability | Actual game outcomes |
| Forecaster | Name for the report reader and probability head together | Final probability, optionally a supporting explanation | The downstream forecasting objectives |

The complete pipeline additionally contains the booster and research model. Fine-tuning the reader and head together means changing both the language representation and its final numerical mapping. It does not mean combining two independent win predictions. Ordinary training through a reader/head does not propagate gradients through separately generated research text into the research model; that model needs its own supervision.

## What the completed experiment establishes

The current system selects source passages and computes supported table facts, encodes them with a frozen Qwen2.5-14B model, and fits a 5,121-parameter linear correction. It does not train the language model to write better basketball analysis. Test Brier was 0.186296 for boosting and 0.193206 for the hybrid, on 14 repeatedly inspected games. This is a negative, inconclusive pilot rather than a verdict on a trained research forecaster.

Our [quality audit](../game_note_research/quality_audit.md) found semantic errors after same-model checking, including a 32B trial confusing field goals with points. Rejected candidate counts are not factual-error rates. Both false acceptance and false rejection matter. Source layout, season boundaries, player identity, citation completeness, and current versus historical availability need explicit checking.

## Proposed prediction path

`Archived pregame sources → trained specialist research calls → cited module reports + booster probability → report reader → probability head → final probability`

1. Process every PDF page while preserving columns, headings, season labels and table cells. Compute numerical summaries from validated structured rows; retain unknown or undecoded content in the audit. Full source coverage does not require copying every repetitive statistic into the final brief.
2. Use three specialist roles initially: availability/rotation; matchup strengths and weaknesses; recent changes/rest/workload/travel. Call one shared research model and shared research LoRA adapter with different module instructions. Separate specialist calls do not require separate large models or separate adapters. Retrieve both teams' relevant evidence where available and explicitly record one-sided coverage. Each finding separates the observed fact, its date, its citation, the basketball interpretation, counterevidence and uncertainty. Identify whether a fact is already represented in the booster features.
3. A reviewer checks evidence against source passages, repairs incomplete citations and requests missing context. Unsupported claims remain visible for correction but do not become accepted facts. A different model provides an additional check, not factual certification. A human-scored reference set remains necessary.
4. Assemble the module outputs deterministically, preserving citations, into a compact research brief explaining plausible advantages for each side and unresolved questions. Explicitly distinguish an evidence-backed interpretation from a measured numerical effect; do not invent percentage-point adjustments. A generative synthesis stage is optional later and would itself need coverage and grounding evaluation.
5. Give that brief and the original booster probability to the forecaster. Its probability head predicts a log-odds correction, initialized to zero:

   `p_hybrid = sigmoid(logit(p_boost) + delta_theta(pregame_brief, p_boost))`.

The research reports are inspectable analysis even before adding a final forecast explanation. If the forecaster also writes an explanation, that output is not proof that each sentence caused its probability adjustment. Evidence-removal checks can test sensitivity separately.

## Hindsight supervision

Split games chronologically before constructing supervision. First make an outcome-blind pass over each training game's pregame sources to enumerate evidence and competing hypotheses under fixed module rubrics. Then a postmortem teacher can read the completed training game, box score and recap. It distinguishes factors foreseeable before tipoff from events or variation that could not have been known then and flags evidence worth revisiting. A final pass writes an ideal pregame report for each module using only supportable pre-cutoff facts, preserving opposing evidence and uncertainty. The postmortem guides attention; its narrative is kept separately from the module targets.

For example, a documented missing rim protector may warrant examining the opponent's interior offense. An unexpected shooting streak during the game does not establish that a pregame analyst should have predicted the winner. Include counterevidence and uncertainty; an upset is compatible with a well-calibrated pregame probability.

These gold module reports are proposed reference targets, not automatically correct reports. Review a sample for citation support, temporal scope, outcome-revealing language and omitted counterarguments before using them for training. Gold report selection can itself carry hindsight even when every quoted fact predates the game. Multiple reports from one game enrich language supervision but remain one independent game outcome.

For research training, the input is the archived pregame source packet and module instruction; the output target is the corresponding gold module report. Supervised token loss trains the shared research adapter to select and interpret relevant evidence. At prediction time it receives only pregame sources and generates its own report. Dev/test reports must come from this outcome-blind process; no gold reports or postmortems enter downstream inputs. Check knowledge cutoffs for all models participating in held-out prediction, including research and verification models. A historical-date instruction alone cannot remove memorized outcomes.

## What gets trained

Keep the research adapter separate from any forecasting adapter, even if they share the same base checkpoint. Train research LoRA on module reports first. To train the downstream forecaster, generate reports for each training block using a researcher fitted only on earlier training blocks. Use the base prompted researcher for any initial block without enough earlier supervision. Keep every packet, module and augmentation from a game together. Random cross-fitting is insufficiently strict here because later notes can explicitly describe earlier games' results.

Pool these chronological, out-of-fold student reports and fit the probability mapping against actual outcomes. Do not feed hindsight gold reports directly to the predictor or train it on reports generated by a student fitted to that same game's gold target. Teacher-gold input performance, if measured, is a labeled privileged-information diagnostic rather than the headline result. Fit the final researcher on the allowed training pool before generating dev/test reports; acknowledge and evaluate the change from fold-specific researchers to this more-trained final researcher.

First compare prompted versus trained research under the same frozen-reader and regularized-head recipe. Preserve that recipe's binary log loss plus L2 and dev-Brier selection so the comparison isolates research improvement. Then compare a forecasting LoRA adapter plus head against the frozen-reader control. This second comparison tests whether adapting the report representation helps. Neither stage presumes that more trainable weights must improve generalization.

For the adapted forecaster, a candidate objective combines token cross-entropy for an optional gold forecast explanation with a development-selected weight on Brier loss against the actual binary home-win outcome. Apply the probability head to the hidden state after the pregame input, **before** any teacher-forced gold explanation tokens. Both losses update the forecasting adapter; only the probability loss trains the numerical head. If the head instead reads generated explanation text, use outcome-blind out-of-fold generated explanations during fitting. Training booster probabilities must likewise come from prior-only fitting rather than an in-sample booster fit that saw that game's outcome.

Use actual outcome labels and ordinary, unweighted Brier evaluation for the primary experiment. Do not initially oversample upsets or winning classes. Market-smoothed supervision should be a separate ablation: it adds another information source during training and would complicate attribution of a gain to game-note text. Any market supervision needs explicit timing/provenance and a matched control receiving the same supervision.

## Evidence needed before scaling

First build a reference set of roughly 20–30 existing training games, covering different publishers and difficult layouts. Compare teacher candidates using the same sources. Score citation correctness, entity/date/season correctness, current availability, numerical accuracy, coverage of relevant evidence and quality of the supported interpretations. Correct omissions as well as hallucinations. Selecting fewer claims alone must not count as better research. Select model size using this evidence rather than a parameter-count rule: the current 14B is a starting candidate and the previously tested 32B is not presumed superior. A newer teacher may build training targets without serving in historical held-out prediction.

Then expand to substantially more distinct games and teams before claiming reliable fine-tuning gains. Thousands of PDF pages or generated claims do not increase the number of independent game outcomes; the completed run had only 80 training games and 13 development games. Limit the development search accordingly.

Keep boosting and boosting-plus-research as the two headline systems. Use small diagnostic ablations for prompted versus trained research, outcome-blind versus hindsight-guided report targets, frozen versus adapted report reading, and a matched correction model with the research brief removed. These diagnose distinct hypotheses and should be staged rather than expanded into a large search on 13 development games. Evaluate the selected recipe on identical, previously untouched chronological games using Brier difference and paired uncertainty estimates, with calibration and log loss as secondary diagnostics. Research quality and forecasting improvement are separate success criteria. The already-inspected 14 games remain exploratory and must not serve as a new final test.

## Follow-up: older training games and memorization safeguards

The user reported a preliminary Qwen2.5-14B recall probe: 1/400 exact scores before the checkpoint versus 0/92 after; upset recall 42% versus 44%; home-win AUC 0.55 versus 0.58. These are user-supplied aggregate results, not a probe independently rerun or validated in this worktree. They support little detected direct recall under that prompt, not a certification that outcomes cannot be accessed through other prompts, richer context, fine-tuning, or hidden-state features. The cohorts have unequal sizes and no supplied uncertainty estimates. The single exact score cannot be classified as a guess from the aggregate results alone. Full-season records are a retrospective diagnostic comparator, not a permissible pregame feature.

This does not require excluding older games from **training**. Their outcomes are already legitimate supervision; any reliance on historical memory is primarily a generalization concern when evaluation is genuinely unseen. Keep dev and fresh final test beyond the pretraining information available to every prediction-time model, with no dev/test postmortems or outcomes entering fitting. The hindsight teacher may use postgame information only for training targets.

A read-only eligibility screen of the existing approved notes and saved numerical inputs found **91 of 149** approved 2023–24 games meet the current booster-history policy. Thirteen lack the baseline's required five prior games for both teams; 45 more lack 200 eligible fitting examples before the start of their calendar month. Eligible notes span January 1–April 9, 2024: 39 in January, 15 in February, 21 in March and 16 in April. Adding all 91 to the previous 80 training examples would yield **171**, subject to full-note preparation and validation. No new forecasts have been fitted and no frozen cohort has been changed. This screen uses `data/processed/gradient_boosting/inputs.jsonl`, `data/processed/game_notes_collection/approved_excerpts.jsonl` and the policy in `config/boosting_notes.json`; see `older_training_eligibility.json` for source hashes and per-game eligibility.

A second read-only audit reproduced these counts and hashes and verified the 91 raw PDFs and their archive timestamps. Existing content approval covers page 1; it does not establish full-packet extraction or research quality. Eighty-four of the eligible packets are from Detroit, Denver, Memphis or Boston, requiring layout/table checks beyond the current Miami/Cleveland adapters. The other 58 older games may still provide research-only supervision after validation, because research training does not require a booster offset. Keep this larger research pool chronological when generating out-of-fold reports.

Implementation must use a new versioned configuration and artifact directory. The existing note preparer excludes dates before September 2024 and the research preparer rejects every pre-checkpoint game, including training; both need an explicitly scoped training-only policy for the new experiment. Preserve evaluation cutoff checks and all completed pilot artifacts.

Two suggested safeguards are useful experiments but require narrower claims:

- **Anonymization:** removing names and absolute dates removes direct identifiers. Distinctive statistics, roster descriptions, schedules and source wording can still reveal identity. A consistent within-game player mapping, relative dates and preservation of roles/units make a useful ablation; also test whether the game or team can still be reidentified. Fit and evaluate the anonymized arm consistently on post-checkpoint games to measure utility loss without ordinary historical-memory exposure. Do not assume lower performance proves memory was removed. [Anonymization and Information Loss](https://arxiv.org/abs/2511.15364) documents this utility issue in financial text, not NBA notes.
- **Reader-only interface:** the existing pilot is not such an isolation boundary. Quoted source text is fed through Qwen, and its hidden states feed the learned head. An explicit-field variant could instead pass only validated, fixed-schema values to a separately trained numerical forecaster, with neither raw LM hidden states nor a second LM encoder downstream. This removes that direct representation path, but outcome-informed selection, omissions, classifications or ordering can still leak through the extracted fields. Predefine requested fields, canonicalize ordering and unknowns, audit coverage, and keep prediction-time extraction outcome-blind. Citation string matching verifies copying, not meaning, salience or temporal scope.

Keep the BAM-inspired research/LoRA forecaster as the proposed main architecture. Use anonymization and the structured-field variant as diagnostic ablations rather than replacing the goal with a restricted table-only system. Extend the recall audit with different prompts and pregame-context cues; on separate chronological diagnostic subsets, compare outcome predictability from identity-only hidden states before and after the checkpoint. Such probes detect some failure modes but do not prove their absence. [Probabilistic extraction research](https://arxiv.org/abs/2410.19482) likewise shows why a single extraction query is an incomplete memorization measure.
