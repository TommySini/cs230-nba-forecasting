# Corrected comparison: boosting plus game notes

The baseline must remain the original gradient-boosting model. The second system
uses that model's probability together with rich pregame game notes. The
statistics-only LM versus injury-text LM pilot (`text_ablation`) did not test
this design; retain it as an exploratory record.

## Exactly two reported models

| Model | Inputs | Prediction |
| --- | --- | --- |
| Original boosting | Existing 18 statistical features | Original home-win probability |
| Boosting plus game-note forecaster | That boosting probability plus reviewed pregame game-note evidence | Adjusted home-win probability |

```text
Prior statistics -> frozen original booster -> p_boost --------> baseline
                                                  |
Verified pregame game notes -> retained evidence --+-> encoder/head -> adjusted probability
```

Correction form: `p_hybrid = sigmoid(logit(p_boost) + correction(evidence))`.
Initialize so the hybrid starts at the boosting prediction.

## Implementations (read this carefully)

| Status | Runner | What it uses | Results |
| --- | --- | --- | --- |
| **Primary (complete)** | `scripts/run_researched_forecaster.py` via `run_research_experiment.py` | All pages of 107 packets; frozen Qwen2.5-14B embeddings; 5,121-parameter linear head; verbatim quotes + code-checked tables | [researched_forecaster/findings.md](../researched_forecaster/findings.md) |
| Historical pilot | `scripts/run_boosting_notes.py` | First-page excerpts only; Qwen2.5-0.5B + LoRA residual head | [boosting_notes/findings.md](../boosting_notes/findings.md) |

Do not treat the first-page 0.5B run as the main hybrid result. The full-packet
research revision is the intended architecture for this cohort.

## Training and evaluation rules

- Keep the original boosting model, parameters, outputs and threshold intact.
- Hybrid training uses out-of-sample boosting probabilities (chronological /
  expanding-window lineage). Never fit the hybrid on a booster that saw that
  game's outcome.
- Select on **dev** only; freeze before **test**. No test IDs in fitting data.
- Compare both systems on the same held-out games. Primary metric: Brier (paired).
- Preserve chronological whole-date splits when the approved note cohort allows.
  Previously inspected games are not a fresh final benchmark after repartitioning.

## Cohort

Schedule-defined collection verified 482 pre-cutoff packet versions covering 404
games; 256 games were approved across 19 publishers. The frozen hybrid cohort is
the **107 games after the September 2024 backbone checkpoint**: 80 train /
13 dev / 14 test (whole dates). The 140-game injury cohort remains separate.

Source gate: replay URL, Memento time, raw hash, game header, season, prior
records. Full PDFs preserved. Test set is small and Cleveland-heavy (11/14).

## Results pointer

Full-packet hybrid: [researched_forecaster/findings.md](../researched_forecaster/findings.md).
First-page pilot: [boosting_notes/findings.md](../boosting_notes/findings.md).
Collection audit: [game_notes_collection/findings.md](../game_notes_collection/findings.md).
Experiment map: [`docs/experiments.md`](../../docs/experiments.md).

References: [BAM architecture](https://arxiv.org/html/2607.09921v1#S4),
[source audit](findings.md), [archive API](https://archive.org/help/wayback_api.php).
