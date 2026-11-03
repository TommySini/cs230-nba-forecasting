# Experiment map

Status for humans and agents. **Keep historical pilots and their reports** — they
are cited in the README and locked by protocol hashes. Do not delete them to
“clean” the repo.

## Active / primary

| Experiment | Scripts | Report | Notes |
| --- | --- | --- | --- |
| Stats gradient boosting | `run_gradient_boosting.py`, `validate_gradient_boosting.py` | `reports/gradient_boosting/` | League-wide baseline; shared feature code |
| Polymarket vs boosting | `collect_polymarket.py`, `evaluate_polymarket.py`, `validate_polymarket.py` | `reports/polymarket/` | Market ceiling on same seasons |
| Full-note researched hybrid | `prepare_full_game_notes.py`, `run_game_note_research.py`, `run_researched_forecaster.py`, `run_research_experiment.py`, validators | `reports/researched_forecaster/`, `reports/game_note_research/`, `reports/full_game_notes/` | **Main text comparison** (complete). Frozen 14B + linear head on boosting log-odds |
| Modal research (vLLM) | `modal_research_vllm.py`, `research_vllm_worker.py`, `import_modal_research.py`, `convert_research_weights.py` | `reports/game_note_research/compute_options.md` | Deploy once; submit with Python CLI |

## Historical pilots (preserve)

| Experiment | Scripts | Report | Notes |
| --- | --- | --- | --- |
| Dataset pilot (100 games) | `collect_pilot.py`, `validate_pilot.py` | `reports/pilot_findings.md`, `reports/coverage.json` | Injury + schedule feasibility |
| Seasonal audit (+60 games) | same collectors with `config/seasonal.json` | `reports/seasonal/` | Expands history used by boosting |
| Narrative article audit | `audit_narrative_sources.py` | `reports/narrative_sources/` | Quarantine examples; not model input |
| Game-note source audits | `audit_game_notes.py`, `collect_game_notes.py`, `review_game_notes.py` | `reports/game_notes_sources/`, `reports/game_notes_collection/` | Provenance for the 107-game cohort |
| Stats vs injury text (0.5B) | `run_text_ablation.py`, `validate_text_ablation.py` | `reports/text_ablation/` | Exploratory; not the intended BAM-style design |
| Boosting + first-page notes (0.5B) | `run_boosting_notes.py`, `validate_boosting_notes.py` | `reports/boosting_notes/` | First hybrid pilot; superseded by full-packet research |

## Probes / archive (do not use for new production runs)

| Item | Location | Notes |
| --- | --- | --- |
| Old Modal MLX batch app | `scripts/archive/modal_research_batch.py` | Superseded; never restart for the main experiment |
| MLX CUDA Modal benchmark app | `scripts/modal_research.py` | Still imported as shared helpers (`volumes`, `unpack`, …) by vLLM; also holds the old `@app` benchmark. Prefer `modal_research_vllm.py` for jobs |
| Device / transfer probes | `benchmark_research_device.py`, `export_research_compute.py`, `run_modal_encoder.py`, `encode_research_mlx.py` | Benchmark and encoder tooling; not daily drivers |
| Agent chronology | `docs/archive/handoff-through-2026-11-01.md` | Superseded by short `HANDOFF.md` |

## Environments

| Install | When |
| --- | --- |
| `requirements.txt` | PDF / light collection deps |
| `requirements-baseline.txt` | Boosting + Polymarket eval |
| `requirements-text.txt` | 0.5B LoRA pilots |
| `requirements-research.txt` | Local MLX research / embeddings |
| `requirements-research-cuda.txt` | Remote MLX CUDA image (not Mac venv) |
| `requirements-research-vllm.txt` | Modal vLLM image (not Mac venv) |
| `requirements-modal.txt` | Local Modal client only |
