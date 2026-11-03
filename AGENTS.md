# Project working instructions

- This is Tommaso Sini and Anthony Viverito's Stanford CS230 NBA forecasting project.
- Read `HANDOFF.md`, `README.md`, and `docs/experiments.md` before continuing.
  Long agent chronology lives in `docs/archive/` — do not treat it as current status.
- Use the repository's `.venv` for all Python work: `.venv/bin/python` and
  `uv pip install --python .venv/bin/python -r requirements.txt` (or the venv's pip).
  Do not install packages into system Python or the global Conda environment.
- The canonical repository is `/Users/tommasosini/Desktop/Repos/cs230-nba-forecasting`.
  Data under `data/` is local and git-ignored. If Superset creates an isolated
  worktree, the downloaded dataset and `.venv` will not automatically be there.
  Prefer the local checkout for this data-collection phase; otherwise explicitly
  arrange dataset access and create a separate venv in the worktree.
- Preserve raw sources and provenance. Never feed raw ESPN responses to the model:
  they contain postgame results, recaps, and current news.
- Keep outcomes separate from pregame inputs. Enforce source timestamps at or
  before each game's prediction cutoff. Unverified narratives stay separate.
- Do not infer full-season or leaguewide narrative coverage from the current
  early-November sample or the Cavaliers-only note adapter.
- After collector changes, run leakage tests and the pilot validator. A cached
  rebuild is available with `scripts/collect_pilot.py --offline`.

