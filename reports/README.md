# Research reports

Start with [Data and provenance](data.md), then [Statistical and market
baselines](baselines.md), then [Text experiments and lessons](text_experiments.md).
Together they explain what the project can measure, the benchmark text must beat,
and what the completed pilots actually established.

Exact results are in the six JSON files in [`results/`](../results/). Frozen
datasets and their manifests remain in the canonical checkout's git-ignored
`data/`; linked local data files require that directory. Design lives in
[`docs/bam/`](../docs/bam/), operational setup in
[`docs/aws_toolkit_setup.md`](../docs/aws_toolkit_setup.md), and next steps in
[`HANDOFF.md`](../HANDOFF.md).

Maintain one report per research question. Explain the finding, method, evidence
and limitations; name the cohort behind every comparison. Update existing prose
and remove what it supersedes. Keep machine output out of this folder, and retain
supporting files only when code needs them or they preserve essential unique
results. Historical details can be recovered from the Git revisions cited in the
reports. Collectors and evaluators never write these narratives.
