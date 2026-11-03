# Independent integrated resume review

Read-only review of the integrated worker, checkpoint helper and cloud wrapper found no new blocking issue. This is code review plus local tests, not proof of exact CUDA replay.

Reviewed SHA256 values:

- `scripts/bam_training.py`: `36c0095db4e44e6dd36bcceb4140dad8d281d0dedc50166b61a62fef71fe96c8`
- `scripts/bam_checkpoint.py`: `106fd62fc78fb5fcdfa12d6a08f264271c3c68de91766b920c415f76961f8c2d`
- `scripts/bam_cloud.py`: `57f3adc7c8b849577bc59fd0fd6b4ccd1b99ae65a286dc5db62c925d3b4fee8b`

The checkpoint helper binds optimizer order to live named tensors, addressing the previously reproduced same-shaped reversed-parameter bug. It rejects changed CPU/CUDA topology, validates all saved hashes and identities before loading, and restores shuffle, Python, CPU and CUDA RNG after model/optimizer construction. Immutable checkpoint directories become visible before an atomic latest-pointer replacement; a pre-pointer crash can leave an orphan without corrupting the previous pointer. Cloud commits remain a separate operation and occur on completed-artifact log markers.

Research recovery restores the exact epoch permutation and completed accumulation-group offset. Forecast recovery includes the probability head, training labels, dev labels and dev row identity. The stored history excludes the current epoch until its immutable validation record is replayed. A crash after validation but before the next checkpoint can reuse that record only when the adapter/head state and complete identity match. Final selected adapter/head copying occurs after all fixed epochs.

The Brier forward sees the pregame prefix; the explanation target enters a separate completion-only CE forward. Separate backward calls accumulate the same weighted objectives before one clip and optimizer update. This reasoning depends on the actual CUDA execution preserving the intended dropout/checkpoint RNG sequence; the prepared interrupted-versus-uninterrupted GPU smoke remains required before claiming exact numerical replay.

The research generation ladder retains raw attempts and successful siblings, retries only declared larger output budgets with the same adapter and source identity, and rejects context overflow without truncating the input. Runtime errors remain recorded and terminal for that immutable attempt. The cloud wrapper binds deployed worker/helper hashes, uses a six-hour function cap with an earlier worker budget and watchdog, and commits on completion markers or elapsed log intervals.

Validation: `.venv/bin/python -m unittest tests.test_bam_training tests.test_bam_cloud tests.test_bam_checkpoint -q` passed 54 tests. No remote job was launched for this review.

Subsequent evidence, 2026-11-02: the parent-run H100 smoke completed under the same worker/helper hashes. `data/probes/bam_cloud/20261102_resume_smoke_v4/artifacts/synthetic_resume_smoke/result.json` records exact equality for all 1,538 research and 1,546 forecast state tensors after six optimizer updates, with zero maximum difference and matching RNG state. The accompanying `independent_replay_audit.json` verifies matching adapter/optimizer/RNG file hashes, epoch cursors, permutations and dev-selected epoch-two provenance. This resolves the pending empirical replay check for that synthetic six-update configuration. It does not establish forecasting performance or universal determinism across longer runs and hardware configurations.
