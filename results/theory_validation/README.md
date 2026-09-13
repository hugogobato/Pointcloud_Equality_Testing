# T10 theory-validation artifacts

Outputs of WP T10 (`experiments/theory_validation.py`), which completes the
Phase-7 local-power fleet (NONPROG0 at n=200/500 and NONPROG1 at n=500 to 300
replicates) and checks the Gaussian-shift predictions. Evidence and review
live in `theory/WP7_local_power.md`, `docs/theory_audit/T10_audit.md` and
`docs/theory_audit/T10_repair_log.md`.

## Validator scope vs the pre-registered judge

`theory_validation.py validate` applies an all-rows criterion: it scores every
directional cell x {DR, unadjusted} row that has a prediction (51 rows on
`t10_full_fleet.json`, 39 on `results/phase7_local_power.json`) and passes iff
no scored row exceeds the 0.06 absolute tolerance.

This is not the pre-registered verdict. The pre-registered rule
(`phase7_local_power.judge`) scores a subset of those prediction gaps (DR
PROG0/PROG1 at n=500, plus selected unadjusted rows) and additionally applies
the four null-cell size checks (PROG1 unadjusted > 0.20; DR and unadjusted
<= 0.08 for PROG0, NONPROG1 and NONPROG0) and the twelve ARE1 rows
(`|dr - unadjusted| <= 0.06` for NONPROG1/NONPROG0). Two consequences matter
when reading the artifacts: 1. the validator can fail on rows the
pre-registered rule ignores, and 2. the validator can pass while the
pre-registered rule fails, because the null-cell size and ARE1 checks are not
part of it.

Both `t10_validator_phase7.json` and `t10_validator_full_fleet.json` therefore
record the pre-registered verdict alongside the validator result in the
`pre_registered_confirmed` field, copied from the input's own `verdict` block.
On both current inputs `passed=false` and `pre_registered_confirmed=false`;
the recorded scientific verdict remains `confirmed=False`, and no T10 claim
depends on the scope split.

## Artifact inventory

1. `t10_shards/*.json.gz`: per-task checkpoints of the new replicates (71
   tasks: NONPROG1@500 reps 190-299 and all of NONPROG0 n=200/500).
2. `t10_full_fleet.json`: full 34-cell fleet with observed rates, MC SEs and
   Gaussian-shift predictions.
3. `t10_validator_phase7.json`, `t10_validator_full_fleet.json`: validator
   self-tests on the partial and full inputs, each with
   `pre_registered_confirmed`.
4. `t10_mismatch_table.md`, `t10_mismatch_diagnostics.json`: gap table with
   the 2-SE flags and the over-tolerance rows.
5. `t10_local_power_full.png`: DR, unadjusted and Gaussian-shift curves.
6. `t10_fleet_timing.json`: wall-time accounting. The 70 entries in `tasks`
   belong to the second invocation (2026-09-10 11:03:26 to 12:38:08); the
   first invocation produced `NONPROG1|500|190-200` at 10:45:10 and is
   recorded under `invocations` and `unrecorded_tasks`
   (`tasks_on_disk=71`, `tasks_recorded=70`).
7. `t10_fleet.log`: stdout log of the second invocation.

The Phase-7 inputs `results/phase7_local_power.json` and
`results/phase7_checkpoint.json` are no longer gitignored (see `.gitignore`),
per the standing rule that experiment results are never gitignored.
