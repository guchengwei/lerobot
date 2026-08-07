# W&B integration package boundary

`WandBLogger` (`src/lerobot/common/wandb_utils.py`) exists and uploads periodic checkpoints to W&B as
`model`-type Artifacts via `log_policy()`, materializes dataset Artifacts before training, and
publishes the final model Artifact at the end of training. It stays in the fork, wired into the
existing training loop (accelerate, multi-rank barriers, checkpoint cadence).

The transfer primitives it calls — and the entire `lerobot-wandb` sidecar CLI — live in a
separately installable **companion distribution** at `packages/lerobot-wandb` (import package
`lerobot_wandb`, console script `lerobot-wandb`), decided in [issue #35]. The companion:

- never installs files into the `lerobot` namespace;
- does not hard-depend on `lerobot` — LeRobot-dependent commands validate the installed version
  at runtime (`compatibility.py`) and fail with an actionable message when LeRobot is absent or
  unsupported;
- imports LeRobot only through the single adapter module `lerobot_wandb/lerobot_adapter.py`
  (dataset metadata/readers, video re-encoding, checkout commit detection);
- owns the `lerobot-wandb` console script and the `wandb-workspaces` optional dependency.

`WandBLogger` imports the companion's public API (`download_artifact`, `upload_directory`,
`validate_dataset_directory`, `inspect_model_directory`, `registry_link_refusal`, sidecar
helpers) with no reverse import: no `lerobot_wandb` module imports fork-only training code.
Installing base LeRobot without the training extra must not require the companion at import time,
so fork-side imports of the companion stay lazy (inside functions or `TYPE_CHECKING`).

The companion base package uses runtime compatibility validation instead of a hard LeRobot
dependency because it is installed into environments that already contain LeRobot (upstream or a
compatible fork): a hard dependency would make the resolver replace or shadow that install,
defeating coexistence.

[issue #35]: https://github.com/guchengwei/lerobot/issues/35
