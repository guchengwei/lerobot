# lerobot-wandb

Companion distribution for moving LeRobot datasets, models, and rollouts to/from
W&B Artifacts. Installs side-by-side with an existing `lerobot` environment without
replacing or shadowing it.

```bash
# Install from this repository (the distribution is not yet published to PyPI):
pip install "lerobot-wandb @ git+https://github.com/guchengwei/lerobot.git@main#subdirectory=packages/lerobot-wandb"

# Fresh environment: also installs a compatible LeRobot:
pip install "lerobot-wandb[lerobot] @ git+https://github.com/guchengwei/lerobot.git@main#subdirectory=packages/lerobot-wandb"
```

Once the distribution is published to PyPI (see the pending publishing tickets), the
install command shortens to `pip install lerobot-wandb` (or `pip install 'lerobot-wandb[lerobot]'`
for a fresh environment).

`lerobot-wandb` never installs files into the `lerobot` package namespace, and the
base distribution does not hard-depend on `lerobot`: commands that need LeRobot
validate the installed version at runtime and fail with an actionable message when
it is absent or unsupported.

## Development installation

The companion is built and installed from this repository. Supported editable
commands:

```bash
# Whole fork (root project + companion), from the repository root:
uv sync --locked --extra dataset --extra training --extra test

# Companion subproject alone, into the active environment:
uv pip install -e packages/lerobot-wandb
```

The fork's `training` extra depends on the companion via a uv path source, so the
whole-fork command installs exactly one `lerobot-wandb` executable (from the
companion) and never a second one from the fork's own distribution.

## Portable companion features

- dataset/model Artifact upload and download
- model promotion and Registry links
- rollout publication and browser-playable previews
- workspace creation (`lerobot-wandb[wandb-workspace]`)

## Fork-only features

Training directly from `dataset.artifact_ref` inside `lerobot-train` and final model
publication from the training lifecycle are fork-specific integrations, not portable
companion behavior.
