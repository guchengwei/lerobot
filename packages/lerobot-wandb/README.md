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

## Dataset video review and v2.1 transfer

`dataset upload` keeps the source directory byte-for-byte as the canonical Artifact and separately
logs one browser-playable H.264/yuv420p video to the upload Run when video exists. A raw `.mp4`
visible under the Artifact **Files** tab is only an Artifact file; W&B media playback comes from the
explicit `wandb.Video` logged on the Run. Use `--no-preview` to keep the previous Artifact-only
behavior.

The transfer path accepts both current LeRobot v3 datasets and canonical v2.1 datasets. v2.1 is
validated against its episode-per-file layout without asking the current v3 reader to load it, so a
GR00T N1.5 copy can be stored and downloaded unchanged even though this fork's current LeRobot
training path still requires v3.

For v2.1, exact episode review is available because each episode owns its video file:

```bash
lerobot-wandb dataset upload \
  --root ./pick-cube-v21 \
  --entity my-team --project so101-pick-cube --name pick-cube-v21 \
  --preview-episode 10
```

Repeat `--preview-episode` to publish more review episodes. Every camera video for each requested
v2.1 episode is logged as Run media under an episode-labelled key. Without the flag, only one
deterministic representative video is logged, so the upload does not duplicate the full dataset as
media.

v3 video files can span multiple episodes, so `--preview-episode` is deliberately refused for v3
rather than labeling a shared chunk as one episode. Omit the flag for one representative v3 chunk,
or review the episode-per-file v2.1 copy when exact episode selection is required.

## Fork-only features

Training directly from `dataset.artifact_ref` inside `lerobot-train` and final model
publication from the training lifecycle are fork-specific integrations, not portable
companion behavior.
