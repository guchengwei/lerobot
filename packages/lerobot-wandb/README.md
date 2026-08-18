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
logs browser-playable H.264/yuv420p MP4 Run Media. A raw `.mp4` visible under the Artifact **Files**
tab is only an Artifact file; W&B media playback comes from the explicit `wandb.Video` logged on the
Run. Use `--no-preview` to keep the previous Artifact-only behavior.

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
episode is logged as Run media under `dataset_video/episode_<six-digit-index>/<camera>`, with camera
names reversibly escaped. v2.1 already stores one file per episode and camera. For v3, where one video
file can span several episodes, the command uses the dataset metadata timestamps to create an
H.264/yuv420p derivative containing only the selected episode. The canonical source file in the
Artifact is never re-encoded or replaced.

To review every episode explicitly, use the bounded all-episode mode:

```bash
lerobot-wandb dataset upload \
  --root ./pick-cube \
  --entity my-team --project so101-pick-cube --name pick-cube \
  --preview-all
```

`--preview-all` and `--preview-episode` are mutually exclusive. All-episode publication defaults to
a maximum of 50 episodes; a larger dataset is refused before a W&B Run is created. Raising
`--preview-max-episodes N` is an explicit opt-in to the additional encoding, storage, and upload
cost. Without either selector, episode 0 is selected deterministically for every camera and logged
under `dataset_video/representative/<camera>` for both v2.1 and v3. The selected episode indices,
dataset schema version, and requested/resolved Artifact references are written to the upload Run
summary. Every prepared batch is also bounded by the smaller of 250 MiB and 20% of the canonical
dataset directory; exceeding that hard budget fails before `wandb.init` with suggestions to select
fewer episodes or use `--no-preview`. The fixed profile is at most 640 pixels wide, 15 fps, CRF 32,
and a two-second GOP; quality is never silently lowered. Use `--no-preview` to generate and publish
no review derivatives.

## Fork-only features

Training directly from `dataset.artifact_ref` inside `lerobot-train` and final model
publication from the training lifecycle are fork-specific integrations, not portable
companion behavior.
