# W&B companion with LeRobot (SO-101) — legacy fork walkthrough

![Broader LeRobot × W&B ecosystem overview (not the companion contract)](./assets/wandb-workflow-overview-en.jpg)

This image is a broader LeRobot × W&B ecosystem overview, not the companion capability contract.
Labels such as **Auto-Upload**, **W&B SDK (Streaming)**, **Deploy / Inference**,
**Closed-Loop Control**, and “all data, models, and results are stored in your private W&B
workspace” describe upstream optional settings, historical fork hooks, or external deployment
context. The companion contract covers explicit Artifact transfer and promotion around ordinary
LeRobot commands. `lerobot-record`, `lerobot-train`, and `lerobot-rollout` remain LeRobot commands.
The companion does not automatically record data, stream every run, deploy a model, or save
everything to W&B. It does not take over the robot control loop.

[English] · [日本語マニュアル](./README.ja.md)

Record on a real robot, publish the dataset, train from that exact dataset version, publish the
trained policy, roll it out on the robot, and publish the rollout with a lineage edge back to the
model that produced it, with W&B as the only remote store.

> [!NOTE]
> **Legacy fork walkthrough.** The current companion manuals are the [English manual](https://github.com/guchengwei/lerobot-wandb/blob/main/MANUAL.md)
> and [Japanese manual](https://github.com/guchengwei/lerobot-wandb/blob/main/MANUAL.ja.md). This
> file remains for the fork-only training path and its surrounding LeRobot commands.
> `lerobot-wandb` is a W&B companion/integration that runs with ordinary upstream LeRobot. It is
> an independent package and release source, not a native LeRobot plugin contract or a
> self-contained product.
> PyPI publication is not available yet; install the companion from the canonical source with
> `pip install "lerobot-wandb @ git+https://github.com/guchengwei/lerobot-wandb.git@main"`.
> The `--dataset.artifact_ref` training path, training-time Artifact materialization, W&B model
> publication fields, and same-run final-model publication shown below are fork-only hooks. The
> `lerobot-record` and `lerobot-rollout` commands remain ordinary LeRobot commands. Use the linked
> canonical manuals for the companion workflow; keep this file for the fork-only path.

The diagram is a conceptual overview. The commands and runtime boundaries below describe this
fork's current end-to-end composition. In particular, W&B is never called from the robot control
loop.

The commands form a tested template. Supply the setup and runtime values explicitly marked in steps
0, 4, and 6, and adjust the hardware ports and camera index described in step 0.
`packages/lerobot-wandb/tests/test_showcase_readme.py` expands the documented values, extracts
the commands, and parses them against the real CLI. It does not exercise a live W&B workspace or
robot hardware.

## Pipeline

```mermaid
flowchart LR
    R[lerobot-record<br/>local dataset] -->|dataset upload| DA[(dataset Artifact)]
    DA -->|--dataset.artifact_ref| T[lerobot-train]
    T -->|final policy| MA[(model Artifact)]
    MA -.->|Registry link| REG[[Registry collection]]
    MA -->|model download| P[local policy directory]
    P -->|--policy.path| RO[lerobot-rollout<br/>on the robot]
    RO -->|rollout upload| RA[(rollout Artifact)]
    MA -->|lineage only| RA
```

Solid arrows move bytes. The Registry link and the policy-to-rollout lineage edge do not.

The `--dataset.artifact_ref` edge and final-model publication are fork-only training hooks.
`lerobot-record` and `lerobot-rollout` are ordinary upstream LeRobot commands; the companion adds
Artifact transfer around them.

## What this example is and is not

Read this before the commands; it is the part that keeps you from being misled.

- **W&B is the only remote store here.** Nothing in this example pushes to the Hugging Face Hub.
  `lerobot-wandb` never touches the Hub.
- **Local disk stays the runtime cache and recording buffer.** Artifact downloads are materialized
  to disk before anything reads them. In this fork, the `--dataset.artifact_ref` training hook also
  materializes the dataset under `output_dir`. W&B is a durable store for finished artifacts, not a
  filesystem the robot reaches through.
- **No W&B call happens inside the robot control loop.** Publishing is a separate step you run after
  recording or rollout, with the robot disconnected.
- **Aliases are mutable; versions are immutable.** `latest` and `candidate` can move; `v3` cannot.
- **Training records the immutable version it actually trained on.** You may pass a mutable alias;
  the Run records the resolved `vN`, so it remains reproducible after the alias moves.
- **A candidate or Registry link is not production approval.** Applying `production` to the exact
  evaluated version is the deliberate promotion step.
- **Rollout success counts are supplied by the operator.** Nothing here scores physical success
  automatically; pass the count from your own judgement.

## 0. Prerequisites

Run this from the root of a clone of this fork. Replace `your-wandb-entity` with the W&B entity that
owns the project.

The worked commands target Linux with a Bash-compatible shell. On Windows PowerShell, activate the
environment with `.venv\Scripts\Activate.ps1` and replace `/dev/ttyACM*` device paths with the
corresponding `COM` ports. The remaining CLI arguments are the same.

### 0.1 Fork development environment (legacy manual path)

The fork's training integration (`lerobot-train --dataset.artifact_ref`, final-model
publication) requires the fork itself. Its `training` extra installs the companion
`lerobot-wandb` distribution automatically:

```bash
uv sync --locked --extra core_scripts --extra feetech --extra training
source .venv/bin/activate
wandb login
export WANDB_ENTITY="your-wandb-entity"
export WANDB_PROJECT="so101-pick-cube"
```

`core_scripts` installs the dataset and hardware stacks, `feetech` the SO-101 motor bus, and
`training` both `wandb` and `accelerate` plus the `lerobot-wandb` companion. The commands below
assume an SO-101 follower on `/dev/ttyACM0`, a leader on `/dev/ttyACM1`, and an OpenCV camera at
index `0`; adjust them to your hardware.

### 0.2 Existing LeRobot environment (companion install)

`lerobot-wandb` runs alongside an already-installed ordinary upstream LeRobot (or a compatible
fork). It does not replace or shadow LeRobot and never installs files into the `lerobot` namespace.
PyPI publication is not available yet, so install the companion from its canonical source:

```bash
# Existing LeRobot environment:
pip install "lerobot-wandb @ git+https://github.com/guchengwei/lerobot-wandb.git@main"

# Fresh environment with a compatible LeRobot:
pip install "lerobot-wandb[lerobot] @ git+https://github.com/guchengwei/lerobot-wandb.git@main"
```

The install command will change to a PyPI release command after publication.

This manual's fork-only hook is the training step (§3), including same-run final-model publication.
It needs the fork's `training` extra. Dataset/model/rollout Artifact transfer and promotion work with
ordinary upstream LeRobot. Canonical companion commands validate the installed version at startup
(supported range: `>=0.6.1,<0.6.2`) and fail with an
actionable message when it is absent or unsupported (`--allow-unsupported-lerobot` is the
documented experimental override).

## 1. Record a teaching dataset

This is standard LeRobot recording. W&B is not involved yet.

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=local/pick-cube \
  --dataset.root=./data/pick-cube \
  --dataset.single_task="Pick up the cube and place it in the bin" \
  --dataset.num_episodes=30 \
  --dataset.push_to_hub=false
```

## 2. Publish the dataset as an Artifact

The directory is fully validated locally before a W&B Run is created: metadata must parse, Parquet
must match the declared schema, referenced videos must exist, and indices must agree. A malformed
dataset costs you no Run and no upload.

```bash
lerobot-wandb dataset upload \
  --root ./data/pick-cube \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --name pick-cube \
  --alias raw
```

The command prints an immutable resolved reference such as
`your-wandb-entity/so101-pick-cube/pick-cube:v0`. W&B records that resolved version when training
starts, even though the next command requests the mutable `raw` alias.

For episode-by-episode review in the Run Media tab, repeat `--preview-episode N`, or use
`--preview-all` to publish every episode and camera. In v3 datasets, each review item is sliced from
the shared video chunk using the episode timestamps; the Artifact keeps the original video bytes.
All-episode mode is capped at 50 episodes by default and fails before creating a Run when the
dataset is larger. Raise `--preview-max-episodes N` explicitly to accept the extra upload cost, or
use `--no-preview` to publish no review media.

## 3. Train directly from the Artifact (fork-only hook)

This is the fork's training composition, not an upstream companion contract. Exactly one of
`dataset.repo_id` and `dataset.artifact_ref` may be set. The fork materializes the Artifact under
`output_dir` before it builds a dataset object, and the Run records both the requested ref and the
resolved `vN`. The `--wandb.model_artifact_name`, `--wandb.model_artifact_aliases`, and
`--wandb.registered_model_name` fields publish the final model in the same training Run; these W&B
config fields and that final-model publication are also fork-only.

```bash
lerobot-train \
  --dataset.artifact_ref="$WANDB_ENTITY/$WANDB_PROJECT/pick-cube:raw" \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir=outputs/train/act_pick_cube \
  --job_name=act_pick_cube \
  --batch_size=8 \
  --steps=100000 \
  --wandb.enable=true \
  --wandb.project="$WANDB_PROJECT" \
  --wandb.entity="$WANDB_ENTITY" \
  --wandb.model_artifact_name=pick-cube-policy \
  --wandb.model_artifact_aliases='["candidate"]' \
  --wandb.registered_model_name=pick-cube-policy \
  --policy.push_to_hub=false
```

`wandb.model_artifact_name` publishes the final checkpoint as its own versioned model Artifact,
separate from periodic per-checkpoint uploads. `wandb.registered_model_name` additionally links a
deployable final policy into the Registry collection `wandb-registry-model/pick-cube-policy`.

Resuming works without downloading the dataset again. Resume with the saved config, not
`output_dir` alone:

```bash
lerobot-train --resume=true \
  --config_path=outputs/train/act_pick_cube/checkpoints/last/pretrained_model/train_config.json
```

The already-materialized dataset under the original Run's `output_dir` is reused, and its identity
is checked against the sidecar written by the first download before training resumes.

> **PEFT/LoRA:** an adapter-only checkpoint is uploaded but is not linked into the Registry. Its
> base model is read verbatim from `adapter_config.json` at load time and is never rebased on the
> downloaded directory, so the Artifact cannot be rolled out on its own. The refusal reason is
> recorded as `registry_link_refused_reason`. Publish a merged checkpoint for deployment.

## 4. Fetch the trained policy on the robot machine

The command downloads transactionally into a staging directory, checks that the expected policy files
and configuration are present, and only then moves it to `root`. It does not load or execute model
weights. An interrupted or invalid download
does not leave a half-written policy at the destination.

```bash
lerobot-wandb model download \
  --ref "$WANDB_ENTITY/$WANDB_PROJECT/pick-cube-policy:candidate" \
  --root ./policies/pick-cube-candidate
```

Copy the full immutable resolved reference printed by the command into `MODEL_REF`. Replace the
example; do not infer a version number from the alias.

```bash
export MODEL_REF="your-wandb-entity/so101-pick-cube/pick-cube-policy:v0"
```

The resulting directory is usable directly as `policy.path`. `candidate` may move later;
`MODEL_REF` must continue to identify the policy that was actually downloaded and run.

## 5. Roll out on the real robot

This is standard `lerobot-rollout` and is offline with respect to W&B. The `rollout_` prefix on the
dataset name is required by the rollout config.

```bash
lerobot-rollout \
  --strategy.type=episodic \
  --policy.path=./policies/pick-cube-candidate \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader \
  --dataset.repo_id=local/rollout_pick-cube \
  --dataset.root=./data/rollout-pick-cube \
  --dataset.num_episodes=20 \
  --dataset.single_task="Pick up the cube and place it in the bin" \
  --dataset.push_to_hub=false
```

Count successful episodes while the rollout runs. You will supply that number in the next step.

## 6. Publish the rollout with policy lineage

Disconnect the robot first; this step is pure upload. Set `EPISODES_SUCCEEDED` to the observed
count. `14` is only an example for a 20-episode rollout.

```bash
export EPISODES_SUCCEEDED="14"

lerobot-wandb rollout upload \
  --root ./data/rollout-pick-cube \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --name pick-cube-rollout \
  --model-ref "$MODEL_REF" \
  --episodes-succeeded "$EPISODES_SUCCEEDED"
```

Use the immutable `MODEL_REF`, not `candidate`. The ref is resolved again at upload time, so an alias
that moved in between could record a policy the robot never used. Recording the wrong model is worse
than recording nothing because the lineage looks authoritative.

The upload Run declares the model as an **input** — resolved for lineage and never downloaded — and
the rollout as an **output** of type `rollout`, distinct from a training dataset. It logs episode
count, success count and rate, frame count, duration, and requested and resolved model refs. The
complete rollout remains in the Artifact with its **original encoding unchanged** (the default
LeRobot video codec is AV1).

> **About the representative video:** one clip is selected deterministically and, with no extra
> command, transcoded to a browser-compatible H.264/yuv420p preview that is logged automatically as
> run media. That preview is a _display derivative only_ — it is never part of the Artifact, and it
> is not the data anything should train on. In Dataset v3, a single `.mp4` may contain as many
> episodes as fit under the writer's file-size target, so the UI clip can represent an episode
> span. The Run summary records the included episodes under `representative_video_episodes`.

## 7. Promote what worked

Nothing is promoted automatically. Promote the exact immutable version evaluated by the rollout.
Do not re-upload the downloaded directory: `model upload` would create a new version with no lineage
edge to the evaluation, while the rollout remains attached to the version actually tested.

```bash
lerobot-wandb model promote \
  --ref "$MODEL_REF" \
  --alias production \
  --registry-collection pick-cube-policy
```

`model promote` updates the existing version without uploading model bytes. The project alias and
Registry link point to the evaluated version, and the printed digest lets you confirm that the bytes
did not change.

A ref that is not a `model` Artifact is rejected. A version missing the files and configuration
required for a deployable policy is refused a Registry link — for example, a periodic weight-only checkpoint without
`config.json`, or an adapter-only checkpoint whose base model is not bundled. The check uses the
immutable file manifest and requires no download. Omitting `registry-collection` still permits a
project alias for such a version.

The Registry link is attempted before the project alias moves. There is no transaction across the
two server-side writes; this ordering ensures that a failed Registry link leaves the production
alias unchanged rather than pointing it at a version that never reached the Registry.

Whether a rollout justifies promotion remains an operator decision made from the Run. This workflow
does not compute it.

## Where things live afterwards

| Thing                     | Where                                                   |
| ------------------------- | ------------------------------------------------------- |
| Teaching dataset          | `dataset` Artifact, `pick-cube`                         |
| Trained policy            | `model` Artifact, `pick-cube-policy` plus Registry link |
| Rollout episodes          | `rollout` Artifact, `pick-cube-rollout`                 |
| Dataset-to-policy lineage | training Run config and model Artifact metadata         |
| Policy-to-rollout lineage | rollout Run input edge and rollout Artifact metadata    |

## 8. Remove / uninstall the W&B integration

For a direct companion install — including an older install of the embedded companion from this
fork — use the package manager. This is package removal, not workflow-data cleanup:

```bash
python -m pip uninstall lerobot-wandb
```

For a uv-managed environment, use:

```bash
uv pip uninstall lerobot-wandb
```

Removing the companion leaves LeRobot, local datasets, downloaded/materialized models, rollout
directories, training outputs, `.wandb_artifact.json` and other workflow metadata, W&B
credentials/configuration, remote W&B Artifacts/Runs/Registry objects/aliases, and shared Python
dependencies untouched.

If this fork workspace was synced with `--extra training`, do not treat the one-off uv uninstall as
the persistent solution. Stop selecting `training` and sync the non-W&B training components
instead (replace `feetech` with your hardware extra when needed):

```bash
uv sync --locked \
  --extra core_scripts \
  --extra feetech \
  --extra dataset \
  --extra accelerate-dep
```

Verify that the companion is absent and ordinary LeRobot training remains available:

```bash
uv pip show lerobot-wandb
.venv/bin/lerobot-train --help
```

The first command should report no installed `lerobot-wandb`; the second should still work. A later
sync that selects `--extra training` will reinstall the integration by design. The generic `wandb`
library may remain if another selected dependency requires it.
