# LeRobot

[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/lerobot.svg)](https://badge.fury.io/py/lerobot)
[![Tests](https://github.com/huggingface/lerobot/actions/workflows/tests.yml/badge.svg)](https://github.com/huggingface/lerobot/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/website?url=https%3A%2F%2Fhuggingface.co%2Fdocs%2Flerobot)](https://huggingface.co/docs/lerobot)
[![Discord](https://img.shields.io/discord/1218717590352785449?logo=discord&logoColor=white)](https://discord.gg/s3KuuzsPFb)
[![Twitter Follow](https://img.shields.io/twitter/follow/lerobot_hf?style=social)](https://x.com/lerobot_hf)

[![LeRobot](https://raw.githubusercontent.com/huggingface/lerobot/main/media/lerobot-logo-light.png)](https://huggingface.co/lerobot)

## 🤗 LeRobot: Making AI for Robotics more accessible with end-to-end learning

LeRobot provides models, datasets, and tools for real-world robotics in PyTorch. The goal is to lower the barrier to entry to robotics so that everyone can contribute and benefit from sharing datasets and pretrained models.

LeRobot contains state-of-the-art approaches that have been shown to transfer to the real-world with a focus on imitation learning and reinforcement learning.

LeRobot offers:

🤗 A library of state-of-the-art policies that are pretrained and ready to deploy.

🤗 A hardware-agnostic, Python-native interface that standardizes control across diverse platforms, from low-cost arms (SO-100) to humanoids.

🤗 A standardized, scalable LeRobotDataset format (Parquet + MP4 or images) hosted on the Hugging Face Hub, enabling efficient storage, streaming and visualization of massive robotic datasets.

🤗 State-of-the-art policies that have been shown to transfer to the real-world ready for training and deployment.

🤗 Comprehensive support for the open-source ecosystem to democratize physical AI.

## This fork: W&B-native SO-101 workflow

This fork adds an optional W&B Artifacts and Registry path for moving finalized datasets and
policies between a recording machine, a training machine, and a robot machine. The worked example
keeps W&B outside the robot control loop and uses no Hugging Face Hub storage on that path.

> [!IMPORTANT]
> `pip install lerobot` installs the upstream package and does **not** include this fork's
> `lerobot-wandb` command. Clone this repository, install its locked environment, and activate it
> before following the manual:

```bash
git clone https://github.com/guchengwei/lerobot.git
cd lerobot
uv sync --locked --extra core_scripts --extra feetech --extra training
source .venv/bin/activate
lerobot-wandb --help
```

Follow the worked end-to-end walkthrough in [English](./examples/wandb_showcase/README.md) or
[日本語](./examples/wandb_showcase/README.ja.md). It covers recording, dataset publication, training
from an immutable Artifact version, model download, real-robot rollout, rollout publication with
lineage, and promotion of the exact evaluated model version. Its commands assume the source
environment above is active.

The integration's terminology and architectural boundaries are documented in
[`CONTEXT.md`](./CONTEXT.md).

## Quick Start

For upstream LeRobot without this fork's W&B integration, install the published package from PyPI:

```bash
pip install lerobot
lerobot-info
```

> [!IMPORTANT]
> For detailed installation guide, please see the [Installation Documentation](https://huggingface.co/docs/lerobot/installation).

## Robots & Control

<div align="center">
  <img src="./media/readme/robots_control_video.webp" width="640px" alt="Reachy 2 Demo">
</div>

LeRobot provides a unified `Robot` class interface that decouples control logic from hardware specifics. It supports a wide range of robots and teleoperation devices.

```python
from lerobot.robots.so100_follower import SO100FollowerConfig, SO100Follower

config = SO100FollowerConfig(
    port="/dev/tty.usbmodem585A0076891",
    id="my_awesome_follower_arm",
)
robot = SO100Follower(config)
robot.connect()

# Get robot state
observation = robot.get_observation()

# Send action
action = {"shoulder_pan.pos": 1.5, "shoulder_lift.pos": -1.0}
robot.send_action(action)
```

For more information, see the [Robots documentation](https://huggingface.co/docs/lerobot/robots).

## Datasets

<div align="center">
  <img src="./media/readme/datasets.webp" width="640px" alt="LeRobotDataset">
</div>

LeRobot provides a standardized dataset format, `LeRobotDataset`, built on top of Apache Parquet and MP4. It is optimized for streaming, visualization, and training.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load a dataset
dataset = LeRobotDataset("lerobot/pusht")

# Access an episode
episode = dataset[0]
```

Browse datasets on the [LeRobot Hub](https://huggingface.co/lerobot).

## Policies

<div align="center">
  <img src="./media/readme/policies.webp" width="640px" alt="LeRobot Policies">
</div>

LeRobot includes implementations of state-of-the-art policies such as ACT, Diffusion Policy, and TDMPC. These policies are designed to be trained on LeRobot datasets and deployed on real robots.

```python
from lerobot.policies.act.modeling_act import ACTPolicy

# Load a pretrained policy
policy = ACTPolicy.from_pretrained("lerobot/act_aloha_sim_transfer_cube_human")
```

For more information, see the [Policies documentation](https://huggingface.co/docs/lerobot/policies).

## Training

Train a policy on a LeRobot dataset using the `lerobot-train` command:

```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --batch_size=8 \
  --steps=100000
```

## Evaluation

Evaluate a trained policy in simulation or on a real robot:

```bash
lerobot-eval \
  --policy.path=outputs/train/act_aloha_sim_transfer_cube_human/checkpoints/last/pretrained_model \
  --env.type=aloha \
  --eval.batch_size=10 \
  --eval.n_episodes=10
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Citation

If you find LeRobot useful in your research, please cite:

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

## License

LeRobot is released under the [Apache 2.0 License](LICENSE).
