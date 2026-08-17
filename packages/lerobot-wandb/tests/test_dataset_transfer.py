# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot_wandb.dataset_transfer import (
    TransferDataset,
    inspect_transfer_dataset,
    select_dataset_preview_sources,
)
from lerobot_wandb.inspect import DatasetDirectoryError, DatasetDirectoryMetadata


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_v21_dataset(root: Path, *, cameras: tuple[str, ...] = ("observation.images.wrist",)) -> None:
    features = {
        "action": {"dtype": "float32", "shape": [1], "names": ["motor"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for camera in cameras:
        features[camera] = {"dtype": "video", "shape": [3, 8, 8], "names": None}

    info = {
        "codebase_version": "v2.1",
        "robot_type": "so101",
        "fps": 30,
        "total_episodes": 2,
        "total_frames": 4,
        "total_tasks": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "total_videos": 2 * len(cameras),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    _write_jsonl(
        root / "meta/episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["pick"], "length": 2},
            {"episode_index": 1, "tasks": ["pick"], "length": 2},
        ],
    )
    _write_jsonl(
        root / "meta/episodes_stats.jsonl",
        [
            {"episode_index": 0, "stats": {}},
            {"episode_index": 1, "stats": {}},
        ],
    )
    _write_jsonl(root / "meta/tasks.jsonl", [{"task_index": 0, "task": "pick"}])

    for episode in range(2):
        data_path = root / f"data/chunk-000/episode_{episode:06d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "action": pa.array([[0.0], [1.0]], type=pa.list_(pa.float32(), 1)),
                "timestamp": pa.array([0.0, 1 / 30], type=pa.float32()),
                "frame_index": pa.array([0, 1], type=pa.int64()),
                "episode_index": pa.array([episode, episode], type=pa.int64()),
                "index": pa.array([episode * 2, episode * 2 + 1], type=pa.int64()),
                "task_index": pa.array([0, 0], type=pa.int64()),
            }
        )
        pq.write_table(table, data_path)
        for camera in cameras:
            video = root / f"videos/chunk-000/{camera}/episode_{episode:06d}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"test-video-bytes")


def test_v21_transfer_accepts_episode_per_file_layout(tmp_path):
    root = tmp_path / "v21"
    _write_v21_dataset(root)

    dataset = inspect_transfer_dataset(root)

    assert dataset.layout == "v2.1"
    assert dataset.metadata.schema_version == "v2.1"
    assert dataset.metadata.total_episodes == 2
    assert dataset.metadata.video_keys == ("observation.images.wrist",)
    preview = select_dataset_preview_sources(dataset)[0]
    assert preview.episode == 0
    assert preview.relative_path == Path(
        "videos/chunk-000/observation.images.wrist/episode_000000.mp4"
    )


def test_v21_default_preview_is_one_deterministic_video(tmp_path):
    root = tmp_path / "v21"
    _write_v21_dataset(root, cameras=("observation.images.front", "observation.images.wrist"))
    dataset = inspect_transfer_dataset(root)

    previews = select_dataset_preview_sources(dataset)

    assert len(previews) == 1
    assert previews[0].episode == 0
    assert previews[0].video_key == "observation.images.front"
    assert previews[0].relative_path == Path(
        "videos/chunk-000/observation.images.front/episode_000000.mp4"
    )


def test_v21_preview_episode_selects_every_camera(tmp_path):
    root = tmp_path / "v21"
    _write_v21_dataset(root, cameras=("observation.images.front", "observation.images.wrist"))
    dataset = inspect_transfer_dataset(root)

    previews = select_dataset_preview_sources(dataset, episodes=[1])

    assert [preview.episode for preview in previews] == [1, 1]
    assert [preview.video_key for preview in previews] == [
        "observation.images.front",
        "observation.images.wrist",
    ]
    assert all("episode_000001.mp4" in str(preview.relative_path) for preview in previews)


def test_v21_missing_video_is_rejected(tmp_path):
    root = tmp_path / "v21"
    _write_v21_dataset(root)
    (root / "videos/chunk-000/observation.images.wrist/episode_000001.mp4").unlink()

    with pytest.raises(DatasetDirectoryError, match="missing v2.1 video for episode 1"):
        inspect_transfer_dataset(root)


def test_v3_exact_episode_preview_is_refused(tmp_path):
    metadata = DatasetDirectoryMetadata(
        schema_version="v3.0",
        robot_type="so101",
        fps=30,
        total_episodes=20,
        total_frames=100,
        total_tasks=1,
        camera_keys=("observation.images.wrist",),
        video_keys=("observation.images.wrist",),
        git_commit=None,
    )
    dataset = TransferDataset(root=tmp_path, layout="v3", metadata=metadata, info={})

    with pytest.raises(DatasetDirectoryError, match="exact only for v2.1"):
        select_dataset_preview_sources(dataset, episodes=[10])
