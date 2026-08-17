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

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("wandb", reason="wandb is required")

from lerobot_wandb import cli
from lerobot_wandb.dataset_transfer import DatasetPreviewSource, TransferDataset
from lerobot_wandb.inspect import DatasetDirectoryMetadata


def _transfer_dataset(root: Path) -> TransferDataset:
    return TransferDataset(
        root=root,
        layout="v2.1",
        metadata=DatasetDirectoryMetadata(
            schema_version="v2.1",
            robot_type="so101",
            fps=30,
            total_episodes=11,
            total_frames=22,
            total_tasks=1,
            camera_keys=("observation.images.wrist",),
            video_keys=("observation.images.wrist",),
            git_commit=None,
        ),
        info={},
    )


def _args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        root=root,
        entity="my-team",
        project="my-project",
        name="pick-cube-v21",
        aliases=["raw"],
        preview_episodes=[10],
        no_preview=False,
    )


def test_dataset_upload_logs_playable_preview_and_keeps_it_through_finish(
    tmp_path, monkeypatch
):
    root = tmp_path / "dataset"
    source = root / "videos/chunk-000/observation.images.wrist/episode_000010.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")

    dataset = _transfer_dataset(root)
    selected = DatasetPreviewSource(
        episode=10,
        video_key="observation.images.wrist",
        relative_path=source.relative_to(root),
    )
    monkeypatch.setattr(cli, "inspect_transfer_dataset", lambda _root: dataset)
    monkeypatch.setattr(
        cli,
        "select_dataset_preview_sources",
        lambda _dataset, *, episodes: [selected],
    )

    state: dict[str, object] = {"prepared": False, "preview": None}

    def _prepare(passed_source: Path, destination: Path) -> Path:
        assert passed_source == source
        destination.write_bytes(b"h264-preview")
        state["prepared"] = True
        state["preview"] = destination
        return destination

    monkeypatch.setattr(cli, "prepare_dataset_preview", _prepare)

    run = MagicMock()
    run.entity = "my-team"
    run.project = "my-project"
    init_calls = []

    def _init(**kwargs):
        assert state["prepared"] is True
        init_calls.append(kwargs)
        return run

    monkeypatch.setattr(cli.wandb, "init", _init)
    monkeypatch.setattr(cli.wandb, "Video", lambda path: f"video:{path}")

    upload_calls = []

    def _upload(passed_run, directory, **kwargs):
        upload_calls.append((passed_run, Path(directory), kwargs))
        return SimpleNamespace(resolved_ref="my-team/my-project/pick-cube-v21:v0")

    monkeypatch.setattr(cli, "upload_directory", _upload)

    def _finish():
        preview = state["preview"]
        assert isinstance(preview, Path)
        assert preview.is_file()

    run.finish.side_effect = _finish

    cli.cmd_dataset_upload(_args(root))

    assert init_calls == [
        {"entity": "my-team", "project": "my-project", "job_type": "dataset_upload", "mode": "online"}
    ]
    assert len(upload_calls) == 1
    assert upload_calls[0][1] == root
    assert upload_calls[0][2]["metadata"]["schema_version"] == "v2.1"
    run.log.assert_called_once()
    media = run.log.call_args.args[0]
    assert list(media) == ["dataset_video/episode_000010/observation_images_wrist"]
    assert str(state["preview"]) in media[next(iter(media))]
    run.finish.assert_called_once()
    assert not Path(state["preview"]).exists()


def test_dataset_preview_failure_happens_before_wandb_init(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    source = root / "videos/chunk-000/observation.images.wrist/episode_000010.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    dataset = _transfer_dataset(root)
    selected = DatasetPreviewSource(10, "observation.images.wrist", source.relative_to(root))

    monkeypatch.setattr(cli, "inspect_transfer_dataset", lambda _root: dataset)
    monkeypatch.setattr(
        cli,
        "select_dataset_preview_sources",
        lambda _dataset, *, episodes: [selected],
    )

    def _fail_preview(*_args, **_kwargs):
        raise RuntimeError("encoder unavailable")

    monkeypatch.setattr(cli, "prepare_dataset_preview", _fail_preview)
    init = MagicMock()
    monkeypatch.setattr(cli.wandb, "init", init)

    with pytest.raises(RuntimeError, match="encoder unavailable"):
        cli.cmd_dataset_upload(_args(root))

    init.assert_not_called()
