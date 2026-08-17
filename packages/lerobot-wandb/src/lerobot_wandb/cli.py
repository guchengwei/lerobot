#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""``lerobot-wandb``: move LeRobot datasets to/from W&B Artifacts. Never touches the Hub.

Deliberately ``argparse``-based rather than wired into the central Draccus config parser,
consistent with this repo's existing precedent for small standalone scripts. Transfer commands
always use an online W&B run: their success contract is a durable, cross-machine artifact, not a
local offline run that still needs a later ``wandb sync``.

Examples:

```shell
lerobot-wandb dataset upload --root ./my-dataset --entity my-team --project my-project \
    --name pick-cube --alias raw

# v2.1/GR00T: log the exact episode as browser-playable run media while preserving the Artifact.
lerobot-wandb dataset upload --root ./my-v21-dataset --entity my-team --project my-project \
    --name pick-cube-v21 --preview-episode 10

lerobot-wandb dataset download --ref my-team/my-project/pick-cube:latest --root ./materialized

lerobot-wandb model upload --root ./outputs/train/pretrained_model --entity my-team \
    --project my-project --name pick-cube-policy --alias candidate \
    --registry-collection pick-cube-policy

lerobot-wandb model download --ref my-team/my-project/pick-cube-policy:latest --root ./policy

lerobot-wandb model promote --ref my-team/my-project/pick-cube-policy:v3 --alias production \
    --registry-collection pick-cube-policy

lerobot-wandb rollout upload --root ./rollout_pick-cube --entity my-team --project my-project \
    --name pick-cube-rollout --model-ref my-team/my-project/pick-cube-policy:v3 \
    --episodes-succeeded 7

Dataset and rollout Artifacts keep their original videos unchanged. Browser-compatible H.264/yuv420p
previews are separate run media; they are review derivatives, never canonical training data.
```
"""

import argparse
import contextlib
import logging
import tempfile
from pathlib import Path

import wandb

from .compatibility import LeRobotCompatibilityError, set_allow_unsupported
from .dataset_transfer import (
    DatasetPreviewSource,
    inspect_transfer_dataset,
    prepare_dataset_preview,
    select_dataset_preview_sources,
    validate_transfer_dataset,
)
from .inspect import (
    inspect_dataset_directory,
    inspect_model_directory,
    registry_link_refusal,
    validate_model_directory,
)
from .refs import parse_artifact_ref
from .rollout import (
    ROLLOUT_ARTIFACT_TYPE,
    RolloutSummary,
    prepare_rollout_preview,
    select_representative_video,
    validate_success_count,
)
from .store import (
    MODEL_ARTIFACT_TYPE,
    declare_input,
    download_artifact,
    promote_model,
    upload_directory,
)
from .workspace import (
    DEFAULT_WORKSPACE_NAME,
    WorkspaceDependencyError,
    WorkspaceNameCollisionError,
    WorkspaceProjectNotFoundError,
    create_rollout_workspace,
)

DATASET_ARTIFACT_TYPE = "dataset"


def _dataset_media_key(source: DatasetPreviewSource, index: int) -> str:
    episode = f"episode_{source.episode:06d}" if source.episode is not None else "representative"
    camera = "".join(character if character.isalnum() else "_" for character in source.video_key).strip("_")
    return f"dataset_video/{episode}/{camera or f'camera_{index:03d}'}"


def cmd_dataset_upload(args: argparse.Namespace) -> None:
    # Transfer validation is version-aware: current v3 uses the current reader contract, while a
    # canonical v2.1 directory is validated locally without pretending the current reader can train
    # from it. Preview selection/preparation is also local and happens before any W&B run exists.
    dataset = inspect_transfer_dataset(args.root)
    aliases = args.aliases or ["latest"]
    sources = [] if args.no_preview else select_dataset_preview_sources(dataset, episodes=args.preview_episodes)

    with contextlib.ExitStack() as exit_stack:
        previews: list[tuple[DatasetPreviewSource, Path]] = []
        if sources:
            root = args.root.resolve()
            tmp_dir = Path(
                exit_stack.enter_context(
                    tempfile.TemporaryDirectory(dir=root.parent, prefix=f"{root.name}-dataset-preview-")
                )
            ).resolve()
            if tmp_dir == root or root in tmp_dir.parents:
                raise ValueError(
                    f"The dataset preview temp dir {tmp_dir} must be outside the dataset root {root}: "
                    "a preview inside the artifact root would be uploaded with it."
                )
            for index, source in enumerate(sources):
                preview = prepare_dataset_preview(
                    root / source.relative_path,
                    tmp_dir / f"preview-{index:03d}.mp4",
                )
                previews.append((source, preview))

        run = wandb.init(entity=args.entity, project=args.project, job_type="dataset_upload", mode="online")
        try:
            result = upload_directory(
                run,
                args.root,
                name=args.name,
                artifact_type=DATASET_ARTIFACT_TYPE,
                aliases=aliases,
                metadata=dataset.metadata.to_wandb_metadata(),
            )
            if previews:
                # Artifact files are canonical bytes, not W&B run media. Explicit wandb.Video
                # values make the selected H.264 previews visible in W&B's Media browser.
                run.log(
                    {
                        _dataset_media_key(
                            source, index
                        ): wandb.Video(str(preview))
                        for index, (source, preview) in enumerate(previews)
                    }
                )
        finally:
            # A transcoded preview lives in the sibling temp dir, so it must outlive run.finish().
            run.finish()

        print(f"Uploaded dataset artifact: {result.resolved_ref}")
        print(f"Aliases applied: {', '.join(aliases)}")
        print(f"Dataset schema: {dataset.metadata.schema_version} ({dataset.layout} transfer layout)")
        if args.no_preview:
            print("Run media preview: disabled (--no-preview).")
        elif not previews:
            print("Run media preview: no video exists in this dataset.")
        else:
            for source, _preview in previews:
                episode = (
                    f"episode {source.episode}"
                    if source.episode is not None
                    else "representative v3 chunk"
                )
                print(f"Run media preview: {episode}, {source.video_key} <- {source.relative_path}")
            print("The Artifact keeps the original video bytes; playback is on this upload Run's Media tab.")


def cmd_dataset_download(args: argparse.Namespace) -> None:
    # Fail fast on a malformed ref before a run ever starts.
    parsed = parse_artifact_ref(args.ref)

    # The lineage run's own home defaults to the artifact's entity/project, but a caller with only
    # read access to the source project (e.g. a shared team dataset) needs to log it somewhere they
    # can actually create runs — `use_artifact` accepts a fully qualified ref regardless of which
    # project the run itself lives in, so overriding here never changes which artifact is fetched.
    run = wandb.init(
        entity=args.entity or parsed.entity,
        project=args.project or parsed.project,
        job_type="dataset_download",
        mode="online",
    )
    try:
        result = download_artifact(
            run,
            parsed,
            expected_type=DATASET_ARTIFACT_TYPE,
            download_root=args.root,
            validator=validate_transfer_dataset,
        )
    finally:
        run.finish()

    print(f"Downloaded dataset artifact {result.resolved_ref} to: {result.local_path}")


def cmd_model_upload(args: argparse.Namespace) -> None:
    # Validate — and pay any local, no-network cost of a bad directory — before a run ever starts.
    metadata = inspect_model_directory(args.root)
    aliases = args.aliases or ["latest"]

    artifact_metadata = metadata.to_wandb_metadata()
    registry_collection = args.registry_collection
    refusal = registry_link_refusal(
        is_self_contained=metadata.is_self_contained,
        base_model_name_or_path=metadata.base_model_name_or_path,
    )
    if registry_collection is not None and refusal is not None:
        logging.warning(
            f"Not linking into Registry collection {registry_collection!r}: {refusal}. The "
            "Artifact is still uploaded — upload a merged checkpoint to register a deployable "
            "version."
        )
        artifact_metadata["registry_link_refused_reason"] = refusal
        registry_collection = None

    run = wandb.init(entity=args.entity, project=args.project, job_type="model_upload", mode="online")
    try:
        result = upload_directory(
            run,
            args.root,
            name=args.name,
            artifact_type=MODEL_ARTIFACT_TYPE,
            aliases=aliases,
            metadata=artifact_metadata,
            registry_collection=registry_collection,
        )
    finally:
        run.finish()

    print(f"Uploaded model artifact: {result.resolved_ref}")
    print(f"Aliases applied: {', '.join(aliases)}")
    if result.registry_collection:
        print(f"Linked into registry collection: {result.registry_collection}")
    elif args.registry_collection:
        print(f"NOT linked into registry collection {args.registry_collection}: {refusal}")


def cmd_model_download(args: argparse.Namespace) -> None:
    # Fail fast on a malformed ref before a run ever starts.
    parsed = parse_artifact_ref(args.ref)

    # The lineage run's own home defaults to the artifact's entity/project, but a caller with only
    # read access to the source project (e.g. a shared team dataset) needs to log it somewhere they
    # can actually create runs — `use_artifact` accepts a fully qualified ref regardless of which
    # project the run itself lives in, so overriding here never changes which artifact is fetched.
    run = wandb.init(
        entity=args.entity or parsed.entity,
        project=args.project or parsed.project,
        job_type="model_download",
        mode="online",
    )
    try:
        result = download_artifact(
            run,
            parsed,
            expected_type=MODEL_ARTIFACT_TYPE,
            download_root=args.root,
            validator=validate_model_directory,
        )
    finally:
        run.finish()

    print(
        f"Downloaded model artifact {result.resolved_ref} to: {result.local_path} "
        "(use directly as a rollout policy path)"
    )


def cmd_model_promote(args: argparse.Namespace) -> None:
    # Fail fast on a malformed ref before any network call.
    parsed = parse_artifact_ref(args.ref)

    result = promote_model(
        parsed,
        alias=args.alias,
        registry_collection=args.registry_collection,
    )

    print(f"Promoted model artifact: {result.resolved_ref}")
    print(f"Alias applied: {args.alias}")
    print(f"Digest (unchanged — nothing was uploaded): {result.digest}")
    if result.registry_collection:
        print(f"Linked into registry collection: {result.registry_collection}")


def cmd_rollout_upload(args: argparse.Namespace) -> None:
    # Everything local and fallible happens before `wandb.init` creates a run: a bad directory, an
    # impossible success count, a malformed model ref or an unavailable preview encoder must not
    # leave an empty run behind.
    metadata = inspect_dataset_directory(args.root)
    validate_success_count(args.episodes_succeeded, metadata.total_episodes)
    parsed_model_ref = parse_artifact_ref(args.model_ref)
    video = select_representative_video(args.root)
    aliases = args.aliases or ["latest"]

    with contextlib.ExitStack() as exit_stack:
        preview_path: Path | None = None
        if video is not None:
            # A display derivative only, for the run UI: browsers play H.264/yuv420p, not the
            # dataset's AV1. It lives in a caller-owned temp dir (not the rollout root) so it can
            # never enter the Artifact manifest; the original stays in the Artifact unchanged.
            # The temp dir is pinned next to the rollout root — never left to TMPDIR, which could
            # point under the root and put the preview inside the Artifact being uploaded.
            root = args.root.resolve()
            tmp_dir = Path(
                exit_stack.enter_context(
                    tempfile.TemporaryDirectory(dir=root.parent, prefix=f"{root.name}-preview-")
                )
            ).resolve()
            if tmp_dir == root or root in tmp_dir.parents:
                raise ValueError(
                    f"The preview temp dir {tmp_dir} must be outside the rollout root {root}: "
                    "a preview inside the artifact root would be uploaded with it."
                )
            preview_path = prepare_rollout_preview(root / video.path, tmp_dir / "preview.mp4")

        run = wandb.init(entity=args.entity, project=args.project, job_type="rollout_upload", mode="online")
        try:
            # Lineage only: the model that produced this rollout is referenced, never downloaded.
            model = declare_input(run, parsed_model_ref, expected_type=MODEL_ARTIFACT_TYPE)
            summary = RolloutSummary.build(
                metadata,
                successes=args.episodes_succeeded,
                model_requested_ref=model.requested_ref,
                model_resolved_ref=model.resolved_ref,
                video=video,
            )
            result = upload_directory(
                run,
                args.root,
                name=args.name,
                artifact_type=ROLLOUT_ARTIFACT_TYPE,
                aliases=aliases,
                metadata={**metadata.to_wandb_metadata(), **summary.to_wandb_metadata()},
            )
            run.summary.update(summary.to_wandb_metadata())
            if preview_path is not None:
                # Path only — no fps=/format=: those don't transcode path input and lie about it.
                run.log({"rollout_video": wandb.Video(str(preview_path))})
        finally:
            # Inside the `with`: the preview temp dir must outlive run.finish().
            run.finish()

        print(f"Uploaded rollout artifact: {result.resolved_ref}")
        print(f"Aliases applied: {', '.join(aliases)}")
        print(f"Model input (lineage): {model.resolved_ref}")
        print(
            f"Episodes: {summary.episodes} | successes: {summary.successes} "
            f"| success rate: {summary.success_rate:.1%} | duration: {summary.duration_s:.1f}s"
        )
        if video is None:
            print("No video in this rollout dataset: nothing logged as run media.")
        else:
            print(
                f"Representative video: {video.path} ({video.video_key}, "
                f"episode(s) {', '.join(str(index) for index in video.episodes)})"
            )


def cmd_workspace_create(args: argparse.Namespace) -> None:
    # Deliberately no `wandb.init`: creating a workspace is project configuration, not a
    # data-producing run. Failures — the actionable classes plus ValueError, the SDK's own
    # not-found/parse error — get a concise actionable line and a nonzero exit instead of
    # a traceback. Unexpected exceptions still surface as tracebacks.
    try:
        result = create_rollout_workspace(
            entity=args.entity,
            project=args.project,
            name=args.name,
            replace=args.replace,
        )
    except (
        WorkspaceDependencyError,
        WorkspaceProjectNotFoundError,
        WorkspaceNameCollisionError,
        ValueError,
    ) as error:
        print(f"Error: {error}")
        raise SystemExit(1) from error
    print(f"{result.status.capitalize()} workspace: {result.url}")


def _add_upload_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, required=True, help="Local directory to upload.")
    parser.add_argument("--entity", default=None, help="W&B entity. Defaults to your W&B default entity.")
    parser.add_argument("--project", required=True, help="W&B project to upload into.")
    parser.add_argument("--name", required=True, help="Artifact collection name.")
    parser.add_argument(
        "--alias", dest="aliases", action="append", default=[], help="Repeatable. Defaults to ['latest']."
    )


def _add_download_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ref", required=True, help="Artifact reference: entity/project/name:version_or_alias"
    )
    parser.add_argument("--root", type=Path, required=True, help="Local directory to materialize into.")
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity to create the lineage run in. Defaults to the artifact's own entity (--ref). "
        "Override if you can read the artifact but can't create runs in its project.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="W&B project to create the lineage run in. Defaults to the artifact's own project (--ref).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lerobot-wandb", description="Move LeRobot datasets and models to/from W&B Artifacts."
    )
    parser.add_argument(
        "--allow-unsupported-lerobot",
        action="store_true",
        help="Experimental: proceed even when the installed LeRobot version is outside the "
        "supported range. Never substitutes for a missing LeRobot install.",
    )
    resource_subparsers = parser.add_subparsers(dest="resource", required=True)

    dataset_parser = resource_subparsers.add_parser("dataset", help="Upload/download a dataset artifact.")
    dataset_action_subparsers = dataset_parser.add_subparsers(dest="action", required=True)

    dataset_upload_parser = dataset_action_subparsers.add_parser(
        "upload",
        help="Validate and upload a local v3.0 or canonical v2.1 dataset as a versioned W&B "
        "Artifact, with a browser-playable review preview when video exists.",
    )
    _add_upload_args(dataset_upload_parser)
    dataset_upload_parser.add_argument(
        "--preview-episode",
        dest="preview_episodes",
        type=int,
        action="append",
        default=[],
        help="Repeatable v2.1-only exact episode selector for W&B run-media previews. All camera "
        "videos for each requested episode are logged. Without this flag, one deterministic "
        "representative video is logged. v3 stores multiple episodes per video file, so exact "
        "episode selection is refused there rather than mislabeled.",
    )
    dataset_upload_parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Upload only the canonical Artifact and skip W&B run media. Useful when no H.264 "
        "encoder is available or review media is intentionally undesired.",
    )
    dataset_upload_parser.set_defaults(func=cmd_dataset_upload)

    dataset_download_parser = dataset_action_subparsers.add_parser(
        "download",
        help="Download and validate a v3.0 or canonical v2.1 dataset Artifact without converting it.",
    )
    _add_download_args(dataset_download_parser)
    dataset_download_parser.set_defaults(func=cmd_dataset_download)

    model_parser = resource_subparsers.add_parser("model", help="Upload/download a model artifact.")
    model_action_subparsers = model_parser.add_subparsers(dest="action", required=True)

    model_upload_parser = model_action_subparsers.add_parser(
        "upload", help="Validate and upload a local model directory as a versioned W&B Artifact."
    )
    _add_upload_args(model_upload_parser)
    model_upload_parser.add_argument(
        "--registry-collection",
        default=None,
        help="If set, link the uploaded version into this unified-Registry collection "
        "(wandb-registry-model/<name>).",
    )
    model_upload_parser.set_defaults(func=cmd_model_upload)

    model_download_parser = model_action_subparsers.add_parser(
        "download", help="Download a model Artifact into a local, rollout-ready policy directory."
    )
    _add_download_args(model_download_parser)
    model_download_parser.set_defaults(func=cmd_model_download)

    model_promote_parser = model_action_subparsers.add_parser(
        "promote",
        help="Alias an existing model version, and optionally link that same version into the "
        "Registry. Uploads nothing.",
    )
    model_promote_parser.add_argument(
        "--ref",
        required=True,
        help="Model version to promote: entity/project/name:version_or_alias. Prefer the immutable "
        "version a rollout actually evaluated (its 'model_artifact_resolved_ref').",
    )
    model_promote_parser.add_argument(
        "--alias",
        required=True,
        help="Alias to move onto this version, e.g. 'production'.",
    )
    model_promote_parser.add_argument(
        "--registry-collection",
        default=None,
        help="If set, also link this version into that unified-Registry collection "
        "(wandb-registry-model/<name>).",
    )
    model_promote_parser.set_defaults(func=cmd_model_promote)

    rollout_parser = resource_subparsers.add_parser("rollout", help="Upload a rollout result.")
    rollout_action_subparsers = rollout_parser.add_subparsers(dest="action", required=True)

    rollout_upload_parser = rollout_action_subparsers.add_parser(
        "upload",
        help="Validate and upload a local rollout dataset as a versioned W&B Artifact, with the "
        "model that produced it declared as a run input.",
    )
    _add_upload_args(rollout_upload_parser)
    rollout_upload_parser.add_argument(
        "--model-ref",
        required=True,
        help="Model artifact that produced this rollout: entity/project/name:version_or_alias. "
        "Referenced for lineage only — never downloaded.",
    )
    rollout_upload_parser.add_argument(
        "--episodes-succeeded",
        type=int,
        required=True,
        help="How many episodes the operator judged successful. Not auto-detected.",
    )
    rollout_upload_parser.set_defaults(func=cmd_rollout_upload)

    workspace_parser = resource_subparsers.add_parser(
        "workspace", help="Set up a reusable rollout-review workspace. No run is created."
    )
    workspace_action_subparsers = workspace_parser.add_subparsers(dest="action", required=True)

    workspace_create_parser = workspace_action_subparsers.add_parser(
        "create",
        help="Create (or reuse, or with --replace refresh) the named rollout-review workspace "
        "in a project. Idempotent: re-running never duplicates or mutates other workspaces.",
    )
    workspace_create_parser.add_argument("--entity", required=True, help="W&B entity that owns the project.")
    workspace_create_parser.add_argument(
        "--project", required=True, help="W&B project to create the workspace in."
    )
    workspace_create_parser.add_argument(
        "--name",
        default=DEFAULT_WORKSPACE_NAME,
        help=f"Display name of the workspace. Defaults to {DEFAULT_WORKSPACE_NAME!r}.",
    )
    workspace_create_parser.add_argument(
        "--replace",
        action="store_true",
        help="If the named workspace already exists, refresh it with the deterministic "
        "LeRobot template instead of reusing it as-is. Only the named workspace is touched.",
    )
    workspace_create_parser.set_defaults(func=cmd_workspace_create)

    return parser


def _init_logging() -> None:
    """Console logging for the sidecar CLI, using standard-library logging only.

    The companion distribution must not depend on LeRobot's private logging setup:
    this mirrors its console format without importing from ``lerobot``.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: list[str] | None = None) -> None:
    _init_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    set_allow_unsupported(args.allow_unsupported_lerobot)
    try:
        args.func(args)
    except LeRobotCompatibilityError as error:
        print(f"Error: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
