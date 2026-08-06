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
"""One-time, explicit setup of a reusable W&B project workspace for reviewing rollout runs.

`wandb-workspaces` is an optional dependency: upload/download commands never touch this module,
and importing it must not require the extra to be installed. All workspace-API calls are
lazily imported and confined here.

Idempotency note: the official ``wandb-workspaces`` API can only look up a workspace by its
*internal* view name (the ``nw=`` URL parameter), never by display name. This module therefore
assigns each managed workspace a deterministic internal name derived from its display name, so
re-running the command finds and reuses the same view instead of duplicating it. A workspace a
user created manually in the UI (with a random internal name) is not discoverable this way — a
documented limitation of the public API, not something to paper over by scraping the web UI.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from typing import Any, Literal

import wandb

DEFAULT_WORKSPACE_NAME = "LeRobot Rollouts"
ROLLOUT_MEDIA_KEY = "rollout_video"
ROLLOUT_JOB_TYPE_FILTER = "JobType = 'rollout_upload'"
_SECTION_NAME = "Rollout Review"
_INTERNAL_NAME_PREFIX = "nw-"
_INTERNAL_NAME_SUFFIX = "-v"
_INSTALL_HINT = 'pip install "lerobot[wandb-workspace]"'


class WorkspaceDependencyError(RuntimeError):
    """Raised when the workspace command is used without the optional extra."""


class WorkspaceProjectNotFoundError(RuntimeError):
    """Raised when the target entity/project does not exist or is not accessible."""


class WorkspaceNameCollisionError(RuntimeError):
    """Raised when the found workspace's display name is not the one requested.

    The internal-name slug is lossy (``"LeRobot Rollouts"`` and ``"LeRobot-Rollouts"``
    collide), so a lookup hit is only trusted once the display name round-trips.
    """


@dataclass(frozen=True, slots=True)
class WorkspaceResult:
    """What the command did to the named workspace, and where it lives."""

    status: Literal["created", "reused", "replaced"]
    url: str


def _workspaces_api() -> tuple[Any, Any]:
    """Lazily import the workspace SDK, with an actionable error when it is absent.

    Returns the ``wandb_workspaces.workspaces`` and ``wandb_workspaces.reports.v2``
    modules, in that order.
    """
    try:
        ws = importlib.import_module("wandb_workspaces.workspaces")
        wr = importlib.import_module("wandb_workspaces.reports.v2")
    except ImportError as error:
        raise WorkspaceDependencyError(
            "`lerobot-wandb workspace create` requires the wandb-workspaces package. "
            f"Install it with `{_INSTALL_HINT}` and try again. Artifact upload/download "
            "commands do not need it."
        ) from error
    return ws, wr


def _slugify(name: str) -> str:
    """Lowercase, non-alphanumeric runs collapse to '-', so names round-trip into a URL param."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _internal_name(name: str) -> str:
    return f"{_INTERNAL_NAME_PREFIX}{_slugify(name)}{_INTERNAL_NAME_SUFFIX}"


def _nw_param(name: str) -> str:
    """The ``nw=`` URL parameter for a workspace, inverse of the internal-name form above."""
    return _internal_name(name).removeprefix(_INTERNAL_NAME_PREFIX).removesuffix(_INTERNAL_NAME_SUFFIX)


def _app_url() -> str:
    from wandb_workspaces._graphql import get_app_url

    return get_app_url(wandb.Api())


def _build_rollout_workspace(entity: str, project: str, name: str) -> Any:
    """The deterministic template: a small curated view bound only to existing rollout keys."""
    ws, wr = _workspaces_api()
    return ws.Workspace(
        entity=entity,
        project=project,
        name=name,
        sections=[
            ws.Section(
                name=_SECTION_NAME,
                is_open=True,
                panels=[
                    wr.MediaBrowser(
                        title="Rollout videos",
                        media_keys=[ROLLOUT_MEDIA_KEY],
                        mode="gallery",
                        gallery_axis="run",
                    ),
                    wr.ScalarChart(title="Success rate", metric="success_rate"),
                    wr.ScalarChart(title="Episodes", metric="episodes"),
                    wr.ScalarChart(title="Successes", metric="successes"),
                    wr.ScalarChart(title="Frames", metric="frames"),
                    wr.ScalarChart(title="Duration", metric="duration_s"),
                ],
            ),
        ],
        runset_settings=ws.RunsetSettings(
            filters=ROLLOUT_JOB_TYPE_FILTER,
            pinned_columns=[
                "summary:model_artifact_requested_ref",
                "summary:model_artifact_resolved_ref",
                "summary:success_rate",
                "summary:episodes",
                "summary:successes",
            ],
        ),
    )


def _ensure_project_exists(entity: str, project: str) -> None:
    """Fail fast, actionably, when the target project is missing or unreadable."""
    try:
        # Project metadata loads lazily; touching `.id` performs the fetch and raises
        # ValueError when the project does not exist.
        _ = wandb.Api().project(project, entity=entity).id
    except ValueError as error:
        raise WorkspaceProjectNotFoundError(
            f"Project `{entity}/{project}` does not exist or you cannot access it. "
            "Create it in the W&B app first, then re-run this command."
        ) from error


def _lookup_workspace(entity: str, project: str, name: str) -> Any | None:
    """Find the workspace with this display name, or ``None``.

    Only workspaces this command created (deterministic internal name) are discoverable —
    see the module docstring. ``Workspace.from_url`` raises ``ValueError`` for an unknown
    view, which maps a lookup miss to ``None``.
    """
    ws, _ = _workspaces_api()
    url = f"{_app_url().rstrip('/')}/{entity}/{project}?nw={_nw_param(name)}"
    try:
        return ws.Workspace.from_url(url)
    except ValueError:
        return None


def create_rollout_workspace(
    *,
    entity: str,
    project: str,
    name: str = DEFAULT_WORKSPACE_NAME,
    replace: bool = False,
) -> WorkspaceResult:
    """Create, reuse, or replace the named rollout-review workspace.

    Never mutates or duplicates any other workspace, and never creates suffixed
    duplicates on retries.
    """
    _workspaces_api()
    _ensure_project_exists(entity, project)

    workspace = _build_rollout_workspace(entity, project, name)
    existing = _lookup_workspace(entity, project, name)

    if existing is None:
        # A deterministic internal name is the handle re-runs look up (see module docstring).
        workspace._internal_name = _internal_name(name)  # noqa: SLF001
        status = "created"
    else:
        # The slug used for lookup is lossy; a hit is only trustworthy when the display
        # name round-trips, or --replace could refresh a differently-named workspace.
        if existing.name != name:
            raise WorkspaceNameCollisionError(
                f"A workspace with a very similar name already exists in `{entity}/{project}` "
                f"(found {existing.name!r}). Pick a distinct `--name`."
            )
        if not replace:
            return WorkspaceResult(status="reused", url=existing.url)
        # Replacing means updating exactly this view in place: keep its server-side
        # identity so the upsert refreshes it instead of inserting a sibling.
        workspace._internal_name = existing._internal_name  # noqa: SLF001
        workspace._internal_id = existing._internal_id  # noqa: SLF001
        status = "replaced"

    workspace.save()
    return WorkspaceResult(status=status, url=workspace.url)
