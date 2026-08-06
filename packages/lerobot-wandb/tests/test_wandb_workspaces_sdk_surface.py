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
"""Network-free guard against ``wandb-workspaces`` drifting out from under the workspace
command (``lerobot_wandb.workspace``) within its pinned version range.

The mocked tests in ``test_workspace.py`` can't detect the real package changing shape; this
asserts the exact classes/parameters/properties that module calls on still exist. It never
imports anything that triggers a network call.
"""

import inspect

import pytest

pytest.importorskip(
    "wandb_workspaces", reason="wandb_workspaces is required (install lerobot-wandb[wandb-workspace])"
)

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
from wandb_workspaces import _graphql
from wandb_workspaces.reports.v2.interface import _metric_to_backend


def _params(callable_obj) -> set[str]:
    return set(inspect.signature(callable_obj).parameters)


def test_workspace_constructor_accepts_expected_params():
    params = _params(ws.Workspace)
    assert {"entity", "project", "name", "sections", "runset_settings"} <= params


def test_workspace_save_and_lookup_surface_exists():
    # save() is the only write path the module uses; from_url is the only lookup path.
    assert hasattr(ws.Workspace, "save")
    assert hasattr(ws.Workspace, "from_url")
    # The module reads the resolved URL off a loaded/saved workspace, and keys reuse on
    # the deterministic internal name/id the SDK preserves across load -> save round-trips.
    assert hasattr(ws.Workspace, "url")
    assert "_internal_name" in ws.Workspace.__dataclass_fields__
    assert "_internal_id" in ws.Workspace.__dataclass_fields__


def test_section_constructor_accepts_expected_params():
    params = _params(ws.Section)
    assert {"name", "panels", "is_open"} <= params


def test_media_browser_accepts_expected_params():
    params = _params(wr.MediaBrowser)
    assert {"media_keys", "mode", "gallery_axis"} <= params


def test_scalar_chart_accepts_expected_params():
    params = _params(wr.ScalarChart)
    assert {"metric", "title"} <= params


def test_summary_metric_serializes_to_summary_metrics():
    # The template binds rollout-fact charts to SummaryMetric because the upload path
    # writes them to run.summary only; a bare string would bind to history.
    assert _params(wr.SummaryMetric) >= {"name"}
    assert _metric_to_backend(wr.SummaryMetric("success_rate")) == "summary_metrics.success_rate"


def test_runset_settings_accepts_expected_params():
    params = _params(ws.RunsetSettings)
    assert {"filters", "pinned_columns"} <= params


def test_app_url_helper_exists():
    # The module builds the lookup URL from the same helper the package's own
    # Workspace.url property uses; if that helper disappears, the module must know.
    assert hasattr(_graphql, "get_app_url")
    assert {"api"} <= _params(_graphql.get_app_url)
