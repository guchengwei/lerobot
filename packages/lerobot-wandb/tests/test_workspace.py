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
"""``lerobot-wandb workspace create`` behavior, network-free.

All W&B calls are mocked at the module boundary of
``lerobot_wandb.workspace``. Template construction is
exercised against the real ``wandb_workspaces`` dataclasses, which are pure.
"""

import pytest

pytest.importorskip("wandb", reason="wandb is required (install lerobot[training])")
pytest.importorskip(
    "wandb_workspaces", reason="wandb_workspaces is required (install lerobot-wandb[wandb-workspace])"
)

from wandb_workspaces.reports.v2 import MediaBrowser, ScalarChart, SummaryMetric

from lerobot_wandb import workspace as ws_mod


def test_template_is_a_curated_rollout_review_workspace():
    template = ws_mod._build_rollout_workspace(
        entity="my-team", project="my-project", name="LeRobot Rollouts"
    )

    assert template.entity == "my-team"
    assert template.project == "my-project"
    assert template.name == "LeRobot Rollouts"

    assert len(template.sections) == 1
    section = template.sections[0]
    assert section.name == "Rollout Review"
    assert section.is_open is True
    assert section.pinned is True

    # The gallery is bound to the browser-playable preview media key from the
    # rollout upload, never to an invented key.
    media_panels = [p for p in section.panels if isinstance(p, MediaBrowser)]
    assert len(media_panels) == 1
    assert media_panels[0].media_keys == ["rollout_video"]
    assert media_panels[0].mode == "gallery"
    assert media_panels[0].gallery_axis == "run"

    # Every scalar panel is bound to a key that `RolloutSummary.to_wandb_metadata()`
    # already logs — the exact names, not renames — and as a SummaryMetric: the upload
    # path writes these facts to run.summary only, so a bare history metric would
    # render an empty chart.
    scalar_panels = {
        (p.title, p.metric.name)
        for p in section.panels
        if isinstance(p, ScalarChart) and isinstance(p.metric, SummaryMetric)
    }
    assert scalar_panels == {
        ("Success rate", "success_rate"),
        ("Episodes", "episodes"),
        ("Successes", "successes"),
        ("Frames", "frames"),
        ("Duration", "duration_s"),
    }
    assert sum(isinstance(p, ScalarChart) for p in section.panels) == len(scalar_panels)

    # The runset is scoped to rollout-upload runs, and the runs table exposes the
    # requested/resolved model refs plus the headline rollout facts as columns.
    runset = template.runset_settings
    assert runset.filters == "JobType = 'rollout_upload'"
    assert runset.pinned_columns == [
        "run:displayName",
        "summary:model_artifact_requested_ref",
        "summary:model_artifact_resolved_ref",
        "summary:success_rate",
        "summary:episodes",
        "summary:successes",
    ]


# ---------------------------------------------------------------------------
# State machine: created / reused / replaced, network-free at the module seam
# ---------------------------------------------------------------------------


class _FakeWorkspace:
    """Stand-in for a real workspace SDK object: only the attributes the module touches."""

    def __init__(
        self, url="https://wandb.ai/my-team/my-project?nw=lerobot-rollouts", name="LeRobot Rollouts"
    ):
        self.url = url
        self.name = name
        self._internal_name = ""
        self._internal_id = ""
        self.saved = 0

    def save(self):
        self.saved += 1
        return self


@pytest.fixture
def fake_ws(monkeypatch):
    monkeypatch.setattr(ws_mod, "_ensure_project_exists", lambda entity, project: None)
    fake = _FakeWorkspace()

    def _build(entity, project, name):
        return fake

    monkeypatch.setattr(ws_mod, "_build_rollout_workspace", _build)
    return fake


def test_create_when_absent(monkeypatch, fake_ws):
    monkeypatch.setattr(ws_mod, "_lookup_workspace", lambda entity, project, name: None)

    result = ws_mod.create_rollout_workspace(entity="my-team", project="my-project")

    assert result.status == "created"
    assert result.url == fake_ws.url
    # The deterministic internal name is the handle re-runs look up by.
    assert fake_ws._internal_name == "nw-lerobot-rollouts-v"
    assert fake_ws.saved == 1


def test_reuse_without_mutation_when_exists(monkeypatch, fake_ws):
    existing = _FakeWorkspace(url="https://wandb.ai/my-team/my-project?nw=lerobot-rollouts")
    monkeypatch.setattr(ws_mod, "_lookup_workspace", lambda entity, project, name: existing)

    result = ws_mod.create_rollout_workspace(entity="my-team", project="my-project")

    assert result.status == "reused"
    assert result.url == existing.url
    assert existing.saved == 0  # nothing is saved: the existing workspace is untouched
    assert fake_ws.saved == 0  # the freshly built template is discarded, never saved


def test_reuse_refuses_a_similarly_named_workspace(monkeypatch, fake_ws):
    """The lookup slug is lossy ('LeRobot Rollouts' vs 'LeRobot-Rollouts'): a hit whose
    display name does not round-trip must never be reused, let alone replaced."""
    existing = _FakeWorkspace(name="LeRobot-Rollouts")
    monkeypatch.setattr(ws_mod, "_lookup_workspace", lambda entity, project, name: existing)

    with pytest.raises(ws_mod.WorkspaceNameCollisionError, match="--name"):
        ws_mod.create_rollout_workspace(entity="my-team", project="my-project")

    assert existing.saved == 0
    assert fake_ws.saved == 0

    with pytest.raises(ws_mod.WorkspaceNameCollisionError):
        ws_mod.create_rollout_workspace(entity="my-team", project="my-project", replace=True)


def test_replace_refreshes_only_the_named_workspace(monkeypatch, fake_ws):
    existing = _FakeWorkspace()
    existing._internal_name = "nw-lerobot-rollouts-v"
    existing._internal_id = "view-id-42"
    monkeypatch.setattr(ws_mod, "_lookup_workspace", lambda entity, project, name: existing)

    result = ws_mod.create_rollout_workspace(entity="my-team", project="my-project", replace=True)

    assert result.status == "replaced"
    # The fresh template keeps the existing view's server-side identity, so the
    # upsert updates it in place instead of inserting a duplicate.
    assert fake_ws._internal_name == existing._internal_name
    assert fake_ws._internal_id == existing._internal_id
    assert fake_ws.saved == 1
    # The existing object itself is never saved (nothing else is touched).
    assert existing.saved == 0


def test_retry_never_creates_a_duplicate(monkeypatch, fake_ws):
    """First run creates; the next invocation finds it and reuses it without saving."""
    calls = {"lookups": 0}
    created = _FakeWorkspace()
    built = []

    def _build(entity, project, name):
        built.append(created)
        return created

    monkeypatch.setattr(ws_mod, "_build_rollout_workspace", _build)
    monkeypatch.setattr(
        ws_mod,
        "_lookup_workspace",
        lambda entity, project, name: calls.__setitem__("lookups", calls["lookups"] + 1) or None,
    )

    first = ws_mod.create_rollout_workspace(entity="my-team", project="my-project")
    assert first.status == "created"

    monkeypatch.setattr(ws_mod, "_lookup_workspace", lambda entity, project, name: created)
    second = ws_mod.create_rollout_workspace(entity="my-team", project="my-project")

    assert second.status == "reused"
    assert created.saved == 1  # saved exactly once across both invocations


def test_lookup_builds_the_deterministic_url(monkeypatch):
    """The lookup URL is the exact round-trip of the internal name (nw= param)."""
    monkeypatch.setattr(ws_mod, "_ensure_project_exists", lambda entity, project: None)
    monkeypatch.setattr(ws_mod, "_app_url", lambda: "https://wandb.ai/")
    monkeypatch.setattr(ws_mod, "_build_rollout_workspace", lambda *a, **k: _FakeWorkspace())
    looked_up = []
    import wandb_workspaces.workspaces as ws

    def _fake_from_url(url):
        looked_up.append(url)
        raise ValueError("Workspace not found")

    monkeypatch.setattr(ws.Workspace, "from_url", staticmethod(_fake_from_url))

    result = ws_mod.create_rollout_workspace(entity="my-team", project="my-project")

    assert looked_up == ["https://wandb.ai/my-team/my-project?nw=lerobot-rollouts"]
    assert result.status == "created"


def test_missing_optional_dependency_is_actionable(monkeypatch, fake_ws):
    def _no_workspaces(name):
        raise ImportError(f"No module named '{name}'")

    monkeypatch.setattr(ws_mod.importlib, "import_module", _no_workspaces)

    with pytest.raises(ws_mod.WorkspaceDependencyError, match=r"lerobot-wandb\[wandb-workspace\]"):
        ws_mod.create_rollout_workspace(entity="my-team", project="my-project")

    # The dependency gate fires before any project/network work.
    assert fake_ws.saved == 0


def test_missing_project_fails_actionably(monkeypatch):
    class _BoomApi:
        def project(self, project, entity=None):
            raise ValueError("Project my-project not found")

    def _never_lookup(*args, **kwargs):
        raise AssertionError("lookup must not run")

    monkeypatch.setattr(ws_mod.wandb, "Api", lambda: _BoomApi())
    monkeypatch.setattr(ws_mod, "_lookup_workspace", _never_lookup)

    with pytest.raises(ws_mod.WorkspaceProjectNotFoundError, match="my-team/my-project"):
        ws_mod.create_rollout_workspace(entity="my-team", project="my-project")
