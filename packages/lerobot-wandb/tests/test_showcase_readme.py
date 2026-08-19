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
"""The W&B showcase remains executable after its documented values are supplied.

Documentation that drifts from its CLI is worse than no documentation: a reader pastes a command
that no longer exists and concludes the tool is broken. These tests cover the mechanical half of
"verified to actually work": setup values are declared once and reused, runtime values remain
explicit, and every resulting command still parses against the real CLI. They deliberately do not
claim to exercise live W&B or robot hardware.
"""

import re
import shlex
from dataclasses import fields, make_dataclass
from pathlib import Path
from string import Template

import pytest

pytest.importorskip("wandb", reason="wandb is required (install lerobot[training])")

from lerobot_wandb import cli

REPO_ROOT = Path(__file__).parents[3]
README = REPO_ROOT / "examples" / "wandb_showcase" / "README.md"
ROOT_README = REPO_ROOT / "README.md"
PACKAGE_README = REPO_ROOT / "packages" / "lerobot-wandb" / "README.md"
JAPANESE_README = REPO_ROOT / "examples" / "wandb_showcase" / "README.ja.md"

CANONICAL_COMPANION_URL = "https://github.com/guchengwei/lerobot-wandb"
SOURCE_INSTALL = 'pip install "lerobot-wandb @ git+https://github.com/guchengwei/lerobot-wandb.git@main"'
SOURCE_INSTALL_WITH_LEROBOT = (
    'pip install "lerobot-wandb[lerobot] @ git+https://github.com/guchengwei/lerobot-wandb.git@main"'
)
OLD_SUBDIRECTORY_MARKER = "subdirectory=" + "packages/lerobot-wandb"

# A fenced bash block, then every backslash-continued command inside it that starts with the CLI
# under test. Other tools shown in the README (lerobot-record, lerobot-train, lerobot-rollout) are
# parsed by draccus from a much larger config surface and are deliberately out of scope here.
_BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.S)
_EXPORT = re.compile(r'^export\s+([A-Z_][A-Z0-9_]*)="([^"]*)"\s*$', re.M)
_FORK_DATASET_FIELDS = frozenset({"artifact_ref"})
_FORK_WANDB_FIELDS = frozenset({"model_artifact_aliases", "model_artifact_name", "registered_model_name"})


def _config_supports_fork_training_hooks(dataset_config: type, wandb_config: type) -> bool:
    dataset_fields = {field.name for field in fields(dataset_config)}
    wandb_fields = {field.name for field in fields(wandb_config)}
    return dataset_fields >= _FORK_DATASET_FIELDS and wandb_fields >= _FORK_WANDB_FIELDS


def _fork_training_hooks_available() -> bool:
    try:
        from lerobot.configs.default import DatasetConfig, WandBConfig
    except ImportError:
        return False
    return _config_supports_fork_training_hooks(DatasetConfig, WandBConfig)


def _readme_text() -> str:
    return README.read_text()


def _documented_env() -> dict[str, str]:
    return dict(_EXPORT.findall(_readme_text()))


def _expand_documented_values(command: str) -> str:
    expanded = Template(command).substitute(_documented_env())
    assert "$" not in expanded, f"README command contains an undeclared shell value: {expanded}"
    return expanded


def _raw_readme_commands() -> list[str]:
    commands = []
    for block in _BASH_BLOCK.findall(_readme_text()):
        for command in block.replace("\\\n", " ").splitlines():
            command = command.strip()
            if command.startswith("lerobot-wandb "):
                commands.append(" ".join(command.split()))
    return commands


def _readme_commands() -> list[str]:
    return [_expand_documented_values(command) for command in _raw_readme_commands()]


def test_the_readme_actually_contains_commands():
    """Guard the guard: a regex that silently matches nothing would make every test below vacuous."""
    commands = _readme_commands()
    assert len(commands) >= 3
    # The pipeline is only end-to-end if every stage that crosses machines is shown, promotion
    # included — until #24 it was the one step that dropped out of the CLI into an SDK snippet.
    joined = " ".join(commands)
    for expected in ("dataset upload", "model download", "rollout upload", "model promote"):
        assert expected in joined


def test_public_docs_point_to_the_canonical_companion_source():
    for path in (ROOT_README, README, JAPANESE_README, PACKAGE_README):
        text = path.read_text()
        assert CANONICAL_COMPANION_URL in text, path
        assert SOURCE_INSTALL in text, path
        assert OLD_SUBDIRECTORY_MARKER not in text, path

    assert SOURCE_INSTALL_WITH_LEROBOT in PACKAGE_README.read_text()
    assert SOURCE_INSTALL_WITH_LEROBOT in README.read_text()


def test_root_readme_describes_the_companion_boundary_and_fork_hooks():
    text = ROOT_README.read_text()
    fork_start = text.index("## W&B companion for upstream LeRobot")
    quick_start = text.index("## Quick Start")
    pypi_install = text.index("pip install lerobot", quick_start)

    assert fork_start < quick_start < pypi_install
    fork_section = text[fork_start:quick_start]
    assert "ordinary upstream LeRobot" in fork_section
    assert "native LeRobot plugin contract" in fork_section
    assert "package and release source" in fork_section
    assert "fork-only" in fork_section
    assert "--dataset.artifact_ref" in fork_section
    assert "training-time Artifact materialization" in fork_section
    assert "same-run final-model publication" in fork_section
    assert "lerobot-record" in fork_section
    assert "lerobot-rollout" in fork_section
    assert "uv sync --locked --extra core_scripts --extra feetech --extra training" in fork_section
    assert "source .venv/bin/activate" in fork_section
    assert "lerobot-wandb --help" in fork_section


def test_showcase_declares_and_reuses_operator_values():
    text = _readme_text()
    env = _documented_env()

    assert env["WANDB_ENTITY"] == "your-wandb-entity"
    assert env["WANDB_PROJECT"] == "so101-pick-cube"
    assert re.fullmatch(r"[^/]+/[^/]+/pick-cube-policy:v\d+", env["MODEL_REF"])
    assert env["MODEL_REF"].startswith(f"{env['WANDB_ENTITY']}/{env['WANDB_PROJECT']}/")
    assert env["EPISODES_SUCCEEDED"].isdigit()
    assert "source .venv/bin/activate" in text
    assert "my-team" not in text

    commands = " ".join(_raw_readme_commands())
    assert '--entity "$WANDB_ENTITY"' in commands
    assert '--project "$WANDB_PROJECT"' in commands
    assert "$WANDB_ENTITY/$WANDB_PROJECT" in commands
    assert '--model-ref "$MODEL_REF"' in commands
    assert '--ref "$MODEL_REF"' in commands
    assert '--episodes-succeeded "$EPISODES_SUCCEEDED"' in commands

    assert text.index('export WANDB_ENTITY="') < text.index("lerobot-wandb dataset upload")
    assert text.index('export MODEL_REF="') < text.index("--strategy.type=episodic")
    assert text.index('export EPISODES_SUCCEEDED="') < text.index("lerobot-wandb rollout upload")


@pytest.mark.parametrize("command", _readme_commands(), ids=lambda c: " ".join(c.split()[:3]))
def test_readme_command_parses_against_the_real_cli(command):
    args = cli.build_parser().parse_args(shlex.split(command)[1:])
    assert callable(args.func)


def test_fork_training_capability_gate_distinguishes_config_surfaces():
    upstream_dataset = make_dataclass("UpstreamDatasetConfig", [("repo_id", object)])
    upstream_wandb = make_dataclass("UpstreamWandBConfig", [("project", object)])
    fork_dataset = make_dataclass("ForkDatasetConfig", [(field, object) for field in _FORK_DATASET_FIELDS])
    fork_wandb = make_dataclass("ForkWandBConfig", [(field, object) for field in _FORK_WANDB_FIELDS])

    assert not _config_supports_fork_training_hooks(upstream_dataset, upstream_wandb)
    assert _config_supports_fork_training_hooks(fork_dataset, fork_wandb)


def _readme_train_command() -> list[str]:
    for block in _BASH_BLOCK.findall(_readme_text()):
        for command in block.replace("\\\n", " ").splitlines():
            if command.strip().startswith("lerobot-train "):
                expanded = _expand_documented_values(" ".join(command.split()))
                return shlex.split(expanded)[1:]
    raise AssertionError("the showcase README no longer shows a lerobot-train command")


def test_readme_train_command_parses_and_validates(tmp_path):
    """The training command is the one place the README drives *this effort's* config surface
    (`dataset.artifact_ref`, `wandb.model_artifact_name`, ...) rather than the standalone CLI, so it
    is parsed and validated rather than only eyeballed.
    """
    if not _fork_training_hooks_available():
        pytest.skip("fork-only training hooks are unavailable in this LeRobot environment")

    pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
    import draccus
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig  # noqa: F401  (registers act)

    # Only environment-dependent values may be substituted here. Anything that decides whether the
    # command is *valid* must not be: injecting `--policy.push_to_hub=false` to get a green test is
    # how the first version of this file hid a README command that could not run at all.
    args = [
        arg for arg in _readme_train_command() if not arg.startswith(("--output_dir=", "--policy.device="))
    ]
    args += [f"--output_dir={tmp_path / 'run'}", "--policy.device=cpu"]

    cfg = draccus.parse(TrainPipelineConfig, args=args)
    cfg.validate()

    assert cfg.dataset.artifact_ref == "your-wandb-entity/so101-pick-cube/pick-cube:raw"
    assert cfg.dataset.repo_id is None
    assert cfg.wandb.entity == "your-wandb-entity"
    assert cfg.wandb.project == "so101-pick-cube"
    assert cfg.wandb.model_artifact_name == "pick-cube-policy"
    assert cfg.wandb.registered_model_name == "pick-cube-policy"
    # The showcase promises W&B is the only remote store; `push_to_hub` defaults to True.
    assert cfg.policy.push_to_hub is False


def _readme_command_for(tool: str) -> list[str]:
    for block in _BASH_BLOCK.findall(_readme_text()):
        for command in block.replace("\\\n", " ").splitlines():
            if command.strip().startswith(f"{tool} "):
                expanded = _expand_documented_values(" ".join(command.split()))
                return shlex.split(expanded)[1:]
    raise AssertionError(f"the showcase README no longer shows a {tool} command")


def test_readme_record_command_parses():
    """`lerobot-record` is upstream, not built here, but a stale flag in our README misleads the
    reader just as badly. Parsed with plain draccus — it declares no `__get_path_fields__` values
    that the CLI wrapper would have to strip first.
    """
    pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
    import draccus
    from lerobot.scripts.lerobot_record import RecordConfig

    cfg = draccus.parse(RecordConfig, args=_readme_command_for("lerobot-record"))

    assert cfg.dataset.repo_id == "local/pick-cube"
    assert cfg.dataset.push_to_hub is False  # the showcase never touches the Hub


def test_readme_records_the_immutable_model_version_for_the_rollout():
    """The rollout upload must name the full version the robot actually ran, not an alias or a
    reconstructed ref: `cmd_rollout_upload` resolves the ref again at upload time, so an alias that
    moved in between would record a model the rollout never used — wrong, and authoritative-looking.
    """
    raw_upload = next(c for c in _raw_readme_commands() if c.startswith("lerobot-wandb rollout upload"))
    assert '--model-ref "$MODEL_REF"' in raw_upload

    upload = _expand_documented_values(raw_upload)
    model_ref = upload.split("--model-ref ", 1)[1].split()[0].strip('"')
    assert re.fullmatch(r"[^/]+/[^/]+/[^:]+:v\d+", model_ref), (
        f"--model-ref must pin a full immutable version, got {model_ref!r}"
    )


def test_readme_rollout_command_uses_a_rollout_prefixed_dataset_name():
    """`lerobot-rollout` resolves `--policy.path` through `parser.wrap` rather than plain draccus,
    so parsing it here would need a real checkpoint on disk. Check instead the one rule the rollout
    config enforces about the command shown, which a reader would otherwise hit at runtime.
    """
    args = _readme_command_for("lerobot-rollout")
    repo_id = next(a.split("=", 1)[1] for a in args if a.startswith("--dataset.repo_id="))

    assert repo_id.split("/", 1)[-1].startswith("rollout_")
    # `DatasetRecordConfig.push_to_hub` defaults to True and the episodic strategy's teardown acts
    # on it, so without this flag the rollout is published to the Hub behind the reader's back.
    assert "--dataset.push_to_hub=false" in args
    # The rollout upload command must point at the directory this command writes.
    root = next(a.split("=", 1)[1] for a in args if a.startswith("--dataset.root="))
    upload = next(c for c in _readme_commands() if c.startswith("lerobot-wandb rollout upload"))
    assert f"--root {root}" in upload
