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
"""Keep the English and Japanese W&B walkthroughs operationally equivalent."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]
SHOWCASE = REPO_ROOT / "examples" / "wandb_showcase"
ENGLISH = SHOWCASE / "README.md"
JAPANESE = SHOWCASE / "README.ja.md"
ENGLISH_DIAGRAM = SHOWCASE / "assets" / "wandb-workflow-overview-en.jpg"
JAPANESE_DIAGRAM = SHOWCASE / "assets" / "wandb-workflow-overview-ja.jpg"
CANONICAL_COMPANION_URL = "https://github.com/guchengwei/lerobot-wandb"
CANONICAL_EN_MANUAL_URL = "https://github.com/guchengwei/lerobot-wandb/blob/main/MANUAL.md"
CANONICAL_JA_MANUAL_URL = "https://github.com/guchengwei/lerobot-wandb/blob/main/MANUAL.ja.md"
SOURCE_INSTALL = 'pip install "lerobot-wandb @ git+https://github.com/guchengwei/lerobot-wandb.git@main"'
OLD_SUBDIRECTORY_MARKER = "subdirectory=" + "packages/lerobot-wandb"
_BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.S)


def _bash_blocks(path: Path) -> list[str]:
    return [block.strip() for block in _BASH_BLOCK.findall(path.read_text())]


def test_manuals_link_to_each_other_and_render_local_diagrams() -> None:
    english = ENGLISH.read_text()
    japanese = JAPANESE.read_text()

    assert "README.ja.md" in english
    assert "README.md" in japanese
    assert "./assets/wandb-workflow-overview-en.jpg" in english
    assert "./assets/wandb-workflow-overview-ja.jpg" in japanese

    for diagram in (ENGLISH_DIAGRAM, JAPANESE_DIAGRAM):
        data = diagram.read_bytes()
        assert data.startswith(b"\xff\xd8\xff")
        assert data.endswith(b"\xff\xd9")
        assert 5_000 < diagram.stat().st_size <= 1024 * 1024


def test_localized_manual_uses_the_same_commands_as_english() -> None:
    assert _bash_blocks(JAPANESE) == _bash_blocks(ENGLISH)


def test_legacy_manuals_point_to_the_canonical_companion_and_mark_fork_hooks() -> None:
    english = ENGLISH.read_text()
    japanese = JAPANESE.read_text()

    assert CANONICAL_COMPANION_URL in english
    assert CANONICAL_COMPANION_URL in japanese
    for text in (english, japanese):
        assert CANONICAL_EN_MANUAL_URL in text
        assert CANONICAL_JA_MANUAL_URL in text
        assert ">=0.6.1,<0.6.2" in text
        assert ">=0.6.1,<0.7.0" not in text
    assert SOURCE_INSTALL in english
    assert SOURCE_INSTALL in japanese
    assert OLD_SUBDIRECTORY_MARKER not in english
    assert OLD_SUBDIRECTORY_MARKER not in japanese

    assert "legacy" in english.lower()
    assert "レガシー" in japanese
    for text, fork_marker in ((english, "fork-only"), (japanese, "fork 専用")):
        assert fork_marker in text
        assert "--dataset.artifact_ref" in text
        assert "lerobot-record" in text
        assert "lerobot-rollout" in text
        assert "wandb.model_artifact_name" in text
        assert "wandb.registered_model_name" in text


def test_manuals_explain_the_ecosystem_image_boundary() -> None:
    english = ENGLISH.read_text()
    japanese = JAPANESE.read_text()

    assert "broader LeRobot × W&B ecosystem overview" in english
    assert "not the companion contract" in english
    for marker in (
        "Auto-Upload",
        "W&B SDK (Streaming)",
        "Deploy / Inference",
        "Closed-Loop Control",
        "all data, models, and results",
        "explicit Artifact transfer and promotion",
        "`lerobot-train`",
        "does not automatically record data",
        "does not take over the robot control loop",
    ):
        assert marker in english

    assert "LeRobot × W&B ecosystem の広い全体像" in japanese
    assert "companion contract ではありません" in japanese
    for marker in (
        "Auto-Upload",
        "W&B SDK (Streaming)",
        "Deploy / Inference",
        "Closed-Loop Control",
        "すべてのデータ、model、結果",
        "明示的に行う Artifact transfer と promotion",
        "`lerobot-train`",
        "data の自動記録",
        "loop の引き取り",
    ):
        assert marker in japanese


def test_localized_manuals_keep_fork_publication_at_training_step() -> None:
    english = ENGLISH.read_text()
    japanese = JAPANESE.read_text()

    assert "training step (§3), including same-run final-model publication" in english
    assert "same-run final-model publication (§7)" not in english
    assert "training step（§3）" in japanese
    assert "同じ Run からの final-model publication もここで" in japanese
    assert "行います。" in japanese
    assert "final-model publication（§7）" not in japanese


def test_japanese_manual_keeps_product_and_runtime_terms_in_english() -> None:
    japanese = JAPANESE.read_text()
    for term in (
        "Artifact",
        "Run",
        "Registry",
        "alias",
        "lineage",
        "rollout",
        "teleoperation",
        "checkpoint",
        "policy",
        "remote store",
        "runtime cache",
        "control loop",
        "immutable",
    ):
        assert term in japanese
