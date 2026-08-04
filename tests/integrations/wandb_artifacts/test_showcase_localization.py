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
ENGLISH_DIAGRAM = "https://github.com/user-attachments/assets/97eb417f-cd69-41db-ba78-7ad2503381b0"
JAPANESE_DIAGRAM = "https://github.com/user-attachments/assets/30906c5b-d7eb-400c-ab07-e2773fa1c7e5"
_BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.S)


def _bash_blocks(path: Path) -> list[str]:
    return [block.strip() for block in _BASH_BLOCK.findall(path.read_text())]


def test_manuals_link_to_each_other_and_render_supplied_diagrams() -> None:
    english = ENGLISH.read_text()
    japanese = JAPANESE.read_text()

    assert "README.ja.md" in english
    assert "README.md" in japanese
    assert f"]({ENGLISH_DIAGRAM})" in english
    assert f"]({JAPANESE_DIAGRAM})" in japanese
    assert ENGLISH_DIAGRAM != JAPANESE_DIAGRAM


def test_localized_manual_uses_the_same_commands_as_english() -> None:
    assert _bash_blocks(JAPANESE) == _bash_blocks(ENGLISH)


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
