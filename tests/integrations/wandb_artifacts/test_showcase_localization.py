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
import struct
from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]
SHOWCASE = REPO_ROOT / "examples" / "wandb_showcase"
ENGLISH = SHOWCASE / "README.md"
JAPANESE = SHOWCASE / "README.ja.md"
ENGLISH_DIAGRAM = SHOWCASE / "assets" / "wandb-workflow-overview-en.png"
JAPANESE_DIAGRAM = SHOWCASE / "assets" / "wandb-workflow-overview-ja.png"
_BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.S)


def _bash_blocks(path: Path) -> list[str]:
    return [block.strip() for block in _BASH_BLOCK.findall(path.read_text())]


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data[12:16] == b"IHDR"
    return struct.unpack(">II", data[16:24])


def test_manuals_link_to_each_other_and_render_local_diagrams() -> None:
    english = ENGLISH.read_text()
    japanese = JAPANESE.read_text()

    assert "README.ja.md" in english
    assert "README.md" in japanese
    assert "./assets/wandb-workflow-overview-en.png" in english
    assert "./assets/wandb-workflow-overview-ja.png" in japanese
    assert _png_dimensions(ENGLISH_DIAGRAM) == (1536, 1024)
    assert _png_dimensions(JAPANESE_DIAGRAM) == (1448, 1086)


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
