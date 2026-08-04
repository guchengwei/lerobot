#!/usr/bin/env python

import struct
from pathlib import Path

import yaml

SHOWCASE_DIR = Path(__file__).parents[3] / "examples" / "wandb_showcase"
ENGLISH_README_PATH = SHOWCASE_DIR / "README.md"
JAPANESE_README_PATH = SHOWCASE_DIR / "README.ja.md"
ENGLISH_DIAGRAM_PATH = SHOWCASE_DIR / "assets" / "wandb-workflow-overview-en.png"
JAPANESE_DIAGRAM_PATH = SHOWCASE_DIR / "assets" / "wandb-workflow-overview-ja.png"

TECHNICAL_TERMS = (
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
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data[12:16] == b"IHDR"
    return struct.unpack(">II", data[16:24])


def _shell_blocks(markdown: str) -> list[str]:
    return [part.split("```", 1)[0].strip() for part in markdown.split("```bash\n")[1:]]


def _commands(markdown: str) -> list[str]:
    return [block for block in _shell_blocks(markdown) if block.startswith(("uv ", "wandb ", "lerobot-"))]


def test_manuals_link_to_each_other_and_use_supplied_diagrams():
    english = _read(ENGLISH_README_PATH)
    japanese = _read(JAPANESE_README_PATH)

    assert "[日本語版](README.ja.md)" in english
    assert "[English](README.md)" in japanese
    assert "(assets/wandb-workflow-overview-en.png)" in english
    assert "(assets/wandb-workflow-overview-ja.png)" in japanese
    assert _png_dimensions(ENGLISH_DIAGRAM_PATH) == (1536, 1024)
    assert _png_dimensions(JAPANESE_DIAGRAM_PATH) == (1448, 1086)


def test_japanese_manual_keeps_cli_commands_and_technical_terms():
    english = _read(ENGLISH_README_PATH)
    japanese = _read(JAPANESE_README_PATH)

    assert _commands(japanese) == _commands(english)
    assert all(term in japanese for term in TECHNICAL_TERMS)
    assert any("\u3040" <= character <= "\u30ff" for character in japanese)
    assert "your-team" not in japanese
    assert "my-team" not in japanese

    rollout_upload = _commands(japanese)[-2]
    checkpoint_upload = _commands(japanese)[-3]
    promote = _commands(japanese)[-1]

    assert '--model-ref "$MODEL_REF"' in rollout_upload
    assert '--ref "$MODEL_REF"' in checkpoint_upload
    assert '--ref "$MODEL_REF"' in promote


def test_japanese_yaml_fragments_match_english():
    english = _read(ENGLISH_README_PATH)
    japanese = _read(JAPANESE_README_PATH)

    english_yaml = [yaml.safe_load(part.split("```", 1)[0]) for part in english.split("```yaml\n")[1:]]
    japanese_yaml = [yaml.safe_load(part.split("```", 1)[0]) for part in japanese.split("```yaml\n")[1:]]

    assert japanese_yaml == english_yaml
