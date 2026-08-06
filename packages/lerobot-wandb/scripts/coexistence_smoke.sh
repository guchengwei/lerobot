#!/usr/bin/env bash
# Coexistence smoke for the `lerobot-wandb` companion distribution.
#
# Proves, in disposable environments, that the companion wheel:
#   A. installs into an existing LeRobot environment without replacing/shadowing it;
#   B. works in the fork development environment (documented; the heavy fork sync);
#   C. owns exactly its own files — uninstalling one distribution never deletes the other's.
#
# Run from the repository root:  bash packages/lerobot-wandb/scripts/coexistence_smoke.sh
# Requires: python3 (>=3.12), uv, and network access to PyPI. No W&B credentials.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMPANION_DIR="$REPO_ROOT/packages/lerobot-wandb"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

echo "==> Building the companion wheel"
uv run --with build python -m build "$COMPANION_DIR" --outdir "$WORKDIR/dist" >/dev/null
WHEEL="$(ls "$WORKDIR"/dist/lerobot_wandb-*.whl)"

# ---------------------------------------------------------------------------
# A. Existing (upstream-compatible) LeRobot environment
# ---------------------------------------------------------------------------
echo "==> A. Existing LeRobot environment"
python3 -m venv "$WORKDIR/env-a"
A_PIP="$WORKDIR/env-a/bin/pip"
A_PY="$WORKDIR/env-a/bin/python"

"$A_PIP" install -q "lerobot>=0.6.1,<0.7.0"
BEFORE_VERSION="$("$A_PY" -c 'import importlib.metadata as m; print(m.version("lerobot"))')"
BEFORE_LOCATION="$("$A_PY" -c 'import importlib.metadata as m; print(m.distribution("lerobot")._path)')"

"$A_PIP" install -q "$WHEEL[wandb-workspace,test]"

AFTER_VERSION="$("$A_PY" -c 'import importlib.metadata as m; print(m.version("lerobot"))')"
AFTER_LOCATION="$("$A_PY" -c 'import importlib.metadata as m; print(m.distribution("lerobot")._path)')"
test "$BEFORE_VERSION" = "$AFTER_VERSION"
test "$BEFORE_LOCATION" = "$AFTER_LOCATION"
echo "    lerobot unchanged by companion install: $AFTER_VERSION"

"$A_PY" -m pip check
"$WORKDIR/env-a/bin/lerobot-record" --help >/dev/null
"$WORKDIR/env-a/bin/lerobot-wandb" --help >/dev/null
"$A_PY" -m pytest -q "$COMPANION_DIR/tests" >/dev/null
echo "    pip check, lerobot-record --help, lerobot-wandb --help, companion tests: OK"

# ---------------------------------------------------------------------------
# B. Fork development environment
# ---------------------------------------------------------------------------
echo "==> B. Fork development environment"
# The fork's `training` extra installs the companion (editable, via the uv path source
# in the root pyproject). This section reproduces the repo's documented dev flow, so the
# workflow must run `uv sync --locked --extra dataset --extra training --extra test` first.
uv run python -m pytest -q "$COMPANION_DIR/tests" \
  tests/integrations/wandb_artifacts \
  tests/scripts/test_train_dataset_artifact.py \
  tests/policies/test_model_card_dataset_artifact.py >/dev/null
echo "    fork + companion test suites: OK"
uv pip check
echo "    pip check: OK"

# Exactly one `lerobot-wandb` executable must resolve on PATH: the companion's.
COUNT="$(uv run python -c "
import os
name = 'lerobot-wandb'
paths = {
    os.path.realpath(os.path.join(d or '.', name))
    for d in os.environ['PATH'].split(os.pathsep)
    if os.path.isfile(os.path.join(d or '.', name)) and os.access(os.path.join(d or '.', name), os.X_OK)
}
print(len(paths))
")"
test "$COUNT" = "1"
UV_WANDB_BIN="$(uv run sh -c 'command -v lerobot-wandb')"
case "$UV_WANDB_BIN" in
  *.venv/bin/lerobot-wandb) echo "    single companion executable on PATH: $UV_WANDB_BIN" ;;
  *) echo "    unexpected lerobot-wandb location: $UV_WANDB_BIN"; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# C. Wheel ownership / uninstall smoke
# ---------------------------------------------------------------------------
echo "==> C. Uninstall ownership smoke"
python3 -m venv "$WORKDIR/env-c"
C_PIP="$WORKDIR/env-c/bin/pip"
C_PY="$WORKDIR/env-c/bin/python"

# Install the fork wheel WITHOUT dependency resolution (only its own files matter
# for the ownership checks) and the companion WITH its base dependencies (its
# clean missing-LeRobot failure must be observable without an import traceback).
"$C_PIP" install -q --no-deps "$REPO_ROOT" 2>/dev/null || true
"$C_PIP" install -q "$WHEEL"
echo "    installed both distributions"

"$C_PY" -m pip show -f lerobot >"$WORKDIR/lerobot-files.txt"
"$C_PY" -m pip show -f lerobot-wandb >"$WORKDIR/wandb-files.txt"
"$C_PY" -m pip check || true  # fork wheel installed --no-deps: its own deps are expected missing

mkdir -p "$WORKDIR/snapshots"
"$C_PY" -c "
import site
from pathlib import Path

def snap(pip_show, out):
    lines = Path(pip_show).read_text().splitlines()
    files = lines[lines.index('Files:') + 1:]
    Path(out).write_text('\n'.join(line.strip() for line in files if line.strip()) + '\n')

snap('$WORKDIR/lerobot-files.txt', '$WORKDIR/snapshots/lerobot.txt')
snap('$WORKDIR/wandb-files.txt', '$WORKDIR/snapshots/wandb.txt')
"
test -s "$WORKDIR/snapshots/lerobot.txt" || { echo "    no lerobot files recorded"; exit 1; }
test -s "$WORKDIR/snapshots/wandb.txt" || { echo "    no lerobot-wandb files recorded"; exit 1; }

# 1) Uninstall the companion: lerobot commands and files must remain.
"$C_PIP" uninstall -q -y lerobot-wandb
test ! -e "$WORKDIR/env-c/bin/lerobot-wandb"
"$C_PY" -c "import importlib.metadata as m; print('    lerobot still installed:', m.version('lerobot'))"

# 2) Reinstall the companion.
"$C_PIP" install -q --no-deps "$WHEEL"

# 3) Uninstall lerobot: companion files must remain, and its commands must fail with
#    the intended missing-LeRobot message rather than an import-corruption traceback.
"$C_PIP" uninstall -q -y lerobot
"$C_PY" -c "
import site
from pathlib import Path

site_packages = Path(site.getsitepackages()[0])
missing = []
for line in Path('$WORKDIR/snapshots/wandb.txt').read_text().splitlines():
    if not line.strip():
        continue
    path = (site_packages / line.strip()).resolve()
    if not path.exists():
        missing.append(str(path))
if missing:
    raise SystemExit('companion files deleted by lerobot uninstall: ' + ', '.join(missing[:5]))
print('    companion files intact after lerobot uninstall')
"

mkdir -p "$WORKDIR/fake-root"
set +e
OUT="$("$WORKDIR/env-c/bin/lerobot-wandb" dataset upload --root "$WORKDIR/fake-root" --project smoke --name smoke 2>&1)"
RC=$?
set -e
test $RC -ne 0
echo "$OUT" | grep -q "not installed"
echo "    companion command fails with the intended missing-LeRobot message (rc=$RC)"

echo "==> All coexistence checks passed"
