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
"""Dependency-neutral helpers for W&B artifact references used by core config."""

import re

_REF_PATTERN: re.Pattern[str] = re.compile(
    r"^(?P<entity>[^/:\s]+)/(?P<project>[^/:\s]+)/(?P<name>[^/:\s]+):(?P<version_or_alias>[^/:\s]+)$"
)


def artifact_collection_name(raw: str) -> str:
    """Return the collection name from a fully qualified W&B artifact reference.

    The validation intentionally mirrors the companion parser without importing the optional
    ``lerobot_wandb`` distribution into core configuration.
    """
    if not isinstance(raw, str):
        raise ValueError(f"Artifact reference must be a string, got {type(raw).__name__}.")

    match = _REF_PATTERN.fullmatch(raw)
    if match is None:
        raise ValueError(
            f"Invalid W&B artifact reference {raw!r}. Expected the form "
            "'entity/project/name:version' or 'entity/project/name:alias'."
        )

    return match.group("name")
