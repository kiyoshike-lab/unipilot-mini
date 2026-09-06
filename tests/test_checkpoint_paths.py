from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from training.checkpoint_paths import (
    CHECKPOINT_ROOT_ENV,
    checkpoint_path,
    checkpoint_root,
    display_path,
    ensure_checkpoint_storage,
    existing_checkpoint_path,
    resolve_checkpoint_output_dir,
)


def test_default_checkpoint_root_is_project_checkpoints(monkeypatch, tmp_path):
    monkeypatch.delenv(CHECKPOINT_ROOT_ENV, raising=False)

    assert checkpoint_root(tmp_path) == tmp_path / "checkpoints"
    assert checkpoint_path(tmp_path, "run", "checkpoint.pt") == tmp_path / "checkpoints" / "run" / "checkpoint.pt"


def test_env_checkpoint_root_routes_new_checkpoints(monkeypatch, tmp_path):
    external = tmp_path / "external-checkpoints"
    monkeypatch.setenv(CHECKPOINT_ROOT_ENV, str(external))

    assert checkpoint_root(tmp_path) == external
    assert checkpoint_path(tmp_path, "phase", "checkpoint.pt") == external / "phase" / "checkpoint.pt"


def test_relative_checkpoints_output_dir_routes_to_env_root(monkeypatch, tmp_path):
    external = tmp_path / "external-checkpoints"
    monkeypatch.setenv(CHECKPOINT_ROOT_ENV, str(external))

    assert resolve_checkpoint_output_dir(tmp_path, "checkpoints/smoke") == external / "smoke"
    assert resolve_checkpoint_output_dir(tmp_path, "other-output") == tmp_path / "other-output"


def test_checkpoint_save_smoke_to_env_root(monkeypatch, tmp_path):
    external = tmp_path / "external-checkpoints"
    target = external / "phase" / "checkpoint.pt"
    monkeypatch.setenv(CHECKPOINT_ROOT_ENV, str(external))

    routed = checkpoint_path(tmp_path, "phase", "checkpoint.pt", create_parent=True)
    ensure_checkpoint_storage(routed)
    torch.save({"ok": True}, routed)

    assert routed == target
    assert torch.load(target, map_location="cpu", weights_only=False) == {"ok": True}


def test_existing_checkpoint_prefers_external_but_preserves_local_fallback(monkeypatch, tmp_path):
    local = tmp_path / "checkpoints" / "run" / "checkpoint.pt"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"local")
    external = tmp_path / "external-checkpoints"
    monkeypatch.setenv(CHECKPOINT_ROOT_ENV, str(external))

    assert existing_checkpoint_path(tmp_path, "run", "checkpoint.pt") == local

    external_file = external / "run" / "checkpoint.pt"
    external_file.parent.mkdir(parents=True)
    external_file.write_bytes(b"external")

    assert existing_checkpoint_path(tmp_path, "run", "checkpoint.pt") == external_file


def test_display_path_handles_external_checkpoint_root(tmp_path):
    external = tmp_path.parent / "outside-checkpoints" / "checkpoint.pt"

    assert display_path(tmp_path, tmp_path / "checkpoints" / "a.pt") == "checkpoints/a.pt"
    assert display_path(tmp_path, external) == str(external.resolve())


@pytest.mark.skipif(os.name != "nt", reason="Windows drive-letter behavior only")
def test_unavailable_drive_fails_before_silent_c_fallback(monkeypatch, tmp_path):
    missing_drive = next(
        (letter for letter in "QRSTUVWXYZ" if not Path(f"{letter}:\\").exists()),
        None,
    )
    if missing_drive is None:
        pytest.skip("no missing drive letter available")
    monkeypatch.setenv(CHECKPOINT_ROOT_ENV, f"{missing_drive}:\\AI\\unipilot-mini\\checkpoints")

    target = checkpoint_path(tmp_path, "phase", "checkpoint.pt")
    with pytest.raises((OSError, RuntimeError)):
        ensure_checkpoint_storage(target)
