from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path


CHECKPOINT_ROOT_ENV = "UNIPILOT_CHECKPOINT_ROOT"
WARNING_FREE_BYTES = 20 * 1024**3
MINIMUM_FREE_BYTES = 10 * 1024**3


def checkpoint_root(project_root: Path, *, create: bool = False) -> Path:
    configured = os.getenv(CHECKPOINT_ROOT_ENV)
    root = Path(configured).expanduser() if configured else project_root / "checkpoints"
    root = root.resolve()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def checkpoint_path(project_root: Path, *parts: str, create_parent: bool = False) -> Path:
    path = checkpoint_root(project_root, create=create_parent) / Path(*parts)
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_checkpoint_output_dir(project_root: Path, output_dir: str | Path) -> Path:
    path = Path(output_dir).expanduser()
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "checkpoints":
        return checkpoint_root(project_root) / Path(*parts[1:])
    return project_root / path


def existing_checkpoint_path(project_root: Path, *parts: str) -> Path:
    configured = os.getenv(CHECKPOINT_ROOT_ENV)
    external = (Path(configured).expanduser().resolve() / Path(*parts)) if configured else None
    local = project_root / "checkpoints" / Path(*parts)
    if external is not None and external.exists():
        return external
    return local


def display_path(project_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(project_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(resolved)


def ensure_checkpoint_storage(path: Path, *, minimum_free_bytes: int = MINIMUM_FREE_BYTES) -> dict:
    target_dir = path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target_dir)
    if usage.free < minimum_free_bytes:
        raise RuntimeError(
            f"checkpoint destination has only {usage.free / 1024**3:.2f} GiB free: {target_dir}. "
            f"Refusing to write checkpoint; free at least {minimum_free_bytes / 1024**3:.0f} GiB "
            f"or set {CHECKPOINT_ROOT_ENV} to a larger drive."
        )
    warning = usage.free < WARNING_FREE_BYTES
    if warning:
        warnings.warn(
            f"checkpoint destination has only {usage.free / 1024**3:.2f} GiB free: {target_dir}",
            RuntimeWarning,
            stacklevel=2,
        )
    return {
        "path": str(target_dir),
        "free_gib": usage.free / 1024**3,
        "warning": warning,
    }
