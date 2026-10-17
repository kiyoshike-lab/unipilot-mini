"""Fail-closed repository-local source closure for PHASE63 execution.

This module is intentionally CPU-only.  It performs no runtime imports of the
execution graph: the graph is derived from Python ASTs and every unresolved
repository-local dynamic import rejects closure generation.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Iterable


class SourceDependencyError(RuntimeError):
    """A repository-local execution dependency could not be bound safely."""


MANDATORY_EXECUTION_ROOTS = (
    "training/phase63_worker.py",
    "training/phase63_thermal_guard.py",
    "scripts/run_phase63_study.py",
    "training/exclusive_cuda_runner.py",
    "training/gpu_execution_lock.py",
    "training/gpu_execution_guard.py",
    "training/checkpoint_paths.py",
    "training/run_foundation_v36_lr_review.py",
    "training/phase63_binding_contract.py",
)


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _module_candidates(root: Path, parts: list[str]) -> list[Path]:
    base = root.joinpath(*parts)
    return [base.with_suffix(".py"), base / "__init__.py"]


def _package_inits(root: Path, parts: list[str]) -> list[Path]:
    result: list[Path] = []
    for index in range(1, len(parts)):
        candidate = root.joinpath(*parts[:index], "__init__.py")
        if candidate.is_file():
            result.append(candidate)
    return result


def _source_package_parts(relative: str) -> list[str]:
    path = Path(relative)
    if path.name == "__init__.py":
        return list(path.parts[:-1])
    return list(path.parts[:-1])


def _resolve_module(root: Path, module: str | None, level: int, current: str) -> list[Path]:
    package = _source_package_parts(current)
    if level:
        if level > len(package) + 1:
            raise SourceDependencyError("SOURCE_DEPENDENCY_UNRESOLVED:relative:" + current)
        parts = package[:len(package) - level + 1]
    else:
        parts = []
    if module:
        parts.extend(module.split("."))
    if not parts:
        raise SourceDependencyError("SOURCE_DEPENDENCY_UNRESOLVED:empty-module:" + current)
    candidates = _module_candidates(root, parts)
    found = [path for path in candidates if path.is_file()]
    if found:
        return _package_inits(root, parts) + found
    top = root / parts[0]
    if level or top.is_dir() or top.with_suffix(".py").is_file():
        raise SourceDependencyError("SOURCE_DEPENDENCY_UNRESOLVED:" + ".".join(parts))
    return []


def _attribute_root_name(value: ast.AST) -> str | None:
    while isinstance(value, ast.Attribute):
        value = value.value
    return value.id if isinstance(value, ast.Name) else None


def _dynamic_imports(tree: ast.AST, relative: str) -> None:
    importlib_names = {"importlib"}
    dynamic_names = {"__import__", "exec", "eval", "SourceFileLoader"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if node.module in {"importlib", "importlib.util", "importlib.machinery"}:
                    dynamic_names.add(alias.asname or alias.name)
                if node.module == "builtins" and alias.name in {"__import__", "exec", "eval"}:
                    dynamic_names.add(alias.asname or alias.name)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in dynamic_names:
            raise SourceDependencyError("SOURCE_DEPENDENCY_UNRESOLVED:dynamic:" + relative)
        if isinstance(func, ast.Attribute):
            if _attribute_root_name(func.value) in importlib_names and func.attr in {
                "import_module", "spec_from_file_location", "module_from_spec",
            }:
                raise SourceDependencyError("SOURCE_DEPENDENCY_UNRESOLVED:dynamic:" + relative)


def _direct_dependencies(root: Path, relative: str) -> set[str]:
    path = root / relative
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    except (OSError, SyntaxError) as exc:
        raise SourceDependencyError("SOURCE_DEPENDENCY_UNRESOLVED:parse:" + relative) from exc
    _dynamic_imports(tree, relative)
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for candidate in _resolve_module(root, alias.name, 0, relative):
                    result.add(_relative(candidate, root))
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_module(root, node.module, node.level, relative)
            for candidate in resolved:
                result.add(_relative(candidate, root))
            if node.module is not None:
                module_parts = node.module.split(".")
                package = _source_package_parts(relative)
                prefix = package[:len(package) - node.level + 1] if node.level else []
                for alias in node.names:
                    for candidate in _module_candidates(root, prefix + module_parts + [alias.name]):
                        if candidate.is_file():
                            result.add(_relative(candidate, root))
    return result


def execution_dependency_closure(root: Path, roots: Iterable[str] = MANDATORY_EXECUTION_ROOTS) -> tuple[str, ...]:
    root = Path(root).resolve()
    pending = sorted(set(roots))
    discovered: set[str] = set()
    while pending:
        relative = pending.pop(0)
        if relative in discovered:
            continue
        path = root / relative
        if not path.is_file():
            raise SourceDependencyError("SOURCE_DEPENDENCY_UNRESOLVED:missing:" + relative)
        discovered.add(relative)
        pending.extend(sorted(_direct_dependencies(root, relative) - discovered))
    return tuple(sorted(discovered))


def source_manifest(root: Path, roots: Iterable[str] = MANDATORY_EXECUTION_ROOTS) -> list[dict[str, int | str]]:
    root = Path(root).resolve()
    entries: list[dict[str, int | str]] = []
    for relative in execution_dependency_closure(root, roots):
        raw = (root / relative).read_bytes()
        entries.append({"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    return entries
