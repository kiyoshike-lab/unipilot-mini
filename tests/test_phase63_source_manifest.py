"""CPU-only security tests for the PHASE63 repository-local source closure."""
from __future__ import annotations

from pathlib import Path

import pytest

from training import phase63_binding_contract as contract
from training.phase63_source_manifest import SourceDependencyError, execution_dependency_closure


ROOT = Path(__file__).parents[1]
PREREG = ROOT / "evaluation" / "phase62" / "phase63-clean-continuation-stability-preregistration.json"
ZROOT = Path(r"Z:\AI\unipilot-mini\checkpoints")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_current_execution_closure_is_deterministic_and_contains_all_mandatory_roots() -> None:
    first = contract.current_source_manifest(ROOT)
    second = contract.current_source_manifest(ROOT)
    assert first == second
    assert [entry["path"] for entry in first] == sorted(entry["path"] for entry in first)
    assert all(set(entry) == {"path", "bytes", "sha256"} for entry in first)
    paths = {entry["path"] for entry in first}
    assert set(contract.RUNTIME_SOURCE_FILES) <= paths
    assert {
        "foundation/base_tokenizer.py",
        "foundation/diagnostic_transformer_v17.py",
        "training/foundation_v31_objective.py",
        "training/optimizer.py",
        "training/run_foundation_v35_thermal_gate.py",
        "training/phase63_source_manifest.py",
    } <= paths


def test_omit_ordinary_dependency_and_thermal_guard_are_both_rejected(tmp_path: Path) -> None:
    from test_phase63_runtime_binding import fixture_binding, put

    ordinary = fixture_binding()
    ordinary["source_manifest"] = [
        entry for entry in ordinary["source_manifest"]
        if entry["path"] != "foundation/base_tokenizer.py"
    ]
    ordinary["source_manifest_sha256"] = contract.canonical_sha256(ordinary["source_manifest"])
    ordinary_path = tmp_path / "ordinary.json"; put(ordinary_path, ordinary)
    with pytest.raises(contract.BindingContractError, match="RUNTIME_SOURCE_MANIFEST_MISMATCH"):
        contract.validate_unarmed_binding(ordinary_path, checkpoint_root=ZROOT, preregistration_path=PREREG)

    thermal = fixture_binding()
    thermal["source_manifest"] = [
        entry for entry in thermal["source_manifest"]
        if entry["path"] != "training/phase63_thermal_guard.py"
    ]
    thermal["source_manifest_sha256"] = contract.canonical_sha256(thermal["source_manifest"])
    thermal_path = tmp_path / "thermal.json"; put(thermal_path, thermal)
    with pytest.raises(contract.BindingContractError, match="RUNTIME_SOURCE_MANIFEST_MISMATCH"):
        contract.validate_unarmed_binding(thermal_path, checkpoint_root=ZROOT, preregistration_path=PREREG)


def test_relative_imports_and_package_init_dependencies_are_closed(tmp_path: Path) -> None:
    _write(tmp_path / "pkg" / "__init__.py", "from .dep import VALUE\n")
    _write(tmp_path / "pkg" / "dep.py", "VALUE = 1\n")
    _write(tmp_path / "pkg" / "entry.py", "from .dep import VALUE\n")
    assert execution_dependency_closure(tmp_path, ("pkg/entry.py",)) == (
        "pkg/__init__.py", "pkg/dep.py", "pkg/entry.py",
    )


@pytest.mark.parametrize("body", [
    "__import__('local_target')\n",
    "import importlib\nimportlib.import_module('local_target')\n",
    "from importlib import import_module\nimport_module('local_target')\n",
    "exec('import local_target')\n",
])
def test_unresolved_dynamic_imports_fail_closed(tmp_path: Path, body: str) -> None:
    _write(tmp_path / "entry.py", body)
    with pytest.raises(SourceDependencyError, match="SOURCE_DEPENDENCY_UNRESOLVED:dynamic:entry.py"):
        execution_dependency_closure(tmp_path, ("entry.py",))
