"""CPU-only synthetic validation tests for the PHASE63L unarmed candidate."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("phase63l_binding", ROOT / "scripts" / "phase63l_binding.py")
assert SPEC and SPEC.loader
binding = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(binding)


def digest(ch: str) -> str:
    return ch * 64


def expected() -> dict:
    manifest = {role: digest("a") for role in binding.EXECUTION_FILES}
    return {"branch": "foundation-research", "implementation_source_commit": digest("b"),
            "source_manifest": manifest, "source_manifest_sha256": binding.canonical_sha256(manifest),
            "scientific_preregistration_sha256": digest("c"),
            "interpreter": {"sys_executable": "C:\\Python\\python.exe", "python_version": "3.test",
                            "torch_version": "test", "torch_cuda_build": "12.test", "cuda_api_calls": 0,
                            "actual_gpu_availability": "NOT_RECHECKED"},
            "checkpoint_root": "Z:\\AI\\unipilot-mini\\checkpoints", "data_path": "data/train.bin",
            "data_sha256": digest("d"), "tokenizer_sha256": digest("e"), "matching_cache_sha256": digest("f"),
            "runs": [{"ordinal": n, "seed": seed, "arm": arm, "runtime_lr": lr, "optimizer_updates": 122,
                      "parent_path": f"Z:\\parent-{seed}.pt", "parent_sha256": digest(letter),
                      "future_output_path": f"Z:\\AI\\unipilot-mini\\checkpoints\\experimental\\phase63\\seed-{seed}\\{arm}\\checkpoint-tokens-16777216.pt"}
                     for n, (seed, arm, lr, letter) in enumerate(((42, "control", 5e-5, "1"), (42, "half-lr", 2.5e-5, "2"),
                        (123, "control", 5e-5, "3"), (123, "half-lr", 2.5e-5, "4"),
                        (2026, "control", 5e-5, "5"), (2026, "half-lr", 2.5e-5, "6")), 1)]}


def valid() -> tuple[dict, dict]:
    exp = expected()
    # The candidate is intentionally derived from expected values; isolate the
    # two test objects so mutation exercises validation rather than its oracle.
    return copy.deepcopy(binding.build_candidate(exp)), copy.deepcopy(exp)


def reject(mutator) -> None:
    candidate, exp = valid()
    mutator(candidate, exp)
    with pytest.raises(binding.ValidationError):
        binding.validate_candidate(candidate, exp)


def test_a_valid_unarmed_candidate_passes() -> None:
    candidate, exp = valid()
    assert binding.validate_candidate(candidate, exp)["pass"] is True


@pytest.mark.parametrize("case,mutator", [
    ("B_source_character", lambda c, e: c["source_manifest"].__setitem__("worker", digest("0"))),
    ("C_old_worker", lambda c, e: c["source_manifest"].__setitem__("worker", digest("9"))),
    ("D_old_supervisor", lambda c, e: c["source_manifest"].__setitem__("supervisor", digest("8"))),
    ("E_missing_guard", lambda c, e: c["source_manifest"].pop("gpu_execution_guard")),
    ("F_changed_validator", lambda c, e: c["source_manifest"].__setitem__("validator", digest("7"))),
    ("G_prereg_mismatch", lambda c, e: c["scientific_preregistration"].__setitem__("sha256", digest("0"))),
    ("H_parent_mismatch", lambda c, e: c["runs"][0].__setitem__("parent_sha256", digest("0"))),
    ("I_missing_run", lambda c, e: c["runs"].pop()),
    ("J_duplicate_run", lambda c, e: c["runs"].__setitem__(1, copy.deepcopy(c["runs"][0]))),
    ("K_lr_changed", lambda c, e: c["runs"][0].__setitem__("runtime_lr", 1e-4)),
    ("L_output_invalid", lambda c, e: c["runs"][0].__setitem__("future_output_path", "C:\\elsewhere.pt")),
    ("M_execution_armed", lambda c, e: c.__setitem__("execution_authorized", True)),
    ("N_no_train_armed", lambda c, e: c.__setitem__("no_train_gpu_authorized", True)),
    ("O_cooling_bypassed", lambda c, e: c["gates"].__setitem__("cooling", "PHASE63_COOLING_VERIFIED")),
    ("P_stale_binding_shape", lambda c, e: c.update({"schema_version": "phase63-execution-binding-v1", "execution_authorized": True})),
    ("Q_schema_mismatch", lambda c, e: c.__setitem__("schema_version", "wrong")),
    ("nan", lambda c, e: c["runs"][0].__setitem__("runtime_lr", math.nan)),
    ("inf", lambda c, e: c["runs"][0].__setitem__("runtime_lr", math.inf)),
])
def test_b_to_q_and_nonfinite_inputs_fail_closed(case, mutator) -> None:
    reject(mutator)


def test_r_tampered_file_hash_differs(tmp_path: Path) -> None:
    candidate, _ = valid()
    payload = binding.canonical_bytes(candidate)
    path = tmp_path / "candidate.json"
    path.write_bytes(payload)
    before = binding.sha256_file(path)
    path.write_bytes(payload.replace(b"CPU_PREPARATION_ONLY", b"CPU_PREPARATION_ONLY!"))
    assert binding.sha256_file(path) != before


def test_generated_candidate_is_strictly_unarmed_when_present(monkeypatch) -> None:
    candidate_path = ROOT / "evaluation" / "phase63" / "phase63l" / "execution-binding-candidate.json"
    if not candidate_path.exists():
        pytest.skip("PHASE63L candidate has not been generated")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    prereg = binding.read_json(binding.PREREG_PATH)
    expected_actual = binding.expected_contract(prereg, binding.source_manifest(), binding.interpreter_metadata())
    candidate = binding.read_json(candidate_path)
    # PHASE63M intentionally changes the runtime source after this historical
    # PHASE63L candidate.  It must remain unarmed, but is no longer reusable.
    if candidate["implementation_source_commit"] != expected_actual["implementation_source_commit"]:
        assert candidate["execution_authorized"] is False
        assert candidate["no_train_gpu_authorized"] is False
        assert candidate["training_authorized"] is False
        return
    assert binding.validate_candidate(candidate, expected_actual)["unarmed"] is True


def test_generator_source_has_no_gpu_entry_points() -> None:
    source = (ROOT / "scripts" / "phase63l_binding.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "cuda" and isinstance(node.value, ast.Name) and node.value.id == "torch"
        for node in ast.walk(tree)
    )
