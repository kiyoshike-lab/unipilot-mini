"""Exercise actual evaluator writes, including an interrupted evaluation."""
import hashlib
from pathlib import Path

import pytest
from evaluation import evaluate_campus_ai_quality as evaluator


NAMES = ('OUTPUT_20', 'OUTPUT_100', 'CLOSE_ANALYSIS', 'CRITICAL_FAILURE', 'REVIEW_QUEUE', 'REPORT')


def repo_hashes():
    root = Path(__file__).resolve().parents[1]/'evaluation'
    filenames = ('campus-ai-quality-20.json','campus-ai-quality-100.json',
                 'campus-v21-close-analysis.json','campus-v21-critical-failure.json',
                 'campus-ai-review-queue.json','campus-ai-quality-report.md')
    return {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in filenames}


def test_real_evaluation_writes_only_to_temporary_paths(isolated_campus_evaluation_outputs, tmp_path):
    before = repo_hashes()
    evaluator.evaluate()
    for name in NAMES:
        path = getattr(evaluator, name)
        assert path.is_relative_to(tmp_path) and path.is_file()
    assert repo_hashes() == before


def test_interrupted_evaluation_preserves_repo_files(monkeypatch, isolated_campus_evaluation_outputs):
    before = repo_hashes()
    write = evaluator.write_json

    def fail_after_temporary_write(path, value):
        write(path, value)
        raise RuntimeError('synthetic interrupted evaluator')

    monkeypatch.setattr(evaluator, 'write_json', fail_after_temporary_write)
    with pytest.raises(RuntimeError, match='synthetic interrupted'):
        evaluator.evaluate()
    assert isolated_campus_evaluation_outputs['OUTPUT_20'].exists()
    assert repo_hashes() == before
