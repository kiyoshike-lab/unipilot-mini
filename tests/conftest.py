"""Keep Campus evaluation writes inside pytest temporary directories.

Production constants remain unchanged outside the monkeypatch lifetime. The
session checksum guard detects future tests that bypass this isolation.
"""
import hashlib
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
OUTPUT_NAMES = ('OUTPUT_20', 'OUTPUT_100', 'CLOSE_ANALYSIS', 'CRITICAL_FAILURE', 'REVIEW_QUEUE', 'REPORT')
FIXTURE_NAMES = ('campus-ai-quality-20.json', 'campus-ai-quality-100.json',
                 'campus-v21-close-analysis.json', 'campus-v21-critical-failure.json',
                 'campus-ai-review-queue.json', 'campus-ai-quality-report.md')


def fixture_hashes():
    return {name: hashlib.sha256((REPO/'evaluation'/name).read_bytes()).hexdigest()
            for name in FIXTURE_NAMES}


@pytest.fixture(scope='session', autouse=True)
def repo_evaluation_files_unchanged():
    before = fixture_hashes()
    yield before
    after = fixture_hashes()
    assert after == before, 'pytest changed repository evaluation files: ' + repr(
        [name for name in before if before[name] != after[name]])


@pytest.fixture(autouse=True)
def isolated_campus_evaluation_outputs(tmp_path, monkeypatch):
    from evaluation import evaluate_campus_ai_quality as evaluator

    output = tmp_path/'campus-evaluation'
    output.mkdir()
    paths = {}
    for name in OUTPUT_NAMES:
        paths[name] = output / getattr(evaluator, name).name
        monkeypatch.setattr(evaluator, name, paths[name])
    return paths
