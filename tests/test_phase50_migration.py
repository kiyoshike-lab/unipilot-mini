from pathlib import Path
import pytest
from scripts.migrate_phase50_checkpoints import copy_exclusive, relative_record, metadata_expectation


def test_copy_is_exclusive_and_preserves_source(tmp_path):
    source=tmp_path/'source.bin';destination=tmp_path/'nested'/'destination.bin'
    source.write_bytes(b'unchanged source')
    copy_exclusive(source,destination)
    assert source.read_bytes()==destination.read_bytes()==b'unchanged source'
    with pytest.raises(FileExistsError): copy_exclusive(source,destination)
    assert destination.read_bytes()==b'unchanged source'


@pytest.mark.parametrize('value', ['../checkpoint.pt', 'checkpoints/../other.pt', 'other/checkpoint.pt'])
def test_migration_rejects_unsafe_records(value):
    with pytest.raises(ValueError): relative_record(value)


def test_migration_metadata_is_derived_from_registered_path():
    p=relative_record('checkpoints/experimental/phase49/arm-B/seed-123/checkpoint-tokens-16384000.pt')
    assert metadata_expectation(p)==(123,16384000,7.5e-5)
