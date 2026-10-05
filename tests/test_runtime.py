import json
from types import SimpleNamespace

import pytest

from notebook_pipeline.errors import ArtifactError, RunStateError
from notebook_pipeline.runtime import REQUIRED_OUTPUTS, NotebookSession, Run, atomic_path


def publish(session):
    for name in REQUIRED_OUTPUTS[session.stage]:
        with session.output(name) as path:
            path.write_text("fixture", encoding="utf-8")
    session.finish()


def test_new_runs_are_unique_and_downstream_requires_success(tmp_path):
    first, second = Run.create(tmp_path), Run.create(tmp_path)
    assert first.id != second.id
    with pytest.raises(RunStateError, match="Complete all notebooks"):
        first.begin("02_analyze")
    first.begin("01_ingest")
    with pytest.raises(RunStateError, match="already started"):
        first.begin("01_ingest")


def test_publication_verifies_integrity_and_blocks_reruns(tmp_path):
    run = Run.create(tmp_path)
    run.begin("01_ingest")
    upstream = NotebookSession(run, "01_ingest")
    publish(upstream)
    run.begin("02_analyze")
    downstream = NotebookSession(run, "02_analyze")
    try:
        source = downstream.input("01_ingest", "clean_sales.parquet")
        source.write_text("modified", encoding="utf-8")
        with pytest.raises(ArtifactError, match="modified"):
            downstream.input("01_ingest", "clean_sales.parquet")
        with pytest.raises(RunStateError):
            with upstream.output("clean_sales.parquet"):
                pass
    finally:
        downstream.close()


def test_atomic_failure_preserves_previous_artifact(tmp_path):
    destination = tmp_path / "data.txt"
    destination.write_text("original")
    with pytest.raises(ValueError):
        with atomic_path(destination) as temporary:
            temporary.write_text("partial")
            raise ValueError("writer failed")
    assert destination.read_text() == "original"
    assert list(tmp_path.iterdir()) == [destination]


def test_cell_failure_records_chain_once_and_prevents_publication(tmp_path):
    run = Run.create(tmp_path)
    run.begin("01_ingest")
    session = NotebookSession(run, "01_ingest")
    session.before_cell(SimpleNamespace(raw_cell="# step: load\nx = 1", cell_id="cell-7"))
    try:
        try:
            raise FileNotFoundError("fixture absent")
        except FileNotFoundError as cause:
            raise ArtifactError("cannot read fixture") from cause
    except ArtifactError as error:
        session.after_cell(
            SimpleNamespace(execution_count=7, error_before_exec=None, error_in_exec=error)
        )
    with pytest.raises(RunStateError):
        session.finish()
    session.close()
    entries = [
        json.loads(line)
        for line in (run.path / "logs/01_ingest.errors.jsonl").read_text().splitlines()
    ]
    assert len(entries) == 1
    assert entries[0]["cell_id"] == "cell-7"
    assert entries[0]["step"] == "load"
    assert entries[0]["timestamp"].endswith("Z")
    assert "FileNotFoundError" in entries[0]["traceback"]
    assert entries[0]["exception_type"] == "ArtifactError"
    assert run.read()["stages"]["02_analyze"]["status"] == "skipped"


def test_finish_cannot_publish_missing_outputs(tmp_path):
    run = Run.create(tmp_path)
    run.begin("01_ingest")
    session = NotebookSession(run, "01_ingest")
    try:
        with pytest.raises(ArtifactError, match="missing outputs"):
            session.finish()
        assert run.read()["stages"]["01_ingest"]["status"] == "running"
    finally:
        session.close()


def test_syntax_error_is_logged(tmp_path):
    run = Run.create(tmp_path)
    run.begin("01_ingest")
    session = NotebookSession(run, "01_ingest")
    session.after_cell(
        SimpleNamespace(
            execution_count=2, error_before_exec=SyntaxError("invalid syntax"), error_in_exec=None
        )
    )
    session.close()
    assert run.read()["status"] == "failed"
    assert "SyntaxError" in (run.path / "logs/01_ingest.errors.jsonl").read_text()
