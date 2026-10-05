import json
import shutil
from pathlib import Path

import nbformat
import pandas as pd
import pytest
from nbclient.exceptions import CellExecutionError, CellTimeoutError

from notebook_pipeline.runner import execute_pipeline

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.integration


@pytest.fixture
def workspace(tmp_path):
    shutil.copyfile(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    shutil.copytree(ROOT / "notebooks", tmp_path / "notebooks")
    shutil.copytree(ROOT / "data", tmp_path / "data")
    return tmp_path


def test_successful_three_kernel_pipeline(workspace):
    run = execute_pipeline(workspace)
    assert run.read()["status"] == "succeeded"
    assert len(list((run.path / "executed").glob("*.ipynb"))) == 3
    quality = json.loads((run.path / "artifacts/01_ingest/quality_report.json").read_text())
    assert quality["duplicates_removed"] == 4
    assert quality["invalid_rows_excluded"] == 3
    clean = pd.read_parquet(run.path / "artifacts/01_ingest/clean_sales.parquet")
    monthly = pd.read_parquet(run.path / "artifacts/02_analyze/monthly_sales.parquet")
    expected = (clean.quantity * clean.unit_price).sum()
    assert monthly.revenue.sum() == pytest.approx(expected)
    assert pd.isna(monthly.revenue_growth.iloc[0])
    assert (run.path / "artifacts/03_visualize/sales_dashboard.png").stat().st_size > 10_000
    for error_log in (run.path / "logs").glob("*.errors.jsonl"):
        assert error_log.read_text() == ""


@pytest.mark.parametrize(
    "failure,stage,error_type",
    [
        ("missing-file", "01_ingest", "ArtifactError"),
        ("schema", "01_ingest", "DataContractError"),
        ("unexpected", "02_analyze", "RuntimeError"),
    ],
)
def test_failure_stops_pipeline_and_preserves_traceback(workspace, failure, stage, error_type):
    with pytest.raises(CellExecutionError):
        execute_pipeline(workspace, demo_failure=failure)
    (run_path,) = (workspace / "runs").iterdir()
    manifest = json.loads((run_path / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["stages"][stage]["status"] == "failed"
    assert manifest["stages"]["03_visualize"]["status"] == "skipped"
    errors = [
        json.loads(line)
        for line in (run_path / f"logs/{stage}.errors.jsonl").read_text().splitlines()
    ]
    assert len(errors) == 1
    assert errors[0]["exception_type"] == error_type
    assert errors[0]["traceback"] and errors[0]["cell_id"]
    executed = nbformat.read(run_path / f"executed/{stage}.ipynb", as_version=4)
    assert any(
        output.output_type == "error"
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )
    assert not (run_path / "executed/03_visualize.ipynb").exists()


def test_timeout_before_setup_has_runner_diagnostics(workspace):
    path = workspace / "notebooks/01_ingest.ipynb"
    notebook = nbformat.read(path, as_version=4)
    notebook.cells.insert(0, nbformat.v4.new_code_cell("import time\ntime.sleep(10)"))
    nbformat.write(notebook, path)
    with pytest.raises(CellTimeoutError):
        execute_pipeline(workspace, timeout=1)
    (run_path,) = (workspace / "runs").iterdir()
    manifest = json.loads((run_path / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    entries = [
        json.loads(line)
        for line in (run_path / "logs/runner.errors.jsonl").read_text().splitlines()
    ]
    assert entries[0]["exception_type"] == "CellTimeoutError"
    assert entries[0]["traceback"] and entries[0]["cell_id"]
    assert (run_path / "executed/01_ingest.ipynb").exists()


def test_error_after_finish_in_same_cell_does_not_publish(workspace):
    path = workspace / "notebooks/01_ingest.ipynb"
    notebook = nbformat.read(path, as_version=4)
    notebook.cells[-1].source += '\nraise ValueError("failed after finish request")'
    nbformat.write(notebook, path)
    with pytest.raises(CellExecutionError):
        execute_pipeline(workspace)
    (run_path,) = (workspace / "runs").iterdir()
    manifest = json.loads((run_path / "manifest.json").read_text())
    assert manifest["stages"]["01_ingest"]["status"] == "failed"
    assert manifest["stages"]["01_ingest"]["artifacts"] == {}
    assert not (run_path / "executed/02_analyze.ipynb").exists()
