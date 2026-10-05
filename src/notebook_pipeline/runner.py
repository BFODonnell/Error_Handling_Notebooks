"""Execute fresh kernels in order, preserving failed notebooks for diagnosis."""

import argparse
import json
import sys
from pathlib import Path

import nbformat
from jupyter_client import AsyncKernelManager
from nbclient import NotebookClient

from .errors import RunStateError
from .observability import close_logger, make_logger
from .runtime import STAGES, Run, atomic_path, project_root, write_json


def execute_pipeline(
    root: Path,
    *,
    input_csv: str | None = None,
    demo_failure: str | None = None,
    timeout: int = 120,
) -> Run:
    root = root.resolve()
    run = Run.create(root)
    context = {"run_id": run.id, "notebook": "runner", "cell_id": None}
    log = make_logger(run.path / "logs", "runner", context)
    print(f"Run: {run.id}\nResults: {run.path}", flush=True)
    log.info("pipeline.started", extra={"fields": {"python": sys.version.split()[0]}})
    write_json(
        run.path / "parameters.json",
        {"input_csv": input_csv, "demo_failure": demo_failure, "timeout_seconds": timeout},
    )
    stage = STAGES[0]
    try:
        for stage in STAGES:
            context.update(notebook=stage, cell_id=None)
            notebook = nbformat.read(root / "notebooks" / f"{stage}.ipynb", as_version=4)
            parameters = next(
                index
                for index, cell in enumerate(notebook.cells)
                if "parameters" in cell.metadata.get("tags", [])
            )
            notebook.cells.insert(
                parameters + 1,
                nbformat.v4.new_code_cell(
                    f"run_id = {run.id!r}\n"
                    f"input_csv = {input_csv!r}\ndemo_failure = {demo_failure!r}",
                    metadata={"tags": ["injected-parameters"]},
                ),
            )
            # Pin the kernel to the interpreter launched by uv; no user-wide kernel install.
            manager = AsyncKernelManager(kernel_name="python3")
            manager.kernel_spec.argv = [
                sys.executable,
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}",
            ]

            def track_cell(cell, cell_index):
                context.update(cell_id=cell.get("id"), cell_index=cell_index)

            client = NotebookClient(
                notebook,
                km=manager,
                timeout=timeout,
                allow_errors=False,
                force_raise_errors=True,
                resources={"metadata": {"path": str(root)}},
                on_cell_execute=track_cell,
            )
            execution_error = None
            try:
                # Explicit cleanup is needed when supplying an external KernelManager.
                client.execute(cleanup_kc=True)
            except BaseException as error:
                execution_error = error
                raise
            finally:
                destination = run.path / "executed" / f"{stage}.ipynb"
                try:
                    with atomic_path(destination) as temporary:
                        nbformat.write(notebook, temporary)
                except Exception as save_error:
                    if execution_error is None:
                        raise
                    log.exception("notebook.save_failed")
                    execution_error.add_note(
                        f"Saving the diagnostic notebook also failed: {save_error}"
                    )
            if run.read()["stages"][stage]["status"] != "succeeded":
                raise RunStateError(f"{stage} did not publish successfully; check its error log.")
            log.info("pipeline.stage_succeeded")
        log.info("pipeline.succeeded")
        return run
    except BaseException as error:
        # One boundary handles cell errors, kernel crashes, timeouts and user interrupts.
        # A normal cell failure already has its original traceback in the stage log.
        recorded = run.read()["stages"][stage]["status"] == "failed"
        log.error(
            "pipeline.failed",
            exc_info=not recorded,
            extra={
                "fields": {
                    "exception_type": type(error).__name__,
                    "detail_log": f"{stage}.errors.jsonl" if recorded else "runner.errors.jsonl",
                }
            },
        )
        if not recorded:
            try:
                run.fail(stage, type(error).__name__)
            except Exception as state_error:
                log.exception("manifest.failure_update_failed")
                error.add_note(f"Updating the run status also failed: {state_error}")
        error.add_note(f"Run {run.id}; diagnostic files: {run.path}")
        raise
    finally:
        close_logger(log)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Sales CSV path (default: data/raw/sales.csv).")
    parser.add_argument("--demo-failure", choices=["missing-file", "schema", "unexpected"])
    parser.add_argument("--timeout", type=int, default=120, help="Seconds allowed per cell.")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        run = execute_pipeline(
            project_root(),
            input_csv=args.input,
            demo_failure=args.demo_failure,
            timeout=args.timeout,
        )
    except KeyboardInterrupt:
        print("Interrupted; see the run's logs and manifest.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Pipeline stopped: {type(error).__name__}", file=sys.stderr)
        for note in getattr(error, "__notes__", []):
            print(note, file=sys.stderr)
        return 1
    print(json.dumps({"run_id": run.id, "status": run.read()["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
