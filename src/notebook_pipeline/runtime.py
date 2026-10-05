"""Run boundaries, artifact publication, and automatic IPython cell instrumentation.

Files are local to a run. A stage publishes only after its final cell succeeds.
The manifest is the handoff contract; failed or incomplete stages cannot be consumed.
"""

import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from filelock import FileLock
from IPython import get_ipython

from .errors import ArtifactError, RunStateError
from .observability import close_logger, make_logger, utc_now

STAGES = ("01_ingest", "02_analyze", "03_visualize")
REQUIRED_OUTPUTS = {
    "01_ingest": {"clean_sales.parquet", "quality_report.json"},
    "02_analyze": {"monthly_sales.parquet", "category_sales.parquet", "kpis.json"},
    "03_visualize": {"sales_dashboard.png"},
}


def project_root() -> Path:
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "notebooks").is_dir():
            return candidate
    raise RunStateError("Start Jupyter or the runner from this project's directory.")


@contextmanager
def atomic_path(destination: Path):
    """Write beside the destination, then replace it atomically on the same volume."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid4().hex}{destination.suffix}")
    try:
        yield temporary
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: dict) -> None:
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Run:
    def __init__(self, root: Path, run_id: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise RunStateError("run_id must contain only letters, digits, underscores or hyphens.")
        self.root = root.resolve()
        self.id = run_id
        self.path = self.root / "runs" / run_id
        self.manifest_path = self.path / "manifest.json"
        if not self.manifest_path.is_file():
            raise RunStateError(f"Unknown run {run_id!r}; execute notebook 01 first.")
        self.lock = FileLock(str(self.path / ".manifest.lock"), timeout=10)

    @classmethod
    def create(cls, root: Path) -> "Run":
        stamp = utc_now().replace("-", "").replace(":", "").replace(".", "")
        run_id = f"{stamp}_{uuid4().hex[:10]}"
        path = root / "runs" / run_id
        path.mkdir(parents=True, exist_ok=False)
        write_json(
            path / "manifest.json",
            {
                "run_id": run_id,
                "created_at": utc_now(),
                "status": "running",
                "stages": {name: {"status": "pending", "artifacts": {}} for name in STAGES},
            },
        )
        return cls(root, run_id)

    def read(self) -> dict:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    @contextmanager
    def update(self):
        with self.lock:
            manifest = self.read()
            yield manifest
            write_json(self.manifest_path, manifest)

    def begin(self, stage: str) -> None:
        if stage not in STAGES:
            raise RunStateError(f"Unknown stage {stage!r}.")
        with self.update() as manifest:
            stages = manifest["stages"]
            if manifest["status"] != "running" or stages[stage]["status"] != "pending":
                raise RunStateError(
                    "This stage has already started or the run is closed. Start a new run."
                )
            if any(stages[name]["status"] != "succeeded" for name in STAGES[: STAGES.index(stage)]):
                raise RunStateError(
                    f"Complete all notebooks before {stage} using this same run_id."
                )
            stages[stage].update(status="running", started_at=utc_now())

    def fail(self, stage: str, error_type: str) -> None:
        with self.update() as manifest:
            manifest.update(status="failed", finished_at=utc_now())
            manifest["stages"][stage].update(
                status="failed", finished_at=utc_now(), error_type=error_type
            )
            for name in STAGES[STAGES.index(stage) + 1 :]:
                if manifest["stages"][name]["status"] == "pending":
                    manifest["stages"][name]["status"] = "skipped"


class NotebookSession:
    def __init__(self, run: Run, stage: str, shell=None):
        self.run, self.stage, self.shell = run, stage, shell
        self.context = {"run_id": run.id, "notebook": stage, "cell_id": None, "step": "setup"}
        self.log = make_logger(run.path / "logs", stage, self.context)
        self.started = time.perf_counter()
        self.pending_finish = False
        self.closed = False
        self.outputs: dict[str, dict] = {}
        self.log.info("notebook.started")
        if shell is not None:
            shell.events.register("pre_run_cell", self.before_cell)
            shell.events.register("post_run_cell", self.after_cell)

    @property
    def run_id(self) -> str:
        return self.run.id

    def before_cell(self, info) -> None:
        self.started = time.perf_counter()
        label = next(
            (line[7:].strip() for line in info.raw_cell.splitlines() if line.startswith("# step:")),
            "cell",
        )
        count = getattr(self.shell, "execution_count", None)
        self.context.update(
            cell_id=getattr(info, "cell_id", None) or (f"In[{count}]" if count else None),
            execution_count=count,
            step=label,
        )
        self.log.info("cell.started")

    def after_cell(self, result) -> None:
        self.context["execution_count"] = result.execution_count
        if self.context["cell_id"] is None:
            self.context["cell_id"] = f"In[{result.execution_count}]"
        error = result.error_before_exec or result.error_in_exec
        elapsed = round((time.perf_counter() - self.started) * 1000, 2)
        if error is not None:
            self.log.error(
                "cell.failed",
                exc_info=(type(error), error, error.__traceback__),
                extra={"fields": {"duration_ms": elapsed}},
            )
            self.pending_finish = False
            try:
                self.run.fail(self.stage, type(error).__name__)
            except Exception:
                # Event callbacks must not replace the cell's original exception.
                self.log.exception("manifest.failure_update_failed")
        else:
            self.log.info("cell.succeeded", extra={"fields": {"duration_ms": elapsed}})
            if self.pending_finish:
                try:
                    self._commit()
                except Exception as error:
                    self.log.exception("notebook.commit_failed")
                    self.run.fail(self.stage, type(error).__name__)

    def _require_running(self) -> None:
        manifest = self.run.read()
        if manifest["status"] != "running" or manifest["stages"][self.stage]["status"] != "running":
            raise RunStateError(
                "This stage is closed or failed. Start a fresh run from notebook 01."
            )

    @contextmanager
    def output(self, name: str):
        self._require_running()
        if Path(name).name != name or name not in REQUIRED_OUTPUTS[self.stage]:
            raise ArtifactError(f"Undeclared output {name!r} for {self.stage}.")
        destination = self.run.path / "artifacts" / self.stage / name
        try:
            with atomic_path(destination) as temporary:
                yield temporary
            self.outputs[name] = {
                "path": destination.relative_to(self.run.path).as_posix(),
                "sha256": checksum(destination),
            }
        except OSError as error:
            raise ArtifactError(
                f"Could not publish {name}; check disk space and permissions."
            ) from error
        self.log.info("artifact.written", extra={"fields": {"artifact": name}})

    def input(self, stage: str, name: str) -> Path:
        manifest = self.run.read()
        if stage not in STAGES[: STAGES.index(self.stage)]:
            raise RunStateError("Inputs must come from an earlier notebook.")
        producer = manifest["stages"][stage]
        if manifest["status"] != "running" or producer["status"] != "succeeded":
            raise RunStateError(f"Upstream notebook {stage} has not succeeded in this run.")
        try:
            entry = producer["artifacts"][name]
            path = (self.run.path / entry["path"]).resolve()
            if (
                not path.is_relative_to(self.run.path.resolve())
                or checksum(path) != entry["sha256"]
            ):
                raise ArtifactError(
                    f"Artifact {name!r} was modified after publication. Start a new run."
                )
            return path
        except (KeyError, OSError) as error:
            raise ArtifactError(
                f"Missing artifact {name!r}; rerun the pipeline from notebook 01."
            ) from error

    def finish(self) -> None:
        """Call in the last cell; publication waits for that entire cell to succeed."""
        self._require_running()
        missing = REQUIRED_OUTPUTS[self.stage] - self.outputs.keys()
        if missing:
            raise ArtifactError(f"Cannot finish: missing outputs {sorted(missing)}.")
        self.pending_finish = True
        if self.shell is None:
            self._commit()

    def _commit(self) -> None:
        with self.run.update() as manifest:
            if manifest["stages"][self.stage]["status"] != "running":
                raise RunStateError("Cannot commit a failed or closed stage.")
            manifest["stages"][self.stage].update(
                status="succeeded", finished_at=utc_now(), artifacts=self.outputs
            )
            if self.stage == STAGES[-1]:
                manifest.update(status="succeeded", finished_at=utc_now())
        self.log.info("notebook.succeeded")
        self.pending_finish = False
        self.close()

    def close(self) -> None:
        if self.closed:
            return
        if self.shell is not None:
            self.shell.events.unregister("pre_run_cell", self.before_cell)
            self.shell.events.unregister("post_run_cell", self.after_cell)
        close_logger(self.log)
        self.closed = True


def start_notebook(stage: str, run_id: str | None = None, *, root: Path | None = None):
    root = Path(root) if root is not None else project_root()
    shell = get_ipython()
    previous = getattr(shell, "_retail_pipeline_session", None) if shell else None
    if previous and not previous.closed:
        if previous.stage == stage and previous.run_id == run_id:
            previous._require_running()
            return previous
        previous.run.fail(previous.stage, "SessionReplaced")
        previous.close()
    if run_id is None:
        if stage != STAGES[0]:
            raise RunStateError("Set run_id to the ID printed by notebook 01, or use run-pipeline.")
        run = Run.create(root)
    else:
        run = Run(root, run_id)
    run.begin(stage)
    session = NotebookSession(run, stage, shell)
    if shell:
        shell._retail_pipeline_session = session
    print(f"Run: {run.id}\nLogs: {run.path / 'logs'}")
    return session
