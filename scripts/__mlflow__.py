"""MLflow tracking for the training scripts.

The same server, credentials file, tags and params as ``scripts/register_mlflow.py``,
except the run is opened when training starts and fed as it goes rather than
replayed from the loss csv afterwards.

Tracking never takes a training run down. A missing mlflow install, a missing
credentials file, an unreachable server or a failed call disables tracking and
prints why; the training loop carries on either way.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from __style__ import Colors, print_info

DEFAULT_TRACKING_URI = "https://ai.swmansion.com/mlflow"
DEFAULT_EXPERIMENT = "3"
CREDENTIALS_FILE = "mlflow_credits.txt"

# Put the tracking credentials here, or leave them empty and drop the pair into
# `mlflow_credits.txt` (username first line, password second) the way
# `register_mlflow.py` reads it. That file is gitignored; THIS ONE IS NOT, so
# anything filled in below travels with the next commit.
MLFLOW_USERNAME = ""
MLFLOW_PASSWORD = ""


def _git(*args: str) -> str:
    try:
        return subprocess.run(("git", *args), capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except Exception:                                                # noqa: BLE001
        return ""


def prompt_run_details(
    name: Optional[str] = None,
    description: Optional[str] = None,
    model_type: Optional[str] = None,
) -> tuple[Optional[str], str, str]:
    """Ask for what identifies the run, skipping whatever came in on the command line.

    Returns ``(name, description, model_type)`` with a ``None`` name meaning "do
    not track". Nothing is asked when stdin is not a terminal, so a queued or
    nohup'd run needs the flags instead of hanging on a prompt nobody can answer.
    """

    interactive = sys.stdin.isatty()

    if name is None:
        if not interactive:
            return None, "", ""
        print()
        name = input("MLflow experiment name (empty to skip tracking): ").strip()
        if not name:
            return None, "", ""
    if description is None:
        description = input("Description (optional): ").strip() if interactive else ""
    if model_type is None:
        model_type = input("Model type (optional): ").strip() if interactive else ""

    return name, description, model_type


class Tracker:
    """An MLflow run, or a no-op that quietly stands in for one."""

    def __init__(
        self,
        name: Optional[str],
        description: str = "",
        model_type: str = "",
        experiment: str = DEFAULT_EXPERIMENT,
        tags: Optional[dict[str, str]] = None,
    ) -> None:
        self.enabled = False
        self._mlflow: Any = None

        if not name:
            print_info("MLflow", "not tracking this run", Colors.WARNING)
            return

        try:
            import mlflow                                            # noqa: PLC0415
        except ImportError:
            print_info("MLflow", "the mlflow package is not installed; not tracking",
                       Colors.WARNING)
            return

        try:
            self._credentials()
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
            # A numeric experiment addresses one that exists; a name creates it
            # on first use. `register_mlflow.py` defaults to id 3, so this does too.
            if experiment.isdigit():
                mlflow.set_experiment(experiment_id=experiment)
            else:
                mlflow.set_experiment(experiment_name=experiment)

            mlflow.start_run(run_name=name, description=description or None)
            run_tags = {"stage": "training"}
            if model_type:
                run_tags["model_type"] = model_type
            for key, value in (("git_commit", _git("rev-parse", "--short", "HEAD")),
                               ("git_branch", _git("rev-parse", "--abbrev-ref", "HEAD"))):
                if value:
                    run_tags[key] = value
            run_tags.update(tags or {})
            mlflow.set_tags(run_tags)
        except Exception as e:                                       # noqa: BLE001
            print_info("MLflow", f"could not start a run ({type(e).__name__}: {e}); "
                                 f"not tracking", Colors.WARNING)
            try:
                mlflow.end_run()
            except Exception:                                        # noqa: BLE001
                pass
            return

        self._mlflow = mlflow
        self.enabled = True
        self._mark_failed_on_crash()
        run = mlflow.active_run()
        print_info("MLflow", f"run '{name}' in experiment {experiment} "
                             f"({run.info.run_id})", Colors.OKGREEN)

    def _mark_failed_on_crash(self) -> None:
        """Close the run as FAILED if training dies.

        mlflow's own atexit hook closes an open run as FINISHED whatever
        happened, so a crashed run would otherwise look like a completed one.
        """

        previous = sys.excepthook

        def hook(exc_type, exc, tb):                                 # noqa: ANN001
            self.finish("FAILED")
            previous(exc_type, exc, tb)

        sys.excepthook = hook

    @staticmethod
    def _credentials() -> None:
        """Resolve the username/password pair: environment, then constants, then file.

        A machine that already exports the variables needs neither of the other
        two, so nothing here overwrites them.
        """

        if os.environ.get("MLFLOW_TRACKING_USERNAME"):
            return

        if MLFLOW_USERNAME:
            os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USERNAME
            os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_PASSWORD
            return

        for directory in (Path.cwd(), Path(__file__).resolve().parent.parent):
            path = directory / CREDENTIALS_FILE
            if path.is_file():
                lines = path.read_text(encoding="utf-8").splitlines()
                if len(lines) >= 2:
                    os.environ["MLFLOW_TRACKING_USERNAME"] = lines[0].strip()
                    os.environ["MLFLOW_TRACKING_PASSWORD"] = lines[1].strip()
                return

    def _guard(self, call, what: str) -> None:
        """Run one tracking call, disabling tracking if it fails.

        A server that goes away mid-run would otherwise raise on every log for
        the rest of training, once per step.
        """

        if not self.enabled:
            return
        try:
            call()
        except Exception as e:                                       # noqa: BLE001
            self.enabled = False
            print_info("MLflow", f"{what} failed ({type(e).__name__}: {e}); "
                                 f"tracking off for the rest of this run", Colors.WARNING)

    def log_params(self, params: dict[str, Any]) -> None:
        self._guard(lambda: self._mlflow.log_params(
            {k: v for k, v in params.items() if v is not None}
        ), "logging params")

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self._guard(lambda: self._mlflow.log_metrics(
            {k: float(v) for k, v in metrics.items()}, step=step
        ), "logging metrics")

    def log_artifact(self, path: Path) -> None:
        if not Path(path).is_file():
            return
        self._guard(lambda: self._mlflow.log_artifact(str(path)), f"uploading {Path(path).name}")

    def finish(self, status: str = "FINISHED") -> None:
        if not self.enabled:
            return
        try:
            self._mlflow.end_run(status=status)
        except Exception:                                            # noqa: BLE001
            pass
        self.enabled = False
