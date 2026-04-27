"""Competition utilities for the CVLab workshop notebook.

The notebook should only wire model metrics into this module. All competition
logic lives here so the same code can be imported from a raw GitHub URL in
Colab, local Jupyter, or any future instructor-side backend script.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import requests

try:
    import torch
except Exception:  # pragma: no cover - torch is available in the workshop env.
    torch = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SUBMISSION_TYPE = "cvlab_face_classification_submission"


def now_iso() -> str:
    """Return a UTC timestamp with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def to_jsonable(value: Any) -> Any:
    """Convert common scientific Python objects into strict JSON values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if torch is not None and isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().numpy())
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    return value


def canonical_json(data: Any) -> str:
    """Stable compact JSON used for hashes and submission IDs."""
    return json.dumps(to_jsonable(data), sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slugify(text: Any, fallback: str = "run", max_len: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return (slug or fallback)[:max_len]


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    with path.open() as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(to_jsonable(data), f, indent=2, allow_nan=False)


def benchmark_fingerprint(root: str | Path) -> dict[str, Any]:
    """Fingerprint benchmark image bytes and relative paths.

    This catches accidental benchmark changes. It is not a security boundary:
    a participant who can edit notebook code can still bypass local checks.
    """
    root = Path(root)
    rows: list[tuple[str, str, int]] = []
    for path in sorted(root.glob("**/*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            rel = path.relative_to(root).as_posix()
            rows.append((rel, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size))

    manifest = "\n".join(f"{rel}\t{digest}" for rel, digest, _ in rows)
    classes = Counter(Path(rel).parts[0] for rel, _, _ in rows if Path(rel).parts)
    return {
        "algorithm": "sha256 of lines: relative_path<TAB>file_sha256, sorted by path",
        "sha256": hashlib.sha256(manifest.encode("utf-8")).hexdigest(),
        "n_files": len(rows),
        "classes": dict(sorted(classes.items())),
    }


def run_key(base_key: str, existing_runs: dict[str, Any]) -> str:
    base = slugify(base_key, fallback="run")
    if base not in existing_runs:
        return base
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    candidate = f"{base}_{stamp}"
    suffix = 2
    while candidate in existing_runs:
        candidate = f"{base}_{stamp}_{suffix}"
        suffix += 1
    return candidate


def normalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    clean = to_jsonable(metrics)
    if clean.get("bench_pred") is not None:
        clean["bench_pred_sha256"] = sha256_text(canonical_json(clean["bench_pred"]))
    return clean


def make_submission(run: dict[str, Any], competition_name: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "type": SUBMISSION_TYPE,
        "competition": competition_name,
        "submitted_at": now_iso(),
        "run": to_jsonable(run),
    }


def validate_submission(submission: dict[str, Any], expected_competition: str | None = None) -> tuple[bool, str]:
    if submission.get("type") != SUBMISSION_TYPE:
        return False, "wrong submission type"
    if expected_competition and submission.get("competition") != expected_competition:
        return False, "wrong competition name"
    run = submission.get("run")
    if not isinstance(run, dict):
        return False, "missing run"
    for key in ["participant", "model", "bench_acc", "benchmark_sha256", "submission_id"]:
        if key not in run:
            return False, f"missing run.{key}"
    return True, "ok"


def leaderboard_from_runs(
    runs: dict[str, Any],
    top_n_per_person: int = 3,
    require_benchmark_ok: bool = True,
) -> pd.DataFrame:
    if not runs:
        return pd.DataFrame()

    df = pd.DataFrame(runs).T.copy()
    if "participant" not in df:
        df["participant"] = "anonymous"
    if "participant_id" not in df:
        df["participant_id"] = df["participant"].map(lambda x: slugify(x, fallback="anonymous"))
    if "saved_at" not in df:
        df["saved_at"] = ""
    if require_benchmark_ok and "benchmark_ok" in df:
        df = df[df["benchmark_ok"] == True]
    if df.empty or "bench_acc" not in df:
        return pd.DataFrame()
    if "bench_f1" not in df:
        df["bench_f1"] = np.nan

    for col in ["bench_acc", "bench_f1", "val_acc", "val_f1"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["bench_acc", "bench_f1", "saved_at"], ascending=[False, False, True])
    df = df.groupby("participant_id", group_keys=False).head(int(top_n_per_person))
    df = df.sort_values(["bench_acc", "bench_f1", "saved_at"], ascending=[False, False, True]).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def leaderboard_payload(
    runs: dict[str, Any],
    competition_name: str,
    top_n_per_person: int,
    benchmark_sha256: str,
) -> dict[str, Any]:
    df = leaderboard_from_runs(runs, top_n_per_person=top_n_per_person)
    entries = [] if df.empty else json.loads(df.replace({np.nan: None}).to_json(orient="records"))
    return {
        "schema_version": 1,
        "competition": competition_name,
        "top_n_per_person": int(top_n_per_person),
        "benchmark_sha256": benchmark_sha256,
        "updated_at": now_iso(),
        "entries": to_jsonable(entries),
    }


def load_submission_files(folder: str | Path, expected_competition: str | None = None) -> dict[str, Any]:
    """Load valid submission JSON files from a folder into a runs mapping."""
    runs: dict[str, Any] = {}
    for path in sorted(Path(folder).glob("*.json")):
        submission = read_json(path, default={})
        ok, reason = validate_submission(submission, expected_competition=expected_competition)
        if not ok:
            print(f"Skipping {path.name}: {reason}")
            continue
        run = submission["run"]
        key = str(run.get("run_key") or run.get("submission_id") or path.stem)
        runs[key] = run
    return runs


@dataclass
class CompetitionManager:
    data_path: str | Path
    benchmark_path: str | Path
    benchmark_size: int
    train_size: int
    validation_size: int
    seed: int
    n_images_per_class: int | None
    validation_split: float
    dataset_zip_url: str
    competition_name: str
    participant_name: str = ""
    top_n_per_person: int = 3
    expected_benchmark_sha256: str = ""
    submit_url: str = ""
    leaderboard_url: str = ""
    display_fn: Callable[[Any], None] | None = None
    input_fn: Callable[[str], str] | None = input

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        self.benchmark_path = Path(self.benchmark_path)
        self.results_path = self.data_path / "results"
        self.results_file = self.results_path / "runs.json"
        self.submissions_path = self.results_path / "submissions"
        self.local_leaderboard_file = self.results_path / "local_leaderboard.json"
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.submissions_path.mkdir(parents=True, exist_ok=True)

        self.benchmark_fingerprint = benchmark_fingerprint(self.benchmark_path)
        self.benchmark_ok = (
            not self.expected_benchmark_sha256
            or self.benchmark_fingerprint["sha256"] == self.expected_benchmark_sha256
        )

    def print_setup(self) -> None:
        print(f"Results file: {self.results_file}")
        print(
            "Benchmark fingerprint: "
            f"{self.benchmark_fingerprint['sha256']} ({self.benchmark_fingerprint['n_files']} files)"
        )
        if not self.benchmark_ok:
            print("WARNING: benchmark fingerprint does not match the workshop fingerprint. Do not submit this run.")

    def ask_participant_name(self, ask: bool = True) -> str:
        name = str(self.participant_name or "").strip()
        if not name and ask and self.input_fn is not None:
            try:
                name = self.input_fn("Leaderboard display name: ").strip()
            except Exception:
                name = ""
        self.participant_name = (name or "anonymous")[:60]
        return self.participant_name

    def load_results(self) -> dict[str, Any]:
        return read_json(self.results_file, default={})

    def save_result(self, key: str, metrics: dict[str, Any], ask_name: bool = True) -> str:
        runs = self.load_results()
        key = run_key(key, runs)
        participant = self.ask_participant_name(ask=ask_name)
        run = {
            "run_key": key,
            "participant": participant,
            "participant_id": slugify(participant, fallback="anonymous"),
            "competition": self.competition_name,
            "saved_at": now_iso(),
            "benchmark_sha256": self.benchmark_fingerprint["sha256"],
            "benchmark_n_files": self.benchmark_fingerprint["n_files"],
            "benchmark_ok": bool(self.benchmark_ok),
            "top_n_per_person": int(self.top_n_per_person),
            "seed": int(self.seed),
            "n_images_per_class": None if self.n_images_per_class is None else int(self.n_images_per_class),
            "validation_split": float(self.validation_split),
            "train_size": int(self.train_size),
            "validation_size": int(self.validation_size),
            "benchmark_size": int(self.benchmark_size),
            "dataset_zip_url": self.dataset_zip_url,
            **normalize_metrics(metrics),
        }
        run["submission_id"] = sha256_text(canonical_json({k: v for k, v in run.items() if k != "submission_id"}))
        runs[key] = run
        write_json(self.results_file, runs)

        submission = make_submission(run, self.competition_name)
        submission_file = self.submissions_path / f"{key}.json"
        write_json(submission_file, submission)
        print(f"[results] Saved run: {key}")
        print(f"[results] Submission JSON: {submission_file}")
        return key

    def leaderboard_from_runs(self, top_n: int | None = None) -> pd.DataFrame:
        return leaderboard_from_runs(
            self.load_results(),
            top_n_per_person=int(top_n or self.top_n_per_person),
        )

    def show_local_leaderboard(self, top_n: int | None = None) -> pd.DataFrame:
        df = self.leaderboard_from_runs(top_n=top_n)
        if df.empty:
            print("No valid competition runs yet.")
            return df

        cols = [
            "rank",
            "participant",
            "model",
            "bench_acc",
            "bench_f1",
            "val_acc",
            "epochs",
            "run_key",
            "saved_at",
        ]
        cols = [c for c in cols if c in df.columns]
        shown = df[cols].rename(
            columns={
                "bench_acc": "Benchmark Acc",
                "bench_f1": "Benchmark F1",
                "val_acc": "Val Acc",
                "saved_at": "Saved at",
            }
        )
        for col in ["Benchmark Acc", "Benchmark F1", "Val Acc"]:
            if col in shown:
                shown[col] = pd.to_numeric(shown[col], errors="coerce").round(4)
        if self.display_fn is not None:
            self.display_fn(shown)
        else:
            print(shown.to_string(index=False))
        return df

    def export_local_leaderboard(self, path: str | Path | None = None, top_n: int | None = None) -> dict[str, Any]:
        path = Path(path) if path is not None else self.local_leaderboard_file
        payload = leaderboard_payload(
            self.load_results(),
            competition_name=self.competition_name,
            top_n_per_person=int(top_n or self.top_n_per_person),
            benchmark_sha256=self.expected_benchmark_sha256 or self.benchmark_fingerprint["sha256"],
        )
        write_json(path, payload)
        print(f"Local leaderboard JSON: {path}")
        return payload

    def submit_result(self, run_key_value: str | None = None) -> dict[str, Any]:
        runs = self.load_results()
        if not runs:
            raise RuntimeError("No runs saved yet. Train and evaluate a model first.")
        if run_key_value is None:
            # JSON preserves insertion order; default to the run saved most recently.
            run_key_value = next(reversed(runs))
        if run_key_value not in runs:
            raise KeyError(f"Unknown run_key {run_key_value!r}. Available: {list(runs)[:5]}")

        payload = make_submission(runs[run_key_value], self.competition_name)
        submission_file = self.submissions_path / f"{run_key_value}.json"
        write_json(submission_file, payload)

        if not self.submit_url:
            print(f"Submission saved locally: {submission_file}")
            print("No COMPETITION_SUBMIT_URL is configured, so nothing was sent over the network.")
            print("Ask the instructor for the official endpoint, then re-run this cell.")
            return payload

        response = requests.post(self.submit_url, json=payload, timeout=30)
        if not response.ok:
            print(f"Submit failed with HTTP {response.status_code}. Local JSON is still saved at: {submission_file}")
            print(response.text[:1000])
            response.raise_for_status()
        print(f"Submitted {run_key_value} to central leaderboard.")
        try:
            return response.json()
        except ValueError:
            return {"status": "submitted", "text": response.text}

    def load_central_leaderboard(self) -> dict[str, Any] | None:
        if not self.leaderboard_url:
            print("No COMPETITION_LEADERBOARD_URL configured.")
            return None
        response = requests.get(self.leaderboard_url, timeout=15)
        response.raise_for_status()
        data = response.json()
        entries = data.get("entries", [])
        print(f"Central leaderboard: {self.leaderboard_url}")
        print(f"Updated: {data.get('updated_at', 'unknown')} | entries: {len(entries)}")
        if entries:
            df = pd.DataFrame(entries)
            if self.display_fn is not None:
                self.display_fn(df)
            else:
                print(df.to_string(index=False))
        return data

    def refresh_config(
        self,
        participant_name: str | None = None,
        top_n_per_person: int | None = None,
        submit_url: str | None = None,
        leaderboard_url: str | None = None,
    ) -> None:
        if participant_name is not None:
            self.participant_name = participant_name
        if top_n_per_person is not None:
            self.top_n_per_person = int(top_n_per_person)
        if submit_url is not None:
            self.submit_url = submit_url
        if leaderboard_url is not None:
            self.leaderboard_url = leaderboard_url


def save_public_leaderboard_from_submissions(
    submissions_folder: str | Path,
    output_path: str | Path,
    competition_name: str,
    top_n_per_person: int,
    benchmark_sha256: str,
) -> dict[str, Any]:
    """Instructor helper: build public leaderboard JSON from submitted files."""
    runs = load_submission_files(submissions_folder, expected_competition=competition_name)
    payload = leaderboard_payload(
        runs,
        competition_name=competition_name,
        top_n_per_person=top_n_per_person,
        benchmark_sha256=benchmark_sha256,
    )
    write_json(output_path, payload)
    return payload
