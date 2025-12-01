"""
Base class for all experiments.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.models.gemini_client import GeminiClient
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentStatus:
    """Status tracking for resumable experiments."""

    total_runs: int = 0
    completed_runs: int = 0
    failed_runs: int = 0
    last_run_timestamp: str = ""
    completed_keys: List[str] = field(default_factory=list)
    failed_keys: Dict[str, str] = field(default_factory=dict)


class BaseExperiment(ABC):
    """
    Abstract base class for experiments.

    Handles:
    - Status tracking and resumption
    - Output management
    - Basic error handling
    """

    def __init__(
        self,
        name: str,
        output_dir: str = "results/raw",
        status_file: Optional[str] = None,
    ):
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results_file = self.output_dir / f"{name}_results.jsonl"
        self.status_file = (
            Path(status_file) if status_file else self.output_dir / f"{name}_status.json"
        )

        self.client = GeminiClient()
        self.status = self._load_status()
        self._completed_key_set: Set[str] = set(self.status.completed_keys)

    def _load_status(self) -> ExperimentStatus:
        """Load existing status or create new."""
        raw_status: Optional[Dict[str, Any]] = None
        if self.status_file.exists():
            try:
                with open(self.status_file, "r") as f:
                    raw_status = json.load(f)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Failed to load status file: {exc}. Starting fresh.")

        if raw_status is None:
            raw_status = {}

        completed_keys = self._dedupe_list(raw_status.get("completed_keys", []))
        failed_keys = raw_status.get("failed_keys", {})

        # Reconstruct results from the JSONL file so resumes skip duplicates.
        result_keys = self._load_result_keys()
        seen = set(completed_keys)
        for key in sorted(result_keys):
            if key not in seen:
                completed_keys.append(key)
                seen.add(key)

        status = ExperimentStatus(
            total_runs=raw_status.get("total_runs", 0),
            completed_runs=len(seen),
            failed_runs=len(failed_keys),
            last_run_timestamp=raw_status.get("last_run_timestamp", ""),
            completed_keys=completed_keys,
            failed_keys=failed_keys,
        )
        return status

    def _dedupe_list(self, items: List[str]) -> List[str]:
        """Remove duplicates while preserving order."""
        seen: Set[str] = set()
        deduped: List[str] = []
        for item in items:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        return deduped

    def _load_result_keys(self) -> Set[str]:
        """Reconstruct completed run keys from the JSONL results file."""
        keys: Set[str] = set()
        if not self.results_file.exists():
            return keys

        with open(self.results_file, "r") as handle:
            for line in handle:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = self._get_run_key_from_result(data)
                if key:
                    keys.add(key)
        return keys

    def _save_status(self) -> None:
        """Persist current status to disk."""
        payload = {
            "total_runs": self.status.total_runs,
            "completed_runs": len(self._completed_key_set),
            "failed_runs": len(self.status.failed_keys),
            "last_run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completed_keys": self.status.completed_keys,
            "failed_keys": self.status.failed_keys,
        }
        with open(self.status_file, "w") as handle:
            json.dump(payload, handle, indent=2)

    def is_completed(self, run_key: str) -> bool:
        """Check if a specific run config is already done."""
        return run_key in self._completed_key_set

    def record_result(self, result: Dict[str, Any], run_key: str) -> None:
        """Append the result to disk and update completion metadata."""
        with open(self.results_file, "a") as handle:
            handle.write(json.dumps(result) + "\n")

        if run_key not in self._completed_key_set:
            self._completed_key_set.add(run_key)
            self.status.completed_keys.append(run_key)
        if run_key in self.status.failed_keys:
            self.status.failed_keys.pop(run_key, None)

        self.status.completed_runs = len(self._completed_key_set)
        self.status.failed_runs = len(self.status.failed_keys)
        self._save_status()

    def record_failure(self, run_key: str, reason: str) -> None:
        """Track failed run keys and persist the reason for downstream reruns."""
        self.status.failed_keys[run_key] = reason
        self.status.failed_runs = len(self.status.failed_keys)
        self._save_status()

    @abstractmethod
    def _get_run_key_from_result(self, result: Dict[str, Any]) -> str:
        """Extract unique run key from a result dict (for reconstruction)."""

    @abstractmethod
    def run(self, dry_run: bool = False, limit: int = None, **kwargs) -> None:
        """Execute the experiment logic."""
