"""
Base class for all experiments.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import time
from dataclasses import dataclass

from src.models.gemini_client import GeminiClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ExperimentStatus:
    """Status tracking for resumable experiments"""
    total_runs: int = 0
    completed_runs: int = 0
    failed_runs: int = 0
    last_run_timestamp: str = ""
    completed_keys: List[str] = None
    
    def __post_init__(self):
        if self.completed_keys is None:
            self.completed_keys = []

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
        status_file: Optional[str] = None
    ):
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_file = self.output_dir / f"{name}_results.jsonl"
        self.status_file = Path(status_file) if status_file else self.output_dir / f"{name}_status.json"
        
        self.client = GeminiClient()
        self.status = self._load_status()
        
    def _load_status(self) -> ExperimentStatus:
        """Load existing status or create new"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    data = json.load(f)
                    return ExperimentStatus(**data)
            except Exception as e:
                logger.warning(f"Failed to load status file: {e}. Starting fresh.")
        
        # If no status file, try to reconstruct from results file
        completed_keys = []
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # Construct a unique key for the run. 
                        # Subclasses should override _get_run_key(data) to match this.
                        key = self._get_run_key_from_result(data)
                        if key:
                            completed_keys.append(key)
                    except:
                        continue
        
        return ExperimentStatus(completed_keys=completed_keys, completed_runs=len(completed_keys))

    def _save_status(self):
        """Save current status to disk"""
        with open(self.status_file, 'w') as f:
            # Convert dataclass to dict
            data = {
                "total_runs": self.status.total_runs,
                "completed_runs": self.status.completed_runs,
                "failed_runs": self.status.failed_runs,
                "last_run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "completed_keys": self.status.completed_keys
            }
            json.dump(data, f, indent=2)

    def is_completed(self, run_key: str) -> bool:
        """Check if a specific run config is already done"""
        return run_key in self.status.completed_keys

    def record_result(self, result: Dict[str, Any], run_key: str):
        """Write result to file and update status"""
        # Append to JSONL
        with open(self.results_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
        
        # Update status
        self.status.completed_keys.append(run_key)
        self.status.completed_runs += 1
        self._save_status()

    @abstractmethod
    def _get_run_key_from_result(self, result: Dict) -> str:
        """Extract unique run key from a result dict (for reconstruction)"""
        pass

    @abstractmethod
    def run(self, dry_run: bool = False, limit: int = None, **kwargs):
        """Execute the experiment logic"""
        pass
