"""Configuration for experiments"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ExperimentConfig:
    """Base configuration for all experiments"""
    model_name: str = "gemini-2.0-flash-exp"
    context_limit: int = 1_000_000
    temperature: float = 0.0
    repetitions: int = 3
    fill_percentages: list = None
    
    def __post_init__(self):
        if self.fill_percentages is None:
            self.fill_percentages = [0.1, 0.3, 0.5, 0.7, 0.9]

@dataclass
class APIConfig:
    """API configuration"""
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    rate_limit_rpm: int = int(os.getenv("RATE_LIMIT_RPM", 15))
    rate_limit_tpm: int = int(os.getenv("RATE_LIMIT_TPM", 1_000_000))
    rate_limit_rpd: int = int(os.getenv("RATE_LIMIT_RPD", 1_500))

# Global config instances
config = ExperimentConfig()
api_config = APIConfig()

