#!/bin/bash

# Context Engineering Experiments - Complete Setup Script
# This script creates the entire project structure
# Run: bash download_and_setup.sh

set -e

PROJECT_NAME="context-engineering-experiments"

echo "=========================================="
echo "Context Engineering Experiments"
echo "Complete Project Setup"
echo "=========================================="
echo ""

# Create project directory
mkdir -p $PROJECT_NAME
cd $PROJECT_NAME

echo "✓ Created project directory: $PROJECT_NAME"

# Create directory structure
mkdir -p src/{models,context_engineering,experiments,corpus/downloaders,evaluation,utils}
mkdir -p data/{raw/{api_docs,financial_reports,academic_papers,padding_corpus},processed,questions}
mkdir -p results/{raw,metrics,analysis,visualizations}
mkdir -p scripts
mkdir -p notebooks
mkdir -p tests
mkdir -p logs

# Create __init__.py files
touch src/__init__.py
touch src/models/__init__.py
touch src/context_engineering/__init__.py
touch src/experiments/__init__.py
touch src/corpus/__init__.py
touch src/evaluation/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py

# Create .gitkeep files
touch data/raw/.gitkeep
touch results/raw/.gitkeep
touch logs/.gitkeep

echo "✓ Directory structure created"

# Create requirements.txt
cat > requirements.txt << 'REQUIREMENTS_EOF'
# Core dependencies
google-generativeai>=0.8.0
python-dotenv>=1.0.0

# Data processing
numpy>=1.24.0
pandas>=2.0.0

# NLP and embeddings
tiktoken>=0.5.0
sentence-transformers>=2.2.0

# Vector stores
faiss-cpu>=1.7.4
chromadb>=0.4.0

# Web scraping
beautifulsoup4>=4.12.0
requests>=2.31.0
selenium>=4.15.0

# Document processing
PyPDF2>=3.0.0
pdfplumber>=0.10.0
python-edgar>=3.0.0
wikipedia>=1.4.0

# Statistical analysis
scipy>=1.10.0
statsmodels>=0.14.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Development
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
REQUIREMENTS_EOF

echo "✓ requirements.txt created"

# Create .env.example
cat > .env.example << 'ENV_EOF'
# Google AI API Configuration
GOOGLE_API_KEY=your_api_key_here

# Rate Limits (Gemini 2.0 Flash Experimental Free Tier)
RATE_LIMIT_RPM=15
RATE_LIMIT_TPM=1000000
RATE_LIMIT_RPD=1500

# Experiment Configuration
EXPERIMENT_REPETITIONS=3
FILL_PERCENTAGES=0.1,0.3,0.5,0.7,0.9

# Budget Alert
BUDGET_LIMIT=174.00
BUDGET_ALERT_THRESHOLD=0.90

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/experiment.log
ENV_EOF

echo "✓ .env.example created"

# Create .gitignore
cat > .gitignore << 'GITIGNORE_EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/
env/
.env

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Results
results/raw/*
results/metrics/*
results/.rate_limiter_state.json
results/.rate_limit_status.json
!results/raw/.gitkeep

# Logs
logs/
*.log

# IDEs
.vscode/
.idea/
*.swp

# OS
.DS_Store

# Notebooks
.ipynb_checkpoints/
GITIGNORE_EOF

echo "✓ .gitignore created"

# Create setup.py
cat > setup.py << 'SETUP_EOF'
from setuptools import setup, find_packages

setup(
    name="context-engineering-experiments",
    version="0.1.0",
    description="Experimental suite for testing context engineering in LLMs",
    packages=find_packages(),
    install_requires=[
        "google-generativeai>=0.8.0",
        "numpy>=1.24.0",
        "tiktoken>=0.5.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.10",
)
SETUP_EOF

echo "✓ setup.py created"

# Create config.py
cat > src/config.py << 'CONFIG_EOF'
"""Configuration for experiments"""
import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ExperimentConfig:
    """Base configuration for all experiments"""
    model_name: str = "gemini-2.0-flash-exp"
    context_limit: int = 1_000_000
    temperature: float = 0.0
    repetitions: int = int(os.getenv("EXPERIMENT_REPETITIONS", "3"))
    fill_percentages: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    
@dataclass
class APIConfig:
    """API configuration"""
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    rate_limit_rpm: int = int(os.getenv("RATE_LIMIT_RPM", "15"))
    rate_limit_tpm: int = int(os.getenv("RATE_LIMIT_TPM", "1000000"))
    rate_limit_rpd: int = int(os.getenv("RATE_LIMIT_RPD", "1500"))
    
@dataclass
class BudgetConfig:
    """Budget tracking configuration"""
    limit_usd: float = float(os.getenv("BUDGET_LIMIT", "174.00"))
    alert_threshold: float = float(os.getenv("BUDGET_ALERT_THRESHOLD", "0.90"))
    
    # Gemini pricing (per 1k tokens)
    input_cost_per_1k: float = 0.00001875
    output_cost_per_1k: float = 0.000075

config = ExperimentConfig()
api_config = APIConfig()
budget_config = BudgetConfig()
CONFIG_EOF

echo "✓ src/config.py created"

# Create utility modules
cat > src/utils/tokenizer.py << 'TOKENIZER_EOF'
"""Token counting utilities"""
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens in text"""
    return len(encoding.encode(text))

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to max tokens"""
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])
TOKENIZER_EOF

cat > src/utils/logging.py << 'LOGGING_EOF'
"""Structured logging"""
import logging
import sys
from pathlib import Path

def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """Get configured logger"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        logger.setLevel(logging.INFO)
    
    return logger
LOGGING_EOF

cat > src/utils/stats.py << 'STATS_EOF'
"""Statistical analysis utilities"""
from scipy import stats
import numpy as np

def paired_t_test(group1, group2):
    """Paired t-test"""
    t_stat, p_value = stats.ttest_rel(group1, group2)
    return {'t_statistic': float(t_stat), 'p_value': float(p_value)}

def cohen_d(group1, group2):
    """Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std
STATS_EOF

echo "✓ Utility modules created"

# Create test file
cat > tests/test_utils.py << 'TEST_EOF'
"""Tests for utilities"""
import pytest
from src.utils.tokenizer import count_tokens, truncate_to_tokens
import numpy as np

def test_count_tokens():
    text = "Hello world"
    tokens = count_tokens(text)
    assert tokens > 0
    assert isinstance(tokens, int)

def test_truncate_to_tokens():
    text = "This is a long text " * 100
    truncated = truncate_to_tokens(text, 50)
    assert count_tokens(truncated) <= 50
TEST_EOF

echo "✓ Tests created"

# Create README placeholder
cat > README.md << 'README_EOF'
# Context Engineering for LLMs: Experimental Suite

⚠️ **IMPORTANT:** Copy the full README.md content from Artifact #1 in the Claude conversation.

This is a placeholder. The full documentation includes:
- Project goals and hypotheses
- Experimental design
- Rate limiting strategy
- Complete setup instructions

For now, follow QUICK_START.md to get started.
README_EOF

cat > QUICK_START.md << 'QUICKSTART_EOF'
# Quick Start Guide

⚠️ **Copy full content from Artifact #4 in Claude conversation**

## Immediate Next Steps:

1. Activate virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure API key:
   ```bash
   cp .env.example .env
   nano .env  # Add your GOOGLE_API_KEY
   ```

4. Check feasibility:
   ```bash
   python scripts/estimate_feasibility.py
   ```

5. Read full documentation in Claude conversation artifacts
QUICKSTART_EOF

echo "✓ Documentation placeholders created"

# Create placeholder note
cat > COPY_ARTIFACTS.txt << 'ARTIFACTS_EOF'
📋 ARTIFACTS TO COPY FROM CLAUDE CONVERSATION
==============================================

You need to copy the following artifacts from the Claude conversation:

1. README.md (Artifact #1)
   → Replace the current README.md

2. PROJECT_PLAN.md (Artifact #2)
   → Create this file

3. RATE_LIMITS_GUIDE.md (Artifact #8)
   → Create this file

4. src/utils/rate_limiter.py (Artifact #5)
   → Create this file

5. scripts/estimate_feasibility.py (Artifact #6)
   → Create this file

6. scripts/check_rate_limits.py (Artifact #7)
   → Create this file

7. QUICK_START.md (Artifact #4)
   → Replace the current QUICK_START.md

How to copy:
1. In Claude conversation, click on each artifact
2. Copy the full content
3. Paste into the corresponding file
4. Save

Or use Claude Code to help:
```bash
claude code
# Ask: "Create src/utils/rate_limiter.py with the content from the conversation"
```
ARTIFACTS_EOF

echo "✓ Created artifact copy guide"

# Initialize git
git init
git add .
git commit -m "Initial project structure"

echo "✓ Git repository initialized"

# Create virtual environment
python3 -m venv venv

echo "✓ Virtual environment created"

echo ""
echo "=========================================="
echo "✅ Basic Setup Complete!"
echo "=========================================="
echo ""
echo "📋 NEXT STEPS:"
echo ""
echo "1. Copy artifacts from Claude conversation:"
echo "   cat COPY_ARTIFACTS.txt"
echo ""
echo "2. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "3. Install dependencies:"
echo "   pip install -r requirements.txt"
echo ""
echo "4. Configure API key:"
echo "   cp .env.example .env"
echo "   nano .env  # Add GOOGLE_API_KEY"
echo ""
echo "5. Copy remaining artifacts (see COPY_ARTIFACTS.txt)"
echo ""
echo "6. Check feasibility:"
echo "   python scripts/estimate_feasibility.py"
echo ""
echo "🚀 Ready to start development with Claude Code!"
echo ""
