#!/bin/bash

# ============================================================================
# Context Engineering Experiments - Automated Setup Script
# ============================================================================
# This script sets up the complete Python environment with all dependencies,
# configurations, and verification steps.
# 
# Usage: bash scripts/setup_environment.sh
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_step() {
    echo -e "${YELLOW}→${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 is not installed or not in PATH"
        return 1
    fi
    return 0
}

# ============================================================================
# Main Setup Process
# ============================================================================

print_header "Context Engineering Experiments - Setup"

# Step 1: Prerequisites Check
print_header "Step 1: Checking Prerequisites"

print_step "Checking Python version..."
if check_command python3; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.10+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python3 not found. Please install Python 3.10+"
    exit 1
fi

print_step "Checking pip..."
if check_command pip3; then
    print_success "pip found"
else
    print_error "pip not found"
    exit 1
fi

print_step "Checking available disk space..."
DISK_AVAILABLE=$(df . | awk 'NR==2 {print $4}')
DISK_AVAILABLE_GB=$((DISK_AVAILABLE / 1024 / 1024))
if [ "$DISK_AVAILABLE_GB" -gt 5 ]; then
    print_success "$DISK_AVAILABLE_GB GB available (≥5GB required)"
else
    print_error "Less than 5GB disk space available ($DISK_AVAILABLE_GB GB)"
    print_info "You may have issues downloading corpus data"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 2: Virtual Environment
print_header "Step 2: Setting Up Virtual Environment"

if [ -d "venv" ]; then
    print_step "venv directory already exists"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_step "Removing existing venv..."
        rm -rf venv
        print_step "Creating new virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_step "Using existing venv..."
    fi
else
    print_step "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_step "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Step 3: Upgrade pip
print_header "Step 3: Upgrading pip, setuptools, and wheel"

print_step "Upgrading pip..."
python -m pip install --quiet --upgrade pip setuptools wheel
print_success "pip, setuptools, wheel upgraded"

# Step 4: Install Dependencies
print_header "Step 4: Installing Project Dependencies"

if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found!"
    exit 1
fi

print_step "Installing dependencies from requirements.txt..."
pip install --quiet -r requirements.txt
print_success "Dependencies installed"

print_step "Checking for conflicts..."
if pip check --quiet 2>/dev/null; then
    print_success "No package conflicts detected"
else
    CONFLICTS=$(pip check 2>&1)
    print_error "Package conflicts detected:"
    echo "$CONFLICTS"
fi

# Step 5: Install Project in Development Mode
print_header "Step 5: Installing Project Package"

if [ -f "setup.py" ]; then
    print_step "Installing package in editable mode..."
    pip install --quiet -e .
    print_success "Package installed"
else
    print_error "setup.py not found"
fi

# Step 6: Create Data Directories
print_header "Step 6: Creating Data Directories"

print_step "Creating data directories..."
mkdir -p data/raw/api_docs
mkdir -p data/raw/academic_papers
mkdir -p data/raw/financial_reports
mkdir -p data/raw/padding_corpus
mkdir -p data/processed/tokenized
mkdir -p data/processed/embeddings
mkdir -p data/questions
print_success "Data directories created"

print_step "Creating results directories..."
mkdir -p results/raw
mkdir -p results/metrics
mkdir -p results/analysis
mkdir -p results/visualizations
print_success "Results directories created"

# Step 7: Environment Configuration
print_header "Step 7: Environment Configuration"

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        print_step "Creating .env from .env.example..."
        cp .env.example .env
        print_success ".env created"
        print_info "⚠️  Please edit .env and add your GOOGLE_API_KEY"
        read -p "Open .env for editing? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if [ -n "$EDITOR" ]; then
                $EDITOR .env
            else
                nano .env
            fi
        fi
    else
        print_error ".env.example not found"
    fi
else
    print_step ".env already exists"
    if grep -q "your_actual_api_key_here" .env; then
        print_info "GOOGLE_API_KEY not configured in .env"
        read -p "Edit .env now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if [ -n "$EDITOR" ]; then
                $EDITOR .env
            else
                nano .env
            fi
        fi
    else
        print_success "GOOGLE_API_KEY appears to be configured"
    fi
fi

# Step 8: Verification
print_header "Step 8: Verification Tests"

print_step "Testing Python environment..."
python -c "
import sys
print(f'  Python: {sys.version.split()[0]}')
print(f'  venv: {\"Yes\" if \"venv\" in sys.prefix else \"No\"}')
" 2>/dev/null || print_error "Python test failed"
print_success "Python environment test passed"

print_step "Testing core imports..."
python -c "
import google.generativeai
import numpy
import scipy
import pandas
import faiss
import sentence_transformers
import tiktoken
print('  ✓ All core packages imported')
" 2>/dev/null && print_success "Core imports test passed" || print_error "Core imports test failed"

print_step "Testing project modules..."
python -c "
from src.config import config, api_config
from src.utils.tokenizer import count_tokens
print('  ✓ Project modules loaded')
" 2>/dev/null && print_success "Project modules test passed" || print_error "Project modules test failed"

print_step "Testing environment variables..."
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY', '').strip()
if api_key and api_key != 'your_actual_api_key_here':
    print('  ✓ GOOGLE_API_KEY configured')
else:
    print('  ⚠️  GOOGLE_API_KEY not configured (required for experiments)')
" 2>/dev/null || print_error "Environment test failed"

# Step 9: Run Unit Tests (Optional)
print_header "Step 9: Running Unit Tests (Optional)"

read -p "Run unit tests? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v pytest &> /dev/null; then
        print_step "Running pytest..."
        if pytest tests/ -q 2>/dev/null; then
            print_success "All tests passed!"
        else
            print_error "Some tests failed (this may be OK for initial setup)"
        fi
    else
        print_error "pytest not found"
    fi
fi

# Step 10: Check Rate Limits (Optional)
print_header "Step 10: Checking API Configuration (Optional)"

read -p "Check API rate limits? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_step "Checking rate limits..."
    if python scripts/check_rate_limits.py 2>/dev/null; then
        print_success "API check passed"
    else
        print_error "API check failed (check that GOOGLE_API_KEY is set in .env)"
    fi
fi

# Final Summary
print_header "🎉 Setup Complete!"

echo -e "${GREEN}Environment is ready to use!${NC}\n"

echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. ${YELLOW}Activate environment:${NC} source venv/bin/activate"
echo -e "  2. ${YELLOW}Edit config:${NC} Check .env and set GOOGLE_API_KEY"
echo -e "  3. ${YELLOW}Download corpus:${NC} bash scripts/download_all.sh"
echo -e "  4. ${YELLOW}Check feasibility:${NC} python scripts/estimate_feasibility.py"
echo -e "  5. ${YELLOW}Run experiments:${NC} python scripts/run_experiment.py --experiment exp1_needle"

echo -e "\n${BLUE}Documentation:${NC}"
echo -e "  • Project plan: ${YELLOW}PLAN.md${NC}"
echo -e "  • README: ${YELLOW}README.md${NC}"

echo -e "\n${GREEN}Happy experimenting! 🚀${NC}\n"
