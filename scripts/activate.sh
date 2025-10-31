#!/bin/bash

# ============================================================================
# Context Engineering Experiments - Custom Activation Script
# ============================================================================
# Activates the virtual environment with a clean, minimal prompt
# 
# Usage: source scripts/activate.sh
# ============================================================================

# Get the directory where this script is located
# Works in bash, zsh, and sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/venv"

# Check if venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at $VENV_PATH"
    echo "Please run: bash scripts/setup_environment.sh"
    return 1
fi

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Load environment variables from .env file
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(cat "$PROJECT_ROOT/.env" | grep -v '^#' | xargs)
fi

# Store original prompt in case user wants to restore it
export _ORIGINAL_PS1="$PS1"

# Set custom minimal prompt
# Shows: (venv_name) >
export PS1="(context_engineering_experiments) > "

echo "✅ Environment activated"
echo "   Project: context_engineering_experiments"
echo "   Python: $(python --version 2>&1)"
echo "   Location: $PROJECT_ROOT"
echo ""
echo "Type 'deactivate' to exit environment"
