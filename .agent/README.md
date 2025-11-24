# Project Configuration for Antigravity Agent

## ⚠️ Critical Project Rules

### Python Environment

**ALWAYS use virtual environment when running Python scripts!**

- ❌ **NEVER** use `/usr/bin/python3` (causes segmentation faults)
- ✅ **ALWAYS** use `source venv/bin/activate` first
- ✅ Or use wrapper scripts: `./run.sh` or `./run_inference.sh`

### Why?

This project runs on **macOS with Apple Silicon (M1/M2/M3)** and uses:
- SHAP
- PyTorch  
- NumPy
- CatBoost
- LightGBM

These libraries cause **segmentation faults** with system Python 3.9.6.

**Solution:** Virtual environment with Homebrew Python 3.13.7

## Available Workflows

Use workflows with slash commands:
- `/run-python` - How to run Python scripts correctly
- `/setup` - How to setup the virtual environment

## Auto-run Commands

Commands in workflows with `// turbo-all` annotation can be auto-executed safely.
