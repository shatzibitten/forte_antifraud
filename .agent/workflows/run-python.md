---
description: Run any Python script with venv activated
---

# Running Python Scripts

**IMPORTANT:** This project uses a virtual environment to avoid segmentation faults on macOS with Apple Silicon.

## Steps to run any Python script:

1. **Always activate the virtual environment first:**
```bash
source venv/bin/activate
```

2. **Then run the Python script:**
```bash
python <script_name>.py
```

// turbo-all

## Quick commands:

**Run main.py:**
```bash
./run.sh
```

**Run inference.py:**
```bash
./run_inference.sh
```

**Run any other Python script:**
```bash
source venv/bin/activate && python <script_name>.py
```

## Why this is required:

- System Python 3.9.6 causes **segmentation faults** with SHAP, PyTorch, NumPy
- Virtual environment uses Python 3.13.7 which is optimized for ARM64
- All dependencies are properly isolated from system libraries

**NEVER use `/usr/bin/python3` or `python3` directly in this project!**
