---
description: Setup virtual environment for the project
---

# Project Setup

This workflow sets up the virtual environment to avoid segmentation faults.

## Steps:

1. **Check if venv exists:**
```bash
ls -la venv/
```

2. **If venv doesn't exist, create it:**
```bash
/opt/homebrew/bin/python3 -m venv venv
```

3. **Activate the virtual environment:**
```bash
source venv/bin/activate
```

4. **Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

5. **Verify Python version (should be 3.13.x):**
```bash
python --version
which python
```

// turbo-all

## Expected output:
- Python version: **3.13.x** (NOT 3.9.6)
- Python path: `.../venv/bin/python` (NOT `/usr/bin/python3`)
