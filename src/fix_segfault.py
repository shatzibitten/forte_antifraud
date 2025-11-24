"""
Fix for Segmentation Fault on macOS ARM64

This module sets environment variables to disable multi-threading in numerical libraries
that can cause segfaults on macOS with Apple Silicon (M1/M2/M3).

Import this module at the very beginning of your main script BEFORE importing numpy, scipy, sklearn, etc.
"""
import os
import logging

logger = logging.getLogger(__name__)

def disable_multithreading():
    """
    Disables multithreading in BLAS/LAPACK libraries to prevent segfaults on macOS ARM64.
    
    Must be called BEFORE importing numpy, scipy, sklearn, torch, etc.
    """
    thread_vars = {
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1',
        'BLIS_NUM_THREADS': '1',
    }
    
    for var, value in thread_vars.items():
        os.environ[var] = value
        logger.debug(f"Set {var}={value}")
    
    logger.info("Disabled multi-threading in numerical libraries to prevent segfaults on ARM64")

# Auto-run on import
disable_multithreading()
