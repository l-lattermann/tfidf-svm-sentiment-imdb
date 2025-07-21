"""Pytest configuration for setting up the test environment."""

# ─── Standard Library Imports ────────────────────────────────────────────────────
import os
import pytest
from pathlib import Path
import logging
import json

# ─── Import src to trigger logger setup ──────────────────────────────────────────
import src 

# Make the test log file directory if it does not exist
@pytest.hookimpl(tryfirst=True)
def pytest_configure():
    LOG_FILE = os.getenv("LOG_FILE") # Get the log file path from environment variable
    if LOG_FILE:
        Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True) # Ensure the directory exists
    open(LOG_FILE, "w").close() # Clear the log file before tests start

def pytest_sessionfinish(session, exitstatus):
    from src.config import logging_config
logger = logging_config.configure_logging()

    # Print coverage summary if coverage.json exists
    coverage_path = "tests/logs/coverage.json"

    if os.path.exists(coverage_path):
        with open(coverage_path) as f:
            cov_data = json.load(f)

        logger.info(f"{'Name':<50} {'Stmts':>6} {'Miss':>6} {'Cover':>6}")
        logger.info("-" * 80)

        for path, info in cov_data.get("files", {}).items():
            summary = info["summary"]
            name = path
            stmts = summary["num_statements"]
            miss = stmts - summary["covered_lines"]
            cover = summary["percent_covered_display"]
            logger.info(f"{name:<50} {stmts:>6} {miss:>6} {cover:>6}%")
    else:
        logger.warning("No coverage.json found.")

    # Delete the coverage file after the session
    coverage_file = os.getenv("COVERAGE_FILE", ".coverage")
    if os.path.exists(coverage_file):
        os.remove(coverage_file)