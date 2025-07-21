"""Pytest configuration for setting up the test environment."""

# ─── Standard Library Imports ────────────────────────────────────────────────────
import os
import pytest
from pathlib import Path
import json
import tempfile


from src.config import logging_config
logger = None

# Export pytest temp root to LOG_ROOT
@pytest.fixture(scope="session", autouse=True)
def export_pytest_temp_root(tmp_path_factory):
    base_temp = tmp_path_factory.getbasetemp().resolve()
    # Move 2 levels up to pytest-of-<user>
    pytest_temp_root = base_temp.parents
    os.environ["LOG_ROOT"] = str(pytest_temp_root)

# Set up logging for the test session
@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    global logger
    log_dir = os.getenv("LOG_DIR")
    if log_dir:
        Path(log_dir).parent.mkdir(parents=True, exist_ok=True)

    logger = logging_config.configure_logging()


def pytest_sessionfinish(session, exitstatus):

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