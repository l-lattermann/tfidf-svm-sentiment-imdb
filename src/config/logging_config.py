"""Module-level setup: path config and logging directory handling."""

# ─── Imports ─────────────────────────────────────────────────────────────────────
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
# ─── Path Setup ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ─── Import Project Config ───────────────────────────────────────────────────────
from src.config.paths import LOG_DIR, ROOT

# ─── Logging Configuration ───────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = os.getenv("LOG_DIR", LOG_DIR)

timestamp = datetime.now()
timestamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = LOG_DIR / f"{timestamp}_project.log"

log_path = Path(LOG_FILE)
log_path.parent.mkdir(parents=True, exist_ok=True)

# ─── Formatter ───────────────────────────────────────────────────────
class PathSanitizerFormatter(logging.Formatter):
    def sanitize(self, msg):
        if isinstance(msg, Path):
            msg = str(msg)
        if isinstance(msg, str) and str(ROOT) in msg:
            msg = msg.replace(str(ROOT), 'project_root')
        return msg

    def format(self, record):
        record.msg = self.sanitize(record.msg)
        if record.args and isinstance(record.args, tuple):
            record.args = tuple(self.sanitize(a) for a in record.args)
        return super().format(record)

# ─── Logger Config ───────────────────────────────────────────────────
def configure_logging(logger_name='project_logger') -> logging.Logger:
    """Configure and return a project logger."""
    logger = logging.getLogger(logger_name)

    if not logger.hasHandlers():
        formatter = PathSanitizerFormatter('%(asctime)s - %(levelname)s - %(message)s')

        # Console handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File handler
        file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.setLevel(LOG_LEVEL)

    return logger
