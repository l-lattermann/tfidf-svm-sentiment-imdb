import sys
from pathlib import Path
# ─── Project Root Setup ──────────────────────────────────────────────────────────
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))
    
import logging_config
from src.config.paths import ROOT
import logging_config
logger = logging_config.configure_logging()
logger.info("Saving file to %s", ROOT / "data" / "file.txt")