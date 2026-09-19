"""Vercel ASGI entrypoint; keep paths anchored to the deployed bundle."""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mapexploc.web import create_web_app  # noqa: E402

if sys.version_info[:2] != (3, 12):
    raise RuntimeError("The public deployment requires Python 3.12")

logging.basicConfig(level=logging.INFO)
app = create_web_app(trusted_root=ROOT)
