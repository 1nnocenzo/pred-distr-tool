"""Small JSON/CSV helpers shared by the study.

Kept separate so the experiment and reporting modules do not re-implement
file handling, and so every output of the study is written the same way
(UTF-8, two-space indent, parent directories created on demand).
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)


def save_json(obj: Any, path: Path | str) -> Path:
    """Write ``obj`` as indented JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)
    logger.info("Wrote %s", path)
    return path


def load_json(path: Path | str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_csv(
    rows: Sequence[Mapping[str, Any]],
    path: Path | str,
    fieldnames: Iterable[str] | None = None,
) -> Path:
    """Write a list of dicts as CSV; column order follows the first row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        logger.warning("No rows to write to %s", path)
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = list(fieldnames) if fieldnames else list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s (%d rows)", path, len(rows))
    return path


def save_text(text: str, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


_LOG_FORMAT = "%(asctime)s  %(name)-24s  %(levelname)-7s  %(message)s"


def setup_logging(log_path: Path | str | None = None, level: int = logging.INFO) -> None:
    """Configure console logging, optionally mirrored to ``log_path``.

    Safe to call from every script: handlers are added once per destination so
    repeated calls (or an import that already configured logging) do not
    duplicate lines.
    """
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(stream)

    if log_path is None:
        return
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(log_path.resolve())
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == resolved:
            return
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(file_handler)
    logger.info("Logging to %s", log_path)
