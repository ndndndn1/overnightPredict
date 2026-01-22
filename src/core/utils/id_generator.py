"""ID generation utilities."""

import secrets
import time
from datetime import datetime
from typing import Optional


def generate_id(prefix: str = "", length: int = 12) -> str:
    """Generate a unique ID with optional prefix.

    Args:
        prefix: Optional prefix for the ID.
        length: Length of the random part.

    Returns:
        A unique ID string.
    """
    random_part = secrets.token_hex(length // 2)
    if prefix:
        return f"{prefix}_{random_part}"
    return random_part


def generate_session_id() -> str:
    """Generate a unique session ID.

    Format: sess_<timestamp>_<random>

    Returns:
        A unique session ID.
    """
    timestamp = int(time.time() * 1000) % 1000000  # Last 6 digits of ms timestamp
    random_part = secrets.token_hex(4)
    return f"sess_{timestamp:06d}_{random_part}"


def generate_prediction_id() -> str:
    """Generate a unique prediction ID.

    Returns:
        A unique prediction ID.
    """
    return generate_id(prefix="pred")


def generate_question_id() -> str:
    """Generate a unique question ID.

    Returns:
        A unique question ID.
    """
    return generate_id(prefix="q")


def generate_task_id() -> str:
    """Generate a unique task ID.

    Returns:
        A unique task ID.
    """
    return generate_id(prefix="task")


def generate_group_id() -> str:
    """Generate a unique group ID for session grouping.

    Returns:
        A unique group ID.
    """
    return generate_id(prefix="grp")


def timestamp_id(prefix: Optional[str] = None) -> str:
    """Generate a timestamp-based ID.

    Format: <prefix>_<YYYYMMDD>_<HHMMSS>_<random>

    Args:
        prefix: Optional prefix.

    Returns:
        A timestamp-based unique ID.
    """
    now = datetime.utcnow()
    date_part = now.strftime("%Y%m%d_%H%M%S")
    random_part = secrets.token_hex(3)

    if prefix:
        return f"{prefix}_{date_part}_{random_part}"
    return f"{date_part}_{random_part}"
