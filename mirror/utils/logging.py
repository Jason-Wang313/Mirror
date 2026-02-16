"""
Structured logging for all API calls.

Every API call is logged to JSONL format for reproducibility and analysis.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class APILogger:
    """
    Logger for API calls. Writes structured JSONL logs.

    Each log entry contains the full request and response data,
    ensuring we never lose an API call.
    """

    def __init__(self, log_dir: str = "results/api_logs", experiment: str = "default"):
        """
        Initialize API logger.

        Args:
            log_dir: Directory to store log files
            experiment: Experiment name for the log file
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{timestamp}_{experiment}.jsonl"

    def log_request(
        self,
        model: str,
        messages: list[dict],
        response: dict,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Log a single API request and response.

        Args:
            model: Model name used
            messages: Input messages sent
            response: Response dict from the provider
            metadata: Optional experiment-specific metadata
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "messages": messages,
            "response": response,
            "metadata": metadata or {},
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    def log_error(
        self,
        model: str,
        messages: list[dict],
        error: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Log a failed API request.

        Args:
            model: Model name attempted
            messages: Input messages sent
            error: Error message
            metadata: Optional experiment-specific metadata
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "messages": messages,
            "error": error,
            "metadata": metadata or {},
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
