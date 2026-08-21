"""Where a delivered run goes.

One method, so the runner never knows the difference between a print and a POST.
Pinning the channel per task is what lets the same digest go to Slack in
production and to stdout while you are still writing it.
"""

import json
import logging
import os
import urllib.request
from typing import Protocol

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class Channel(Protocol):
    def send(self, subject: str, body: str) -> None:
        """Deliver one message, or raise. Raising is not fatal — see run_once."""


class ConsoleChannel:
    """Prints. The channel to develop against, and the one tests assert on."""

    def send(self, subject: str, body: str) -> None:
        print(f"\n=== {subject} ===\n{body}\n", flush=True)


class WebhookChannel:
    """POST {"text": ...} to a URL held in an environment variable.

    Slack-shaped by default, which most incoming-webhook endpoints accept. The
    URL is resolved at construction rather than at send time on purpose: a
    webhook that is only discovered to be missing at 8am is a digest nobody
    gets and nobody notices.
    """

    def __init__(self, url_env: str, timeout: float = 10.0) -> None:
        load_dotenv()
        url = os.getenv(url_env)
        if not url:
            raise ValueError(f"{url_env} is not set; cannot build a WebhookChannel")
        self._url = url
        self._timeout = timeout

    def send(self, subject: str, body: str) -> None:
        payload = json.dumps({"text": f"*{subject}*\n{body}"}).encode()
        request = urllib.request.Request(
            self._url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            response.read()
