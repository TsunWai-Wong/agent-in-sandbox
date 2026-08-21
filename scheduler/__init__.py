"""Scheduled, isolated agent runs.

One mechanism behind both patterns: a trigger fires, a fresh session runs and
reports a typed outcome, and a policy decides whether anyone hears about it.

    digest  = DeliverAlways()        — the run's output is the deliverable
    monitor = DeliverOnTransition()  — silent success, loud failure
"""

from .clock import Scheduler
from .delivery import Channel, ConsoleChannel, WebhookChannel
from .outcome import REPORT_INSTRUCTION, Outcome, OutcomeSink, StateRecord
from .policy import DeliverAlways, DeliverOnTransition, Decision, Policy
from .runner import run_once
from .store import SqliteStore
from .task import ScheduledTask
from .trigger import CronTrigger, IntervalTrigger, OnceTrigger, Trigger

__all__ = [
    "Channel",
    "ConsoleChannel",
    "CronTrigger",
    "Decision",
    "DeliverAlways",
    "DeliverOnTransition",
    "IntervalTrigger",
    "OnceTrigger",
    "Outcome",
    "OutcomeSink",
    "Policy",
    "REPORT_INSTRUCTION",
    "ScheduledTask",
    "Scheduler",
    "SqliteStore",
    "StateRecord",
    "Trigger",
    "WebhookChannel",
    "run_once",
]
