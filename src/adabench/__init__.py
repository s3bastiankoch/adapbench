"""adabench: spend benchmark measurements only where they are needed.

An online scheduler allocates the next measurement to the benchmark nearest a decision
and early-stops each one the moment its verdict is certain, under an anytime-valid
false-claim guarantee that holds despite the adaptive allocation and continuous peeking.
No historical data. The deployable prototype is :class:`adabench.scheduler.Scheduler`."""

from .cs import BettingCS, claims_cs
from .trace import SharedTrace, load_shared
from .datasets import BIG_REL, EPS_D, load, project_of
from .scheduler import FastScheduler, ReplayRunner, Runner, Scheduler

__all__ = [
    "Scheduler", "FastScheduler", "ReplayRunner", "Runner",
    "BettingCS", "claims_cs",
    "SharedTrace", "load_shared",
    "BIG_REL", "EPS_D", "load", "project_of",
]
