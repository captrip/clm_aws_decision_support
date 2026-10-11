"""Backwards-compatible import location. The judges now live in sentinal.judges."""
from sentinal.judges.builtin import HeuristicJudge

__all__ = ["HeuristicJudge"]
