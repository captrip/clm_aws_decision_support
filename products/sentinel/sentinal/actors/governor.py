"""
Governor - Monitors and intervenes during execution

Responsibilities:
- Monitor execution quality
- Decide when to intervene
- Issue instructions (RETRY, IMPROVE_QUALITY, etc.)
- Apply policies
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from sentinal.actors.actor import BaseActor
from sentinal.actors.judges import HeuristicJudge
from sentinal.core.trajectory import (
    SEVERITY, Correction, CorrectionFlag, JudgeContext, TrajectoryJudge,
)
from sentinal.telemetry.dashboard import activity_bus

from sentinal.core.contracts.execution_observations import (
    ExecutionErrored,
)

from sentinal.core.contracts.execution_signals import (
    RetryExecution,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentinal.core.execution.session import ExecutionSession


@dataclass
class _JudgingState:
    """Everything judging one session needs, owned by the `govern_session` call
    that governs it. Deliberately NOT stored on the Governor: the Governor is
    per-runtime and can govern several sessions at once, on different threads.
    """

    judges: list[TrajectoryJudge]
    last_len: int = 0
    """Trajectory length at the last judgment (the growth re-judge trigger)."""
    last_at: float = 0.0
    """Elapsed seconds at the last judgment (the staleness re-judge trigger)."""
    last_verdict: dict[int, Any] = field(default_factory=dict)
    """judge index -> (flag, note, step_index) of its last non-empty verdict, so
    a stateless judge repeating itself on an unchanged trajectory is suppressed."""
    warned: set[tuple[int, str]] = field(default_factory=set)
    """(judge index, reason) pairs already warned about, so a malformed return is
    logged once per judge instead of on every poll."""


class Governor(BaseActor):
    """
    Governor actor - monitors and can intervene.

    Responsibilities:
    - Monitor quality in real-time
    - Issue instructions to Actor
    - Decide when to retry
    - Apply policies
    """

    MAX_RETRIES = 3

    def __init__(self, *args, quality_threshold: float = 0.7,
                 judge: TrajectoryJudge | None = None,
                 judges: Sequence[TrajectoryJudge] | None = None,
                 rejudge_interval: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if judge is not None and judges is not None:
            raise ValueError(
                "pass either judge= (one judge) or judges= (several), not both")
        self.quality_threshold = quality_threshold
        self._interventions = []
        self._judges: list[TrajectoryJudge] = (
            list(judges) if judges is not None
            else [judge] if judge is not None
            else [HeuristicJudge()]
        )
        self._rejudge_interval = rejudge_interval
        # Per-session judge overrides, so a per-call judges= never leaks into a
        # later run and two concurrent runs cannot clobber each other.
        self._session_judges: dict[Any, list[TrajectoryJudge]] = {}
        self._session_judges_lock = threading.Lock()

    def set_judges(self, judges: Sequence[TrajectoryJudge]) -> None:
        """Replace this Governor's default judges.

        Kept for backwards compatibility. NOTE: this mutates Governor-wide state
        and so affects every session this Governor governs afterwards — prefer
        `set_session_judges()` for a single run.
        """
        self._judges = list(judges)

    def set_session_judges(
        self, session_id: Any, judges: Sequence[TrajectoryJudge] | None
    ) -> None:
        """Attach judges to ONE session, before `govern_session` starts.

        This is how the runtime threads `execute(session, judges=...)` through
        without touching `run_actors`' signature. `judges=None` clears any
        override, so the session falls back to the Governor's own judges.
        """
        with self._session_judges_lock:
            if judges is None:
                self._session_judges.pop(session_id, None)
            else:
                self._session_judges[session_id] = list(judges)

    def _resolve_judges(self, session_id: Any) -> list[TrajectoryJudge]:
        """The judge list for this one session: its override if it has one,
        otherwise the Governor's constructor-supplied (or default) judges."""
        with self._session_judges_lock:
            override = self._session_judges.get(session_id)
        return list(override) if override is not None else list(self._judges)

    async def receive(
        self,
        message,
    ):
        """
        Receive observations and decide on interventions.

        Args:
            message: Incoming observation
        """
        if not isinstance(
            message,
            ExecutionErrored,
        ):
            return

        # Temporary policy: retry on error
        await self.publish(
            RetryExecution(
                execution_id=message.execution_id,
                sender=self.ref,
                data=1,
            )
        )

    async def govern_session(
        self,
        session: 'ExecutionSession',
        poll_interval: float = 0.1,
    ) -> None:
        """
        Monitor session and intervene when needed.

        Args:
            session: Execution session to govern
            poll_interval: How often to check (seconds)
        """
        logger.info(f"Governor {self.ref.name} started")

        # Resolve this session's judges ONCE, into a local. Nothing here touches
        # shared mutable state, so a concurrent govern_session() on another
        # thread governs its own session with its own judges.
        state = _JudgingState(judges=self._resolve_judges(session.session_id))
        t0 = time.monotonic()

        sid = str(session.session_id)
        activity_bus.emit("Governor", "governing", status="governing", session_id=sid)

        try:
            while not session.is_terminal():
                trajectory = session.get_trajectory()
                elapsed = time.monotonic() - t0
                # Re-judge when the trajectory grew, or when enough time has
                # passed that a time-based judge deserves another look — without
                # the second condition a stalled trajectory is never judged
                # again. Repeated identical verdicts are de-duplicated below.
                grew = len(trajectory) > state.last_len
                stale = (elapsed - state.last_at) >= self._rejudge_interval
                if grew or stale:
                    state.last_len = len(trajectory)
                    state.last_at = elapsed
                    self._judge_and_arbitrate(session, trajectory, elapsed, state)

                # Check actor status
                memory = session.state.get_all_memory()
                actor_status = memory.get("actor_status")

                # Intervention logic
                if actor_status == "failed":
                    await self._handle_failure(session)

                elif actor_status == "completed":
                    await self._check_quality(session)

                # Poll interval
                await asyncio.sleep(poll_interval)
        finally:
            # Don't let the per-session override dict grow forever.
            self.set_session_judges(session.session_id, None)

        logger.info(f"Governor {self.ref.name} complete ({len(self._interventions)} interventions)")

    async def _handle_failure(self, session: 'ExecutionSession') -> None:
        """
        Handle execution failure.

        Args:
            session: Execution session
        """
        # Check retry count
        retry_count = session.state.current.runtime.retry_count

        if retry_count < self.MAX_RETRIES:
            logger.warning(f"Governor issuing RETRY (attempt {retry_count + 1}/{self.MAX_RETRIES})")

            session.add_instruction("RETRY")

            self._interventions.append({
                "type": "retry",
                "reason": "execution_failed",
                "retry_count": retry_count,
            })

    async def _check_quality(self, session: 'ExecutionSession') -> None:
        """
        Check execution quality and intervene if needed.

        Args:
            session: Execution session
        """
        memory = session.state.get_all_memory()

        # Compute quality (simple heuristic)
        quality = self._compute_quality(memory)

        sid = str(session.session_id)
        activity_bus.emit("Governor", "quality.score", status="governing", detail=f"{quality:.2f}", session_id=sid)

        if quality < self.quality_threshold:
            logger.warning(f"Governor: quality low ({quality:.2f}), issuing IMPROVE_QUALITY")

            session.add_instruction("IMPROVE_QUALITY")
            session.add_instruction("USE_BETTER_TOOLS")

            self._interventions.append({
                "type": "quality_improvement",
                "quality": quality,
                "threshold": self.quality_threshold,
            })

    def _compute_quality(self, memory: dict) -> float:
        """
        Compute quality score from memory.

        Args:
            memory: Runtime memory

        Returns:
            Quality score (0-1)
        """
        quality = 0.0

        # Check for result
        if "result" in memory:
            result = memory["result"]

            if result:
                quality += 0.5

                # Check result has content
                if hasattr(result, 'output'):
                    if result.output:
                        quality += 0.3

        # Check for errors
        if "error" not in memory:
            quality += 0.2

        return min(quality, 1.0)

    def _build_context(self, session, trajectory, elapsed: float) -> JudgeContext:
        """Assemble what judges are allowed to see. Defensive about the session
        shape: tests and adapters pass sessions without a .context."""
        try:
            memory = session.state.get_all_memory()
        except Exception:
            memory = {}
        context = getattr(session, "context", None)
        raw_input = getattr(context, "input", None)
        return JudgeContext(
            trajectory=trajectory,
            elapsed_s=elapsed,
            memory=memory if isinstance(memory, dict) else {},
            input=raw_input if isinstance(raw_input, dict) else {},
        )

    def _judge_and_arbitrate(self, session, trajectory, elapsed: float,
                            state: _JudgingState) -> None:
        """Run every judge, record every finding, push only the most severe."""
        ctx = self._build_context(session, trajectory, elapsed)

        def warn_once(idx: int, reason: str, message: str) -> None:
            # Spec: a malformed return is "logged once per judge", not on every
            # poll for the rest of the run.
            if (idx, reason) in state.warned:
                return
            state.warned.add((idx, reason))
            logger.warning(message)

        verdicts: list[tuple[str, Correction]] = []
        for idx, j in enumerate(state.judges):
            name = getattr(j, "name", None) or type(j).__name__
            try:
                c = j.judge(ctx)
            except Exception as ex:   # a broken judge must not crash the run
                logger.warning(f"Governor judge {name!r} failed: {ex}")
                continue
            if c is None:
                continue
            if not isinstance(c, Correction):
                warn_once(idx, "type",
                          f"Governor judge {name!r} returned {type(c).__name__}, "
                          "expected Correction or None — ignoring")
                continue
            # CorrectionFlag is a str Enum, so Correction(flag="stop") slips
            # past the isinstance check above. Arbitration and recording run
            # outside the per-judge try/except, so an unknown flag would kill
            # the Governor's thread — validate the flag here instead.
            if not isinstance(c.flag, CorrectionFlag) or c.flag not in SEVERITY:
                warn_once(idx, "flag",
                          f"Governor judge {name!r} returned Correction with an "
                          f"unknown flag {c.flag!r}, expected a CorrectionFlag "
                          "— ignoring")
                continue
            if c.flag == CorrectionFlag.CONTINUE:
                continue
            # The built-in judges are stateless predicates: while their condition
            # holds they return the SAME correction every poll. The time-based
            # re-judge trigger would turn that into one duplicate nudge per
            # second, so suppress a verdict identical to this judge's last one.
            # A verdict that CHANGES still gets through.
            fingerprint = (c.flag, c.note, c.step_index)
            if state.last_verdict.get(idx) == fingerprint:
                continue
            state.last_verdict[idx] = fingerprint
            verdicts.append((name, c))

        if not verdicts:
            return

        # Highest severity wins; ties break by registration order because max()
        # keeps the first maximal element.
        winner_idx = max(range(len(verdicts)),
                         key=lambda i: SEVERITY[verdicts[i][1].flag])

        for i, (name, c) in enumerate(verdicts):
            self._interventions.append({
                "type": "correction",
                "judge": name,
                "flag": c.flag.value,
                "note": c.note,
                "meta": dict(c.meta),
                "pushed": i == winner_idx,
            })

        name, winner = verdicts[winner_idx]
        session.push_correction(winner)
        activity_bus.emit(
            "Governor", "correction",
            status="governing",
            detail=f"{name}:{winner.flag.value}:{winner.note or ''}",
            session_id=str(session.session_id),
        )

    def get_interventions(self) -> list:
        """Get all interventions made."""
        return self._interventions.copy()
