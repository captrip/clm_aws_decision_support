"""The cascade. DESIGN.md sections 5 and 6.

Per turn: build context, select an arm under budget, call it, verify. A 0
verdict escalates to a different arm; a None verdict commits and defers to the
sampled judge.

The rule that matters: every bandit update uses the context vector that arm saw
at selection time, not the final one. Arm 2 is chosen with the failure flag set
and retry_depth incremented, so its vector differs from arm 1's. Attributing
arm 2's vector to arm 1 corrupts learning in a way nothing surfaces.
"""

import time

from router import arms as arms_module
from router import config, verify


class Router:
    def __init__(
        self,
        bandit,
        context_builder,
        cost_estimator,
        store,
        client,
        converse_fn=arms_module.bedrock_converse,
        verify_fn=verify.check,
        observer=None,
    ):
        self.bandit = bandit
        self.context_builder = context_builder
        self.cost_estimator = cost_estimator
        self.store = store
        self.client = client
        self.converse_fn = converse_fn
        self.verify_fn = verify_fn
        self.observer = observer

    def _emit(self, event, **payload):
        """Fire a visualization event. Never let it affect the turn.

        A visualization that can break routing is worse than no visualization:
        the observer is a debugging aid, the turn is the product. Every failure
        mode -- a full queue, a disconnected browser, a bug in viz.py -- is
        swallowed here. The observer is duck-typed so policy.py keeps no
        dependency on the visualization module.
        """
        if self.observer is None:
            return
        try:
            self.observer.emit(event, payload)
        except Exception:  # noqa: BLE001 - deliberate, see docstring
            pass

    def _arm_state(self, arm, x):
        """This arm's p/e/ucb on the vector it was selected with.

        Not the turn's final vector: complete() rebuilds x per attempt, so
        attempt 2's differs from attempt 1's (DESIGN.md section 6). Costs about
        0.3ms for all seven arms and only runs when an observer is attached.
        """
        info = self.bandit.score(x)[arm]
        return {"p": info["p"], "e": info["e"], "ucb": info["ucb"]}

    def complete(
        self,
        conversation_id,
        query,
        turn_index=0,
        task_type="prose",
        expected=None,
        schema=None,
        conversation_tokens=0,
        budget=None,
    ):
        total_budget = (
            budget if budget is not None else config.DEFAULT_CONVERSATION_BUDGET
        )
        remaining = total_budget - self.store.get_spent(conversation_id)

        turn_id = None
        if self.observer is not None:
            try:
                turn_id = self.observer.next_turn_id()
            except Exception:  # noqa: BLE001 - an observer must not break a turn
                turn_id = None
        self._emit(
            "turn_start",
            turn_id=turn_id,
            conversation_id=conversation_id,
            query=query,
            task_type=task_type,
            budget=total_budget,
            remaining=remaining,
        )

        failed_arms = []
        tried = []
        attempts = []
        decision_ids = []
        turn_cost = 0.0
        prompt_tokens = max(1, len(query) // 4)

        for depth in range(config.MAX_CASCADE):
            x = self.context_builder.build(
                query=query,
                turn_index=turn_index,
                prompt_tokens=prompt_tokens,
                conversation_tokens=conversation_tokens,
                task_type=task_type,
                failed_arms=failed_arms,
                retry_depth=depth,
            )
            candidates = [a for a in config.ARM_NAMES if a not in tried]
            costs = self.cost_estimator.estimate_all(turn_index)
            arm, scores = self.bandit.select(x, candidates, costs, remaining)
            self._emit(
                "scored",
                turn_id=turn_id,
                depth=depth,
                chosen=arm,
                remaining=remaining,
                scores=scores,
            )

            if arm is None:
                status = "budget_exhausted" if depth == 0 else "exhausted"
                self._emit(
                    "turn_end",
                    turn_id=turn_id,
                    status=status,
                    cascade_depth=depth,
                    cost=turn_cost,
                    decision_ids=decision_ids,
                )
                return {
                    "text": attempts[-1]["text"] if attempts else "",
                    "arm": attempts[-1]["arm"] if attempts else None,
                    "cascade_depth": depth,
                    "cost": turn_cost,
                    "decision_ids": decision_ids,
                    "verdict": attempts[-1]["verdict"] if attempts else None,
                    "status": status,
                    "attempts": attempts,
                }

            tried.append(arm)
            prompt = self._format_prompt(query, attempts)

            started = time.perf_counter()
            try:
                text, usage = self.converse_fn(
                    self.client, arm, prompt, temperature=0.0
                )
                error = None
            except Exception as exc:  # noqa: BLE001 - recorded as a failure
                text, usage, error = "", {}, str(exc)
            latency_ms = (time.perf_counter() - started) * 1000.0

            # token_cost is a pure function of (arm, usage), so computing it
            # before the emit lets the event carry this attempt's real cost.
            # It used to emit cost=None here and compute below, which left the
            # UI with no per-attempt cost at all -- only turn_end's aggregate.
            # The accounting below is unchanged: this is the same value.
            cost = arms_module.token_cost(arm, usage)
            self._emit(
                "called",
                turn_id=turn_id,
                depth=depth,
                arm=arm,
                text=text,
                cost=cost,
                usage=usage,
                error=error,
                latency_ms=latency_ms,
            )

            turn_cost += cost
            remaining -= cost
            self.store.add_spend(conversation_id, cost)
            if not error:
                # An error observed no cost, so 0.0 is not a cost sample.
                # Feeding it in drives the LCB to EPSILON_COST, and paired with
                # learn=False below the arm stays both cheapest and fully
                # optimistic -- a permanent first pick that cannot be escaped.
                # add_spend and turn_cost above are correct as they stand: the
                # spend genuinely was 0.0.
                self.cost_estimator.update(arm, turn_index, cost)

            if error:
                # Infrastructure failure, not a bad answer. Escalate, but do
                # not teach the bandit that this arm is weak: bedrock_converse
                # already exhausted its throttling retries, so the fault is
                # environmental and the penalty would never decay.
                verdict = 0
                learn = False
            else:
                verdict = self.verify_fn(
                    task_type, text, expected=expected, schema=schema
                )
                learn = verdict is not None

            self._emit(
                "verified",
                turn_id=turn_id,
                depth=depth,
                arm=arm,
                verdict=verdict,
                learn=learn,
                task_type=task_type,
            )

            # Update with THIS arm's vector, before the next iteration
            # rebuilds a different one.
            if learn:
                if self.observer is not None:
                    before = self._arm_state(arm, x)
                    pulls_before = self.bandit.models[arm]["pulls"]
                self.bandit.update(arm, x, float(verdict))
                if self.observer is not None:
                    self._emit(
                        "updated",
                        turn_id=turn_id,
                        depth=depth,
                        arm=arm,
                        reward=float(verdict),
                        pulls_before=pulls_before,
                        pulls_after=self.bandit.models[arm]["pulls"],
                        before=before,
                        after=self._arm_state(arm, x),
                    )

            # verdict drives the cascade; logged_reward is what the DB durably
            # claims the bandit learned. They differ only on the error path,
            # where verdict=0 escalates but nothing was learned. Persisting the
            # 0 would let any rebuild-from-DB reapply the very penalty
            # learn=False suppressed, and record_reward could not undo it.
            # needs_judge stays 0: not learned from, and not for the judge.
            logged_reward = None if error else verdict

            decision_id = self.store.log_decision(
                conversation_id=conversation_id,
                turn_index=turn_index,
                arm=arm,
                context_vector=x,
                retry_depth=depth,
                task_type=task_type,
                cost=cost,
                reward=logged_reward,
                needs_judge=verdict is None,
                query=query,
                response_text=text,
            )
            decision_ids.append(decision_id)
            attempts.append(
                {
                    "arm": arm,
                    "text": text,
                    "verdict": verdict,
                    "cost": cost,
                    "error": error,
                    "decision_id": decision_id,
                }
            )

            if verdict is None:
                self._emit(
                    "turn_end",
                    turn_id=turn_id,
                    status="unverified",
                    cascade_depth=depth + 1,
                    cost=turn_cost,
                    decision_ids=decision_ids,
                )
                return {
                    "text": text,
                    "arm": arm,
                    "cascade_depth": depth + 1,
                    "cost": turn_cost,
                    "decision_ids": decision_ids,
                    "verdict": None,
                    "status": "unverified",
                    "attempts": attempts,
                }
            if verdict == 1:
                self._emit(
                    "turn_end",
                    turn_id=turn_id,
                    status="success",
                    cascade_depth=depth + 1,
                    cost=turn_cost,
                    decision_ids=decision_ids,
                )
                return {
                    "text": text,
                    "arm": arm,
                    "cascade_depth": depth + 1,
                    "cost": turn_cost,
                    "decision_ids": decision_ids,
                    "verdict": 1,
                    "status": "success",
                    "attempts": attempts,
                }

            failed_arms.append(arm)
            if remaining <= config.EPSILON_COST:
                self._emit(
                    "turn_end",
                    turn_id=turn_id,
                    status="budget_exhausted",
                    cascade_depth=depth + 1,
                    cost=turn_cost,
                    decision_ids=decision_ids,
                )
                return {
                    "text": text,
                    "arm": arm,
                    "cascade_depth": depth + 1,
                    "cost": turn_cost,
                    "decision_ids": decision_ids,
                    "verdict": 0,
                    "status": "budget_exhausted",
                    "attempts": attempts,
                }

        last = attempts[-1]
        self._emit(
            "turn_end",
            turn_id=turn_id,
            status="exhausted",
            cascade_depth=len(attempts),
            cost=turn_cost,
            decision_ids=decision_ids,
        )
        return {
            "text": last["text"],
            "arm": last["arm"],
            "cascade_depth": len(attempts),
            "cost": turn_cost,
            "decision_ids": decision_ids,
            "verdict": 0,
            "status": "exhausted",
            "attempts": attempts,
        }

    def _format_prompt(self, query, attempts):
        """Include prior failed answers, as the notebooks' format_prompt does."""
        prompt = query
        failed = [a for a in attempts if a["verdict"] == 0 and a["text"]]
        if failed:
            prompt += (
                "\n\nNote: the following previous attempts were incorrect. "
                "Provide a different answer:\n"
            )
            for i, attempt in enumerate(failed, start=1):
                prompt += f"- Attempt {i} (by {attempt['arm']}): {attempt['text']}\n"
        return prompt

    def record_reward(self, decision_id, reward):
        """Apply a delayed reward using the vector stored at selection time.

        Idempotent: a decision that already carries a reward is a no-op.
        A retried webhook or a second judge pass would otherwise update the
        bandit twice from one observation, double-counting it. Returns True
        if the reward was applied.

        Also refuses a vector built under a different projection. Applying one
        writes old-feature-space values into the current bandit's b, which is
        worse than dropping the reward: it silently undoes the reset the version
        bump was for. Same False signal as the idempotency case -- not applied.

        And refuses an infrastructure-error row. Those carry reward NULL so the
        error is not recorded as a bad answer, but NULL alone no longer means
        "awaiting feedback" -- only needs_judge distinguishes them. Without this
        guard a client posting a rating for every id in decision_ids would apply
        exactly the penalty complete() suppressed, to an arm that returned
        nothing at all.
        """
        row = self.store.get_decision(decision_id)
        if row is None:
            raise KeyError(f"no decision with id {decision_id}")
        if row["reward"] is not None:
            self._emit_refusal(decision_id, row, reward, "already_rewarded")
            return False
        if not row["needs_judge"]:
            self._emit_refusal(decision_id, row, reward, "not_awaiting_judge")
            return False
        if row["projection_version"] != config.PROJECTION_VERSION:
            self._emit_refusal(decision_id, row, reward, "projection_mismatch")
            return False
        before = None
        pulls_before = None
        if self.observer is not None:
            before = self._arm_state(row["arm"], row["context_vector"])
            pulls_before = self.bandit.models[row["arm"]]["pulls"]
        self.bandit.update(row["arm"], row["context_vector"], float(reward))
        self.store.attach_reward(decision_id, int(reward))
        if self.observer is not None:
            self._emit(
                "delayed_update",
                decision_id=decision_id,
                arm=row["arm"],
                reward=float(reward),
                applied=True,
                pulls_before=pulls_before,
                pulls_after=self.bandit.models[row["arm"]]["pulls"],
                before=before,
                after=self._arm_state(row["arm"], row["context_vector"]),
            )
        return True

    def _emit_refusal(self, decision_id, row, reward, reason):
        """delayed_update for a record_reward guard: nothing was applied.

        pulls is read once and reported as both before and after, because no
        update happened -- the UI needs the refusal visible, not a fake delta.
        """
        if self.observer is None:
            return
        try:
            pulls = self.bandit.models[row["arm"]]["pulls"]
        except Exception:  # noqa: BLE001 - a stored arm may no longer exist
            pulls = None
        self._emit(
            "delayed_update",
            decision_id=decision_id,
            arm=row["arm"],
            reward=float(reward),
            applied=False,
            pulls_before=pulls,
            pulls_after=pulls,
            before=None,
            after=None,
            reason=reason,
        )
