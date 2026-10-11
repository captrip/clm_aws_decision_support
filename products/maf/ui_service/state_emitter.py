import copy
from typing import Optional, Dict, Any


class StateDiffEmitter:

    def __init__(self):
        self.previous_state_by_source = {}
        self.emit_state_snapshot = False
        self.llm_metrics_history = []

    def emit(self, data: dict):
        # Placeholder for actual emit logic
        pass

    def on_state_update(self, run_id: str, state: dict):
        source = state.get("source", "default")
        previous_state = self.previous_state_by_source.get(source, {})

        # -------------------------------
        # HITL
        # -------------------------------
        current_hitl = state.get("hitl_request")
        previous_hitl = previous_state.get("hitl_request")

        if current_hitl is not None and current_hitl != previous_hitl:
            self.emit({
                "type": "hitl.request",
                "run_id": run_id,
                "payload": current_hitl
            })

        # -------------------------------
        # Intent Understanding
        # -------------------------------
        current_intent = state.get("intent_understanding")
        previous_intent = previous_state.get("intent_understanding")

        if current_intent != previous_intent and current_intent is not None:
            self.emit({
                "type": "intent.result",
                "run_id": run_id,
                "payload": current_intent
            })

        # -------------------------------
        # Action Extraction
        # -------------------------------
        current_action = state.get("action_extraction")
        previous_action = previous_state.get("action_extraction")

        if current_action != previous_action and current_action is not None:
            self.emit({
                "type": "action_extraction.result",
                "run_id": run_id,
                "payload": current_action
            })

        # -------------------------------
        # Planner
        # -------------------------------
        current_planner = state.get("planner") or {}
        previous_planner = previous_state.get("planner") or {}

        current_skills = current_planner.get("executed_skills")
        previous_skills = previous_planner.get("executed_skills")

        if current_skills is not None and current_skills != previous_skills:
            if isinstance(current_skills, list) and isinstance(previous_skills, list):
                new_skills = current_skills[len(previous_skills):]
                if new_skills:
                    self.emit({
                        "type": "planner.result",
                        "run_id": run_id,
                        "payload": new_skills
                    })
            elif isinstance(current_skills, list) and previous_skills is None:
                self.emit({
                    "type": "planner.result",
                    "run_id": run_id,
                    "payload": current_skills
                })
            else:
                self.emit({
                    "type": "planner.result",
                    "run_id": run_id,
                    "payload": current_skills
                })

        # -------------------------------
        # Tool Results Summary (Subagent mode)
        # -------------------------------
        if "planner" not in state and "tool_results_summary" in state:
            current_summary = state.get("tool_results_summary")
            previous_summary = previous_state.get("tool_results_summary")

            if current_summary is not None and current_summary != previous_summary:
                payload = current_summary

                if isinstance(current_summary, list) and isinstance(previous_summary, list):
                    payload = current_summary[len(previous_summary):]

                if payload:
                    self.emit({
                        "type": "tool.summary",
                        "run_id": run_id,
                        "payload": payload
                    })

        # -------------------------------
        # Planner Tool Summary (nested)
        # -------------------------------
        current_summary = current_planner.get("tool_results_summary")
        previous_summary = previous_planner.get("tool_results_summary")

        if current_summary is not None and current_summary != previous_summary:
            if isinstance(current_summary, list) and isinstance(previous_summary, list):
                new_summaries = current_summary[len(previous_summary):]
                if new_summaries:
                    self.emit({
                        "type": "tool.summary",
                        "run_id": run_id,
                        "payload": new_summaries
                    })
            elif isinstance(current_summary, list) and previous_summary is None:
                self.emit({
                    "type": "tool.summary",
                    "run_id": run_id,
                    "payload": current_summary
                })
            else:
                self.emit({
                    "type": "tool.summary",
                    "run_id": run_id,
                    "payload": current_summary
                })

        # -------------------------------
        # Optional full snapshot
        # -------------------------------
        if self.emit_state_snapshot:
            self.emit({
                "type": "state.snapshot",
                "run_id": run_id,
                "payload": state
            })

        # -------------------------------
        # Merge state
        # -------------------------------
        self.previous_state_by_source[source] = self._merge_state(previous_state, state)

    # ---------------------------------------------------
    # Deep Merge
    # ---------------------------------------------------
    def _merge_state(self, base: Optional[dict], update: Optional[dict]) -> dict:
        """Deep-merge state without dropping previously seen keys."""
        merged = copy.deepcopy(base) if base else {}

        for key, value in (update or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_state(merged.get(key), value)
            else:
                merged[key] = copy.deepcopy(value)

        return merged

    # ---------------------------------------------------
    # LLM Metrics
    # ---------------------------------------------------
    def on_llm_metrics(
        self,
        run_id: str,
        llm_run_id: str,
        latency_ms: Optional[float],
        token_usage: Dict[str, int],
        model_name: Optional[str] = None,
        agent_name: Optional[str] = None,
    ):
        """Emit LLM token usage and latency metrics with agent name."""
        metrics = {
            "run_id": run_id,
            "llm_run_id": llm_run_id,
            "latency_ms": round(latency_ms, 2) if latency_ms else None,
            "token_usage": token_usage,
            "model_name": model_name,
            "agent_name": agent_name,
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0),
        }

        self.llm_metrics_history.append(metrics)

        self.emit({
            "type": "llm.metrics",
            "run_id": run_id,
            "payload": metrics
        })

    # ---------------------------------------------------
    # Aggregation
    # ---------------------------------------------------
    def get_aggregated_metrics(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        filtered_metrics = self.llm_metrics_history

        if run_id:
            filtered_metrics = [
                m for m in self.llm_metrics_history if m["run_id"] == run_id
            ]

        if not filtered_metrics:
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "avg_latency_ms": 0,
                "total_latency_ms": 0,
            }

        total_calls = len(filtered_metrics)
        total_tokens = sum(m["total_tokens"] for m in filtered_metrics)
        total_prompt_tokens = sum(m["prompt_tokens"] for m in filtered_metrics)
        total_completion_tokens = sum(m["completion_tokens"] for m in filtered_metrics)

        latencies = [
            m["latency_ms"]
            for m in filtered_metrics
            if m["latency_ms"] is not None
        ]

        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        total_latency = sum(latencies) if latencies else 0

        return {
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "avg_latency_ms": round(avg_latency, 2),
            "total_latency_ms": round(total_latency, 2),
            "min_latency_ms": round(min(latencies), 2) if latencies else 0,
            "max_latency_ms": round(max(latencies), 2) if latencies else 0,
        }

    # ---------------------------------------------------
    # Clear
    # ---------------------------------------------------
    def clear_metrics(self):
        self.llm_metrics_history = []