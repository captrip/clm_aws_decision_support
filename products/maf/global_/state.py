from typing import TypedDict,Annotated
from langgraph.graph import add_messages

from MAFI.agents.types import PlannerState, SummaryState


class Orchestrator_Global_State(TypedDict):
    user_id: str
    user_query: str
    updated_query: str
    next_step: str
    reasoning: str
    user_clearification: str
    intent_understanding: list[dict]
    action_extraction: dict
    knowledge_base: list[dict]
    message: Annotated[str, add_messages]
    plan: list[any]
    selected_skills: list[str]
    selected_skills_id: list[str]
    analysis: list[str]
    executions: list[dict]
    initial_analysis: list[str]
    planner: PlannerState
    post_tool_reflection: list[str]
    final_answer: list[str]
    next_steps: list[str]
    summary: SummaryState
    session_id: str