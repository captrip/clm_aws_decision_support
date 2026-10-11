from typing import Any, Dict, List, Optional, TypedDict
from MAFI.agents.types import PlannerState
from MAFI.types.understanding_states import ActionExtractorOutput, SummarizerOutput

class SummaryState(TypedDict, total = False):
    answer: str
    citation: Optional[List[str]]
class PlannerNodeInput(TypedDict):
    action_extraction: ActionExtractorOutput

class PlannerNodeOutput(TypedDict):
    planner : PlannerState

class PlanStep(TypedDict,total = False):
    step_id : Optional[str]
    action_reference : Optional[str]
    description : Optional[str]
    tool : Optional[str]
    inputs : Optional[Dict[str,Any]]
    expected_output : Optional[str]
    status : Optional[str]
class PlannerGraphState(TypedDict,total = False):
    plan : List[PlanStep]
    analysis : Optional[str]
    selected_skills : Optional[List[str]]
    selected_skill_ids : Optional[List[str]]
    sills_catalog_used : Optional[List[Dict[str,Any]]]
    tools_to_call : List[str]
    errors : Optional[List[str]]
    executions : Optional[List[Dict[str,Any]]]
    executed_skills : Optional[List[str]]
    tool_result_summary : Optional[List[str]]
    
class SummaryNodeInput(TypedDict):
    user: str
    intent_understanding : SummarizerOutput
    planner : PlannerState

class SummaryNodeOutput(TypedDict,total = False):
    summary: SummaryState