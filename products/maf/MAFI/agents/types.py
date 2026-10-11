from typing import TypedDict,Optional,Any,List,Dict
from MAFI.types.understanding_states import SummarizerOutput
class ActionExtractorNodeOutput(TypedDict,total= False):
    intent_understanding: SummarizerOutput

class IUSwarmState(TypedDict,total = False):
    updated_query: str
    iu_goals: str
    iu_risk: str
    intent_understanding: str

class IU_Output(TypedDict, total = False):
    intent_understanding: SummarizerOutput

class SummaryState(TypedDict, total = False):
    answer: str
    citation: Optional[List[str]]

class PlannerState(TypedDict, total = False):
    tool_results_summary: Optional[List[str]]