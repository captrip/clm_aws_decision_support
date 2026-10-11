from typing import Any, Dict, List, Optional, Tuple, Set, TypedDict, Literal

JSONDict = Dict[str, Any]
Span = Tuple[int, int]

BreakdownType = Literal[
    "intent",
    "request",
    "questions",
    "constraints",
    "context",
    "other"
]
ParserStatus = Literal[
    "ok",
    "error",
    "needs_clarification"
]

ActionStatus = Literal[
    "ok",
    "no_action",
    "needs_clarification",
    "error"
]
class BreakdownItem(TypedDict,total=False):
    type: BreakdownType
    text: str
    span: Span
    id: str

class ParserIntent(TypedDict,total=False):
    label: str
    confidence: float

class EntityPreview(TypedDict,total=False):
    value: str
    span: Span
    entity_type: str

class ParserOutput(TypedDict,total=False):
    status: ParserStatus
    intents: List[ParserIntent]
    entities: List[EntityPreview]
    breakdown: List[BreakdownItem]
    missing: List[str]
    coverage: float
    confidence: float

class ActionItem(TypedDict,total=False):
    breakdown_id: str
    action: str
    parameters: JSONDict
    priority: int
    dependencies: List[str]

class ActionExtractorOutput(TypedDict,total=False):
    status: ActionStatus
    actions: List[ActionItem]
    missing: List[str]
    confidence: float
    coverage: float


ConstraintStatus = Literal[
    "ok",
    "error",
    "needs_clarification"
]

ConstraintType = Literal[
    "budget",
    "deadline",
    "scope",
    "security",
    "compliance",
    "quality",
]
ConstraintHardness = Literal[
    "must",
    "should"
]
ConstraintSource = Literal[
    "user",
    "policy",
    "inferred"
]
class ConstraintItem(TypedDict,total = False):
    type: ConstraintType
    rule: str
    hardness: ConstraintHardness
    source: ConstraintSource
    span: Span

class ConstraintExtractorOutput(TypedDict,total = False):
    constraints: List[ConstraintItem]
    status: ConstraintStatus
    confidence: float
    coverage: float
    missing: List[str]

ContextStatus = Literal[
    "ok",
    "error",
    "needs_clarification"
]

class ContextBody(TypedDict, total = False):
    domain: str
    background: str
    user_prefs: JSONDict
    stakeholders: List[str]

class ContextExtractorOutput(TypedDict, total = False):
    context: ContextBody
    status: ContextStatus
    confidence: float
    covarage: float
    missing: List[str]

EntityStatus = Literal[
    "ok",
    "error",
    "needs_clarification"
]

class EntityItem(TypedDict, total = False):
    type: str
    value: str
    canonical: str
    span: Span

class EntityExtractorOutput(TypedDict, total = False):
    entities: List[EntityItem]
    status: EntityStatus
    confidence: float
    covarage: float
    missing: List[str]

RiskStatus = Literal[
    "ok",
    "no_risk",
    "needs_clarification",
    "error"
]
RiskType = Literal[
    "feasibility",
    "data",
    "privacy",
    "security",
    "timeline",
    "cost",
    "compliance",
    "other"
]
RiskSeverity = Literal[
    "low",
    "medium",
    "high"
]
class RiskItem(TypedDict, total = False):
    type: RiskType
    severity: RiskSeverity
    description: str
    mitigation: List[str]

class RiskAssessorOutput(TypedDict, total = False):
    risks: List[RiskItem]
    status: RiskStatus
    confidence: float
    covarage: float
    missing: List[str]

ClarifierStatus = Literal[
    "ok",
    "no_questions",
    "error"
]

class ClarifierQuestions(TypedDict, total = False):
    id: str
    text: str
    reason: str
    blocking: bool

class ClarifierOutput(TypedDict, total = False):
    questions: List[ClarifierQuestions]
    status: ClarifierStatus
    confidence: float

SummarizerStatus = Literal["ready","error",
    "needs_clarification"]

class SummarizerOutput(TypedDict, total = False):
    intent_summary: str
    actions: List[ActionItem]
    constraints: List[ConstraintItem]
    context: ContextBody
    entities: List[EntityItem]
    risks: List[RiskItem]
    open_questions: List[ClarifierQuestions]
    status: SummarizerStatus
    confidence: float


