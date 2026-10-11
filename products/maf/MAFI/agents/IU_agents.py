import json
from typing import TypedDict, Dict, Any, List
import aiosqlite

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from MAFI.agent_factory.base_MAFI_agent import BaseMAFIAgent
from MAFI.agent_factory.types.state_manager import StateManager
from MAFI.agents.types import ActionExtractorNodeOutput, IU_Output,IUSwarmState
from MAFI.prompts.prompts import ACTION_EXTRACTOR_PROMPT,ACTION_EXTRACTOR_SYSTEM_PROMPT, IU_GOAL_SYSTEM_PROMPT, IU_RISK_SYSTEM_PROMPT
from MAFI.types.understanding_states import ActionExtractorOutput
from MAFI.util.extraction_util import llm_output_parser
from config.config import get_shared_llm
from global_.state import Orchestrator_Global_State

class ActionExtractorAgent(BaseMAFIAgent):
    def __init__(
        self,
        name: str = "Action Extractor",
        system_prompt: str = ACTION_EXTRACTOR_SYSTEM_PROMPT,
        human_templete: str = ACTION_EXTRACTOR_PROMPT,
        llm: Any = None,
        tools: List[Any] = [],
        state_manager_: Any = StateManager(),
        default_scope: str = "default",
    ):
        if llm is None:
            llm = get_shared_llm()

        super().__init__(
            name=name,
            system_prompt=system_prompt,
            human_templete=human_templete,
            llm=llm,
            tools=tools,
            state_manager_=state_manager_,
            default_scope=default_scope,
        )

    async def __call__(self, state:Orchestrator_Global_State)->ActionExtractorNodeOutput:
        
        action_extraction_input = {
            "user_query": state.get('user_query','{}'),
            "memory_info": state.get('memory_info','{}')
        }
        result = await super().arun(action_extraction_input)
        thoughts,state_update = llm_output_parser(result['content'],ActionExtractorOutput)
        ae = dict(state_update)
        if "actions" not in ae or ae["actions"] is None:
            ae["actions"] = []
        if "status" not in ae:
            ae["status"] = "no_action"

        return {"action_extraction":ae}
    
class IU_GoalsAgent(BaseMAFIAgent):
    def __init__(
        self,
        name: str = "IU Goal",
        system_prompt: str = IU_GOAL_SYSTEM_PROMPT,
        human_templete: str = "{updated_query}",
        llm: Any = None,
        tools: List[Any] = [],
        state_manager_: Any = StateManager(),
        default_scope: str = "default",
    ):
        if llm is None:
            llm = get_shared_llm(max_tokens=300)

        super().__init__(
            name=name,
            system_prompt=system_prompt,
            human_templete=human_templete,
            llm=llm,
            tools=tools,
            state_manager_=state_manager_,
            default_scope=default_scope,
        )

    async def __call__(self, state:IUSwarmState)->IUSwarmState:
        
        updated_query = state.get("updated_query","")
        result: Dict[str,Any] = await super().arun({"updated_query",updated_query})
        content = result.get("content","")
        return {"iu_goals":str(content).strip()}
    
class IU_RiskAgent(BaseMAFIAgent):
    def __init__(
        self,
        name: str = "IU Risk",
        system_prompt: str = IU_RISK_SYSTEM_PROMPT,
        human_templete: str = "{updated_query}",
        llm: Any = None,
        tools: List[Any] = [],
        state_manager_: Any = StateManager(),
        default_scope: str = "default",
    ):
        if llm is None:
            llm = get_shared_llm(max_tokens=300)

        super().__init__(
            name=name,
            system_prompt=system_prompt,
            human_templete=human_templete,
            llm=llm,
            tools=tools,
            state_manager_=state_manager_,
            default_scope=default_scope,
        )

    async def __call__(self, state:IUSwarmState)->IUSwarmState:
        
        updated_query = state.get("updated_query","")
        result: Dict[str,Any] = await super().arun({"updated_query",updated_query})
        content = result.get("content","")
        return {"iu_risk":str(content).strip()}
    
class IU_MergeAgent:
    async def __call__(self, state: IUSwarmState)->IUSwarmState:
        parts = []
        goals = state.get("iu_goals")
        risk = state.get("iu_risk")

        if goals:
            parts.append("GOALS SECTION:\n"+goals)
        if risk:
            parts.append("RISK & CONSTRAINTS SECTION:\n" + risk)

        toon = "\n\n".join(parts).strip()
        return {"intent_understanding":toon}
    
class IUSwarm:
    def __init__(self):
        self._graph = self._build()

    def _build(self):
        builder = StateGraph(IUSwarmState)
        builder.add_node("iu_goals",IU_GoalsAgent())
        builder.add_node("iu_risk",IU_RiskAgent())
        builder.add_node("iu_merge",IU_MergeAgent())

        builder.add_edge(START,"iu_goals")
        builder.add_edge(START,"iu_risk")
        builder.add_edge("iu_goals","iu_merge")
        builder.add_edge("iu_risk","iu_merge")
        builder.add_edge("iu_merge",END)

        return builder.compile()
    
    async def ainvoke(self,state:IUSwarmState)->IUSwarmState:
        return await self._graph.ainvoke(state)
    
class IntentUnderstandingAgent:
    def __init__(self,name:str = "Intent Understanding Agent"):
        self.description = "Intent Understanding"
        self.name = name
        self._iu_swarm = IUSwarm()

    async def __call__(self, state:Orchestrator_Global_State)->IU_Output:
        updated_query = state.get("updated_query") or state.get("user_query") or ""
        if not updated_query.strip():
            return {"intent_understanding":""}
        iu_input: IUSwarmState = {"updated_query":updated_query}
        iu_state_out = await self._iu_swarm.ainvoke(iu_input)
        iu_toon = iu_state_out.get("intent_understanding","")
        return {"intent_understanding": iu_toon}