from typing import Any, Dict, List

from MAFI.agent_factory.base_MAFI_agent import BaseMAFIAgent
from MAFI.agent_factory.types.state_manager import StateManager
from MAFI.prompts.prompts import ORCHESTRATOR_HUMAN_TEMPLETE
from MAFI.util.extraction_util import llm_output_parser
from config.config import get_shared_llm
from global_.state import Orchestrator_Global_State


class OrchestratorAgent(BaseMAFIAgent):
    def __init__(
        self,
        name: str = "Orchestrator Agent",
        system_prompt: str = "You are a helpful assistant",
        human_templete: str = ORCHESTRATOR_HUMAN_TEMPLETE,
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

    async def __call__(self, state:Orchestrator_Global_State)->Orchestrator_Global_State:
        
        orchestrator_input = {
            "user_query":state.get("user_query",""),
            "memory_info":state.get("memory_info",{})
        }
        result: Dict[str,Any] = await super().arun(orchestrator_input)
        content = result.get("content","").strip()
        thought,state_update = llm_output_parser(content,Orchestrator_Global_State)
        agent_output = Orchestrator_Global_State(**state)
        agent_output['updated_query'] = state_update['updated_query']
        agent_output['next_step'] = state_update['next_step']
        return agent_output