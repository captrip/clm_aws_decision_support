from typing import Any, Dict, List, Optional
import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from MAFI.agent_factory.base_MAFI_agent import BaseMAFIAgent
from MAFI.agent_factory.types.state_manager import StateManager
from MAFI.subgraphs.skills_catalog import (
    build_llm_catalog,
    build_llm_catalog_from_db,
    canonicalize_selction_from_db,
    canonicalize_selections
)
from config.config import get_shared_llm
from MAFI.prompts.prompts import PLANNER_HUMAN_TEMPLATE,PLANNER_SYSTEM_PROMPT
def _extract_json(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in the text.")
    return json.loads(m.group())

class MAFIActionPlannerAgent(BaseMAFIAgent):
    def __init__(
        self,
        name: str = "MAFI Action Planner",
        system_prompt: str = PLANNER_SYSTEM_PROMPT,
        human_templete: str = PLANNER_HUMAN_TEMPLATE,
        llm: Any = None,
        tools: List[Any] = [],
        state_manager_: Any = None,
        default_scope: str = "default",
        db_path: str = "subgraph_registry.db",
    ):
        if llm is None:
            llm = get_shared_llm()
        self.db_path = db_path
        
        super().__init__(
            name=name,
            system_prompt=system_prompt,
            human_templete=human_templete,
            llm=llm,
            tools=tools,
            state_manager_=state_manager_,
            default_scope=default_scope,
        )

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        intent_understanding = state.get("intent_understanding", {})
        catalog,id_to_name = build_llm_catalog_from_db(
            db_path=self.db_path,
            limit=100,
            include_tags=True,
            max_description_length=220,
        )
        planner_input = {
            "intent_understanding": json.dumps(intent_understanding, ensure_ascii=False),
            "skills_catalog": json.dumps(catalog, ensure_ascii=False),
        }

        result = await super().arun(planner_input)

        raw = result['content'].strip()
        data = _extract_json(raw)

        plan: List[str] = data.get("plan", [])
        selected_skill_ids: List[str] = data.get("selected_skills", [])
        analysis: List[str] = data.get("analysis", [])

        if not isinstance(plan, list) or not isinstance(selected_skill_ids, list):
            raise ValueError("Invalid output format: 'plan' should be a list of strings and 'selected_skills' should be a list of skill ids.")
        
        selected_names = canonicalize_selction_from_db(
            selected_skill_ids,id_to_name=id_to_name,db_path=self.db_path
        )
        out = {
            "plan": plan,
            "selected_skill_ids": selected_skill_ids,
            "selected_names": selected_names,
            "analysis": analysis,
            "skills_catalog": catalog,
        }
        if self.state_manager:
            self.state_manager.set_scope(scope = self.scope,data=out)

        return out
