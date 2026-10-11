from typing import Optional,Dict,Any
from langgraph.graph import StateGraph,START,END
from langgraph.types import interrupt,Command
from MAFI.agent_factory.types.state_manager import StateManager
from MAFI.agents.action_planner import MAFIActionPlannerAgent
from MAFI.subgraphs.dispatcher_concurrent import dispatch_selected_skill
from MAFI.subgraphs.skills_catalog import build_llm_catalog_from_db
from global_.state import Orchestrator_Global_State
from functools import partial

class PlannerGraph:
    def __init__(
            self,
            state_manager: Optional[StateManager] = None,
            auto_load_skills: bool = True,
            db_path: str = "subgraph_registry.db",
            SubgraphFactory = None,
            tools_registry = None
        ):
        self.state_manager = state_manager
        self.auto_load_skills = auto_load_skills
        self.db_path = db_path
        self.SubgraphFactory = SubgraphFactory
        self.tools_registry = tools_registry
        self.planner = MAFIActionPlannerAgent(state_manager_=state_manager,default_scope="planner")

    
    def hitl_approval(self,state: Dict[str,Any]) ->Command:
        catalog, _id_to_name = build_llm_catalog_from_db(
            db_path=self.db_path,
            limit=None,
            include_tags=True,
            max_description_length=220
        )

        interrupt_payload = {
            "session_id": state.get("session_id"),
            "user_id": state.get("user_id"),
            "plan": state.get("plan"),
            "analysis": state.get("analysis"),
            "selected_skill_ids": state.get("selected_skill_ids",[]),
            "selected_skills": state.get("selected_skills",[]),
            "catalog": catalog,
            "message": "Review and approve the proposed Plan"
        }

        response = interrupt(interrupt_payload)
        decision = response.get("decision", "procede").strip().lower()
        selected_skill_ids = response.get("selected_skill_ids", []) or []
        selected_skills = response.get("selected_skills", []) or []
        edited_plan = response.get("plan", None)

        updates = {}

        id_to_name = {
            item.get("id"):item.get("name") for item in catalog
            if item.get("id") and item.get("name")
        }
        name_to_id = {
            item.get("name","").lower():item.get("id") for item in catalog
            if item.get("id") and item.get("name")
        }

        if selected_skill_ids:
            updates['selected_skill_ids'] = selected_skill_ids
            updates['selected_skills'] = [id_to_name.get(sid,sid) for sid in selected_skill_ids]
        elif selected_skills:
            updates['selected_skills'] = selected_skills
            updates['selected_skill_ids'] = [name_to_id.get(name.lower(),name) for name in selected_skills]
        
        if edited_plan is not None:
            updates['plan'] = edited_plan
        
        if decision == "redo":
            return Command(goto='planner', update=updates)
        elif decision == "end":
            return Command(goto=END, update=updates)
        else:
            return Command(goto='dispatch', update=updates)
        
    def build(self):
        builder = StateGraph(Orchestrator_Global_State)
        builder.add_node("planner",self.planner)
        builder.add_node("hitl_approval",self.hitl_approval)
        dispatcher_node = partial(
            dispatch_selected_skill,
            auto_load_from_db = self.auto_load_skills,
            db_path = self.db_path,
            SubgraphFactory = self.SubgraphFactory,
            tools_registry = self.tools_registry,
            state_manager = self.state_manager
        )
        builder.add_node("dispatch",dispatcher_node)
        builder.add_edge(START,"planner")
        builder.add_edge("planner","hitl_approval")
        builder.add_edge("dispatch",END)

        return builder.compile()

