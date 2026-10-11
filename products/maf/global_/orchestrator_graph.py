import json
from typing import Optional
from langgraph.graph import StateGraph, END, START
import re
from MAFI.prompts.prompts import SUMMARISER_HUMAN_TEMPLET, SUMMARIZER_PROMPT
from MAFI.types.understanding_states import SummarizerOutput
from MAFI.util.extraction_util import llm_output_parser
from global_.state import Orchestrator_Global_State
from config.config import get_shared_llm
from global_.planner.planner_graph import PlannerGraph
from MAFI.agent_factory.base_MAFI_agent import BaseMAFIAgent
from MAFI.agent_factory.types.state_manager import StateManager
from MAFI.agents.orchestrator_agent import OrchestratorAgent
from MAFI.agents.IU_agents import IntentUnderstandingAgent,ActionExtractorAgent
from global_.types import PlannerGraphState, PlannerNodeInput, PlannerNodeOutput, SummaryNodeOutput

class Global_Orchestrator:

    def __init__(
            self,
            auto_load_skills: bool = True,
            db_path: str = "subgraph_registry.db",
            SubgraphFactory = None,
            tools_registry = None
    ):
        self.description = "global orchestrator agent"
        self.orchestrator_agent = OrchestratorAgent()
        self.action_extractor = ActionExtractorAgent()
        self.intent_understanding = IntentUnderstandingAgent()
        self.state_manager = StateManager()
        self.auto_load_skills = auto_load_skills
        self.db_path = db_path
        self.SungraphFactory = SubgraphFactory
        self.tools_registry = tools_registry
    
    async def plannerNode(
            self,
            state: PlannerNodeInput,
            config: Optional[dict] = None,
            ) -> PlannerNodeOutput:
        if not hasattr(self,'_planner_graph'):
            self._planner_graph = PlannerGraph(
                state_manager=self.state_manager,
                auto_load_skills=self.auto_load_skills,
                db_path=self.db_path,
                SubgraphFactory=self.SungraphFactory,
                tools_registry=self.tools_registry
            ).build()
        
        planner_input = {
            "intent_understanding": state['action_extraction']['actions'],
            "session_id": state.get('session_id'),
            "user_id": state.get("user_id")
        }

        result = await self._planner_graph.ainvoke(planner_input,config=config)
        self.state_manager.set_scope(scope="planner",data=result)
        return {"planner": PlannerGraphState(**result)}
    
    async def summary_node(
            self,
            state: Orchestrator_Global_State
    ) -> SummaryNodeOutput:
        def extract_tool_outputs(planner: dict) -> str:
            summary_text = planner.get("tool_result_summary")
            if summary_text:
                return summary_text
            executions = planner.get("executions",[])
            tool_summaries = []
            for exe in executions:
                skill = exe.get("skill","unknown_skill")
                messages = exe.get("result" , {}).get("messages",[])
                for msg in messages:
                    if hasattr(msg,"content"):
                        content = msg.content
                        tool_name = getattr(msg,"name","")
                    elif isinstance(msg,dict):
                        content = msg.get("content","")
                        tool_name = msg.get("name","")
                    else:
                        content = str(msg)
                        tool_name = ""
                    summary = f"[{skill} - {tool_name}]: {content}"
                    tool_summaries.append(summary)
                return "\n".join(tool_summaries)
        
        intent_understanding = (
            state.get("intent_understanding")
            or state.get("action_extraction")
            or ""
        )
        if isinstance(intent_understanding, dict) and "summary" in intent_understanding:
            intent_understanding = intent_understanding.get("summary")
        
        #TODO: knowledge Base 

        planner = state.get("planner",{}) or {}

        tool_results_summaries = []
        for exe in planner.get("executions" , []):
            summary = exe.get("result",{}).get("tool_results_summary")
            if summary:
                tool_results_summaries.extend(
                    summary if isinstance(summary,list) else [summary]
                )
        planner_summary = "\n".join(tool_results_summaries) or planner.get("tool_results_summary","")
        tool_output = extract_tool_outputs(planner=planner) if planner else ""
        planner_in = f"Summary : \n {planner_summary} \n\n Tool Outputs : \n{tool_output}".strip()

        self.state_manager.set_scope(scope="planner",data=planner)

        summary_agent = BaseMAFIAgent(
            name="summaryAgent",
            llm=get_shared_llm(max_tokens=500),
            tools=[],
            system_prompt=SUMMARIZER_PROMPT,
            human_templete=SUMMARISER_HUMAN_TEMPLET,
            state_manager_=self.state_manager,
        )

        summariser_input = {
            "user" : state.get("user_query","") or state.get("user_id",""),
            "intent_understanding" : intent_understanding,
            "planner" : planner_in,
            "memory_info" : "Nothing"
        }
        result = await summary_agent.arun(summariser_input)
        content = result.get("content") if isinstance(result,dict) else None
        final_answer_text = None
        if isinstance(content,str):
            try:
                _,parsed = llm_output_parser(content,SummarizerOutput)
                if isinstance(parsed,dict):
                    final_answer_text = parsed.get("final_answer")
            except Exception as e:
                final_answer_text = None
        
        if not final_answer_text:
            final_answer_text = content if isinstance(content.str) else str(result)
        
        confidence_match = re.search(
            r"\bConfidence\s*:s*(High|Medium|Low)\b",final_answer_text,re.IGNORECASE
        )
        confidence_value = confidence_match.group(1).capitalize() if confidence_match else "Low"

        formatted = (
            f"**Final Answer**\n{final_answer_text}\n\n"
            f"**Confidence**\n{confidence_value}\n\n"
        )
        summary_payload = {
            "agent" : result.get("agent","summaryAgent") if isinstance(result,dict) else "summaryAgent",
            "answer" : formatted,
            "content" : formatted,
            "raw" : result,
            "final_answer" : final_answer_text,
            "confidance": confidence_value
        }

        self.state_manager.set_scope("summary",data=summary_payload)
        return {
            "summary":summary_payload,
            "final_answer":[final_answer_text]
        }
    
    #TODO: KB Node

    def build(
            self,
            enable_checkpointer : bool = False
    ) -> StateGraph:
        builder = StateGraph(Orchestrator_Global_State)
        builder.add_node("orchestrator", self.orchestrator_agent)
        builder.add_node("action_extractor",self.action_extractor)
        builder.add_node("intent_understanding",self.intent_understanding)
        builder.add_node("action",self.plannerNode)
        builder.add_node("summary",self.summary_node)

        builder.add_edge(START,"orchestrator")
        def _next_from_orchestrator(state: Orchestrator_Global_State):
            ns = state.get("next_step")
            if ns=="user_clarification":
                return "user_clarification"
            if ns=="intent_understanding":
                return ["action_extractor","intent_understanding"]
            if ns=="summary":
                return "summary"
        builder.add_conditional_edges("orchestrator",_next_from_orchestrator)
        builder.add_edge("action_extractor","action")
        builder.add_edge("action","summary")
        builder.add_edge("summary",END)

        if enable_checkpointer:
            pass
        return builder.compile()
        

                
