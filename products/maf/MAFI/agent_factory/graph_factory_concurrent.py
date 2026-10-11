from typing import Any, Dict, List, Optional,Type
import json
from langgraph.graph import StateGraph,START, END
from langgraph.prebuilt import ToolNode
from config.config import get_shared_llm
from langchain_core.messages import HumanMessage, SystemMessage
from MAFI.subgraphs.subgraph_registry import SubgraphRegistry
from MAFI.agent_factory.types.state_manager import StateManager

#TODO: logger

def _normalize_schema(schema: Optional[Type]) -> Type:
    if schema is None:
        return dict
    if isinstance(schema, dict):
        raise TypeError("Schema should be a type, not an instance. For example, use 'dict' instead of '{}'.")
    return schema

class SubgraphBrain:
    def __init__(
            self,
            name: str,
            prompt: str,
            tools: List[Any],
            state_mgr : Optional[StateManager],
            scope: Optional[str] = None,
    ):
        self.name = name
        self.prompt = prompt
        self.tools = tools or []
        self.state_mgr = state_mgr
        self.scope = scope or name
        self.llm = get_shared_llm()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_summariser = get_shared_llm(max_tokens=300)

        async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                ctx = state.get("__context",{}) or {}

                plan_lines: List[str] = []
                if isinstance(state.get("plan"),list):
                    plan_lines = [str(item) for item in state["plan"]]

                planner_ctx = ctx.get("planner") if isinstance(ctx, dict) else None
                if not plan_lines and isinstance(planner_ctx, dict) and isinstance(planner_ctx.get("plan"), list):
                    plan_lines = [str(item) for item in planner_ctx["plan"]]

                analysis_lines: List[str] = []
                if isinstance(state.get("analysis"),list):
                    analysis_lines = [str(item) for item in state["analysis"]]
                elif isinstance(planner_ctx, dict) and isinstance(planner_ctx.get("analysis"), list):
                    analysis_lines = [str(item) for item in planner_ctx["analysis"]]

                plan_text = "\n".join(plan_lines).strip()
                analysis_text = "\n".join(analysis_lines).strip()

                user_query_parts:List[str] = []
                if plan_text:
                    user_query_parts.append(f"Planner plan steps:\n{plan_text}")
                if analysis_text:
                    user_query_parts.append(f"Planner analysis:\n{analysis_text}")

                user_query = "\n\n".join(user_query_parts) if user_query_parts else "No specific query, just follow the prompt instructions."
                msgs = [
                    SystemMessage(content=self.prompt),
                    SystemMessage(
                        content=f"shared context snapshot (may include 'planner' and other scopes): {ctx}"
                        ),
                    HumanMessage(content=user_query)
                ]

                response = await self.llm_with_tools.ainvoke(msgs)
                result = {
                    "messages":[response],
                    "llm_response": response,
                }

                if self.state_mgr:
                    self.state_mgr.update(
                        self.scope,
                        {
                            "llm":{
                                "raw_response": response,
                                "user_query": user_query,
                                "used_context": ctx,
                            }
                        },)
                    
                return result
            except Exception as e:
                error_msg = f"Error in SubgraphBrain '{self.name}': {str(e)}"
                if self.state_mgr:
                    self.state_mgr.update(
                        self.scope,
                        {
                            "error": error_msg
                        },)
                    
                return {"error": error_msg}
            
        async def summarize_tools_node(self,state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                tool_payload: Dict[str, Any] = {}
                for key in ["tool_outputs", "messages","results","data"]:
                    if key in state and state[key]:
                        tool_payload[key] = state[key]
                if not tool_payload:
                    return state
                tool_results_text = json.dumps(tool_payload, default=str)[:3500]
                if not tool_results_text:
                    return state
                
                msgs = [
                    SystemMessage(content=f"Summarise the following tool outputs and results in a concise way, focusing on key information that would be useful for an agent to know. If there are any errors or important details, include those in the summary. Tool outputs and results: {tool_results_text}"),
                ]

                resp = await self.tool_summariser.ainvoke(msgs)
                summary_text = str(getattr(resp,"content",resp)).strip()
                if not summary_text:
                    summary_text = tool_results_text[:500]
                new_state = dict(state)
                existing = new_state.get("tool_results_summary") or []
                if isinstance(existing, list):
                    existing.append(summary_text)
                    new_state["tool_results_summary"] = existing
                else:
                    new_state["tool_results_summary"] = [existing, summary_text]
                return new_state
            except Exception as e:
                error_msg = f"Error summarizing tools in SubgraphBrain '{self.name}': {str(e)}"
                return {"error": error_msg}
            


class SubgraphFactory:
    def __init__(
            self,
            name: str,
            description: str,
            prpompt: str,
            tools: Optional[List[Any]] = None,
            state_mgr : Optional[StateManager] = None,
            state_schema: Optional[Type] = None,
            scope: Optional[str] = None,
            ):
        self.name = name
        self.description = description
        self.prompt = prpompt
        self.tools = tools or []
        self.state_mgr = state_mgr
        self.state_schema = _normalize_schema(state_schema)
        self.scope = scope or name
        self.graph = None
        self.agent = None

        self.graph = self._create_subgraph()

        SubgraphRegistry.register(
            self.name,
            {
                "name": self.name,
                "description": self.description,
                "prompt": self.prompt,
                "tools": self.tools,
                "state_schema": self.state_schema,
                "scope": self.scope,
                "graph": self.graph,
                "agent": self.agent,
                "state_mgr": self.state_mgr,
            },
            
            )

    def _create_subgraph(self) :
        self.agent = SubgraphBrain(
            name=self.name,
            prompt=self.prompt,
            tools=self.tools,
            state_mgr=self.state_mgr,
            scope=self.scope,
        )
        graph = StateGraph(name=self.state_schema)
        graph.add_node("subagent_reason", self.agent)
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_node("tool_summariser", self.agent.summarize_tools_node)
        graph.add_edge(START, "subagent_reason")
        graph.add_edge("subagent_reason", "tools")
        graph.add_edge("tools", "tool_summariser")
        graph.add_edge("tool_summariser", END)

        compiled = graph.compile()
        return compiled
    
    async def ainvoke(self, state: Dict[str, Any]):
        ctx = self.state_mgr.context().to_dict() if self.state_mgr else {}

        enriched_state = dict(state)
        enriched_state["__context"] = ctx
        return await self.graph.ainvoke(enriched_state)
         