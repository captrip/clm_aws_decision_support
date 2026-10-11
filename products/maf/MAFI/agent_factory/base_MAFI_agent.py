from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from MAFI.agent_factory.types.state_manager import StateManager

@dataclass
class BaseMAFIAgent:
    name: str
    system_prompt: str
    human_templete:str
    llm: Any
    tools: List[Any]
    state_manager_: Any
    default_scope: str = "default"
    read_context_from_state_manager: bool = True


    def __post_init__(self):
        self.state_manager = StateManager()
        self.state_manager.merge_global(self.state_manager_.get_global())

    def _resolve_scope(self, context: Dict[str, Any]) -> str:
        scope = context.get("scope")
        if isinstance(scope, str):
            return scope.strip()
        
        return self.default_scope
    
    def _compose_system_text(self, state: Dict[str, Any], scope_name: str) -> str:
        base_context = (state.get("context") or "").strip()

        sections = [
            "[ROLE]",
            self.system_prompt.strip(),
            "\n[CONTEXT]",
            base_context,
        ]
        if self.read_context_from_state_manager:
            try:
                scope_ctx = self.state_manager.get_scope(self._resolve_scope(state)) or {}
            except Exception as e:
                scope_ctx = {}

            try:
                global_ctx = self.state_manager.get_global() or {}
            except Exception as e:
                global_ctx = {}

            sections.extend([
                "\n[SCOPE]",
                f"- name: {scope_name}",
                f"- keys: {list(scope_ctx.keys())}",
                "\n[GLOBAL]",
                f"- keys: {list(global_ctx.keys())}",
            ])

        sections.extend([
            "\n[OUTPUT STYLE]",
            "- Be concise and actionable.",
        ])
        return "\n".join(sections)
    
    def _build_messages(self, state: Dict[str, Any],scope_name: str) -> List[BaseMessage]:
        
        system_text = self._compose_system_text(state, scope_name)
        human_text = self.human_templete.format(**state)

        return [
            SystemMessage(content=system_text),
            HumanMessage(content=human_text),
        ]
    
    def _persist_last_output(self, scope_name: str, content: str)-> None:
        
        try:
            self.state_manager.update(scope_name, {"last_output": content})
        except Exception as e:
            pass

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        scope_name = self._resolve_scope(state)
        messages = self._build_messages(state, scope_name)
        response = self.llm.invoke(messages)
        content = getattr(response, "content", str(response))
        self._persist_last_output(scope_name, content)
        return {"agent": self.name, "content": content, "raw":response}
    
    async def arun(self, state: Dict[str, Any]) -> Dict[str, Any]:
        scope_name = self._resolve_scope(state)
        messages = self._build_messages(state, scope_name)
        response = await self.llm.ainvoke(messages)
        content = getattr(response, "content", str(response))
        self._persist_last_output(scope_name, content)
        return {"agent": self.name, "content": content, "raw":response}
    
    