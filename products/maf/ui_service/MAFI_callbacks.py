from __future__ import annotations
import json
import time
from typing import Any, Dict, Optional
from langchain_core.callbacks.base import AsyncCallbackHandler

class MAFICallback(AsyncCallbackHandler):
    run_inline = True

    def __init__(self,run_id:str, state_emitter):
        self.run_id = run_id
        self.state_emitter = state_emitter
        self.llm_start_times: Dict[str,float] = {}
        self.llm_count = 0

    async def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id = None, tags = None, metadata = None, **kwargs):
        print(f"The kwargs are as follows {kwargs}")
        llm_run_id = str(kwargs.get("run_id", f"{self.run_id}_{self.llm_count}"))
        agent_name = None
        if "tags" in kwargs and kwargs["tags"]:
            agent_name = next((tag for tag in kwargs["tags"] if not tag.startswith("seq:") and not tag.startswith("langchain")), None)

        if not agent_name:
            agent_name = kwargs.get("name") or serialized.get("name")

        self.llm_start_times[llm_run_id] = {
            "start_time": time.time(),
            "agent_name": agent_name
        }
        self.llm_count +=1

    async def on_llm_end(self, response, *, run_id, parent_run_id = None, tags = None, **kwargs):
        llm_run_id = str(kwargs.get("run_id",""))
        end_time = time.time()
        start_info = self.llm_start_times.get(llm_run_id)
        if isinstance(start_info, dict):
            start_time = start_info.get("start_time")
            agent_name = start_info.get("agent_name")
        else:
            start_time = start_info
            agent_name = None
        
        latency_ms = (end_time - start_time) * 1000 if start_time else None
        if llm_run_id in self.llm_start_times:
            del self.llm_start_times[llm_run_id]
        
        token_usege = {}
        if hasattr(response,"llm_output") and response.llm_output:
            token_info = response.llm_output.get("token_usege",{})
            token_usege = {
                "prompts_token": token_info.get("prompt_tokens",0),
                "completion_token": token_info.get("completion_token",0),
                "total_tokens": token_info.get("total_tokens",0)
            }
        
        self.state_emitter.on_llm_metrics(
            run_id = self.run_id,
            llm_run_id = llm_run_id,
            latency_ms = latency_ms,
            token_usege = token_usege,
            agent_name = agent_name
        )

    async def on_llm_error(self, error, *, run_id, parent_run_id = None, tags = None, **kwargs):
        llm_run_id = str(kwargs.get("run_id",""))
        if llm_run_id in self.llm_start_times:
            del self.llm_start_times[llm_run_id]

    async def on_chain_end(self, outputs:Any, **kwargs:Any):
        state = self._normalize_state(outputs)
        source_name = kwargs.get("name")
        if not source_name and kwargs.get("tags"):
            for tag in kwargs.get("tags"):
                if isinstance(tag,str) and not tag.startswith("seq:") and not tag.startswith("langchain"):
                    source_name = tag
                    break
        state["_source"] = source_name or "unknown"
        if "run_id" in kwargs and kwargs.get("run_id") is not None:
            state["_lc_run_id"] = str(kwargs.get("run_id"))

        self.state_emitter.on_state_update(self.run_id,state)

    @staticmethod
    def _normalize_state(outputs: Any) -> Dict[str,Any]:
        if isinstance(outputs,dict):
            return outputs
        
        if isinstance(outputs,str):
            s = outputs.strip()
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed,dict):
                        return parsed
                    return {"output": parsed}
                except json.JSONDecodeError:
                    pass
            return {"output": outputs}
        return {"output": outputs, "_type":type(outputs).__name__}