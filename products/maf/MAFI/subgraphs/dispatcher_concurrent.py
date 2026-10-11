from typing import Dict,Any,List,Optional
from MAFI.subgraphs.subgraph_registry import SubgraphRegistry
from MAFI.subgraphs.registry_sync import RegistrySync
from MAFI.agent_factory.types.state_manager import StateManager
import asyncio
import time

async def _execute_single_skill(
        skill_name: str,
        entry: Dict[str,Any],
        state: Dict[str,Any],
        skill_index:int = 0,
        total_skills: int = 1
)-> Dict[str,Any]:
    start_time = time.time()
    runner = entry.get("runner")
    graph = entry.get("graph")
    state_manager: Optional[StateManager] = entry.get("state_mgr")
    scope: str = entry.get("scope",skill_name)

    enriched = dict(state)
    if state_manager:
        enriched["__context"] = state_manager.context().to_dict()

    try:
        if runner is not None:
            result = await runner(enriched)
        elif graph is not None:
            result = await graph.ainvoke(enriched)
        else:
            result = {
                "error":f"Skill '{skill_name}' has no runner or graph"
            }
        execution_time = (time.time() - start_time)*1000

        if state_manager:
            state_manager.update(scope,{"execution_result":result})
        
        return {
            "skill": skill_name,
            "result": result,
            "executed": True,
            "duration_ms": round(execution_time,2),
            "timestamp": time.time()
        }
    except Exception as e:
        execution_time = (time.time() - start_time)*1000
        return {
            "skill": skill_name,
            "result": str(e),
            "executed": False,
            "duration_ms": round(execution_time,2),
            "timestamp": time.time()
        }
async def dispatch_selected_skill(
        state: Dict[str,Any],
        auto_load_from_db: bool = True,
        db_path: str = "subgraph_registry.db",
        SubgrapgFactory = None,
        tools_registry = None,
        state_manager: Optional[StateManager] = None)-> Dict[str,Any]:
    selected = state.get("selected_skills",[])
    if not selected:
        return{"executions":[],"executed_skills":[]}
    
    registry_snapshot = SubgraphRegistry.list_all()

    if auto_load_from_db:
        alredy_registered = [s for s in selected if s in registry_snapshot]
        missing_skills = [s for s in selected if s not in registry_snapshot]
        if alredy_registered:
            #TODO: add a logger here to debug in case of fail
            pass
        if missing_skills:
            if SubgrapgFactory is None or tools_registry in None:
                #TODO: add a logger here to debug in case of fail
                pass
            else:
                sync = RegistrySync(db_path=db_path)
                try:
                    for skill_name in missing_skills:
                        success = sync.recreate_subgraph_from_db(
                            name=skill_name,
                            SubgraphFactory=SubgrapgFactory,
                            tools_registry=tools_registry,
                            state_mgr=state_manager
                        )
                        if success:
                            print(f"loaded {skill_name}")
                        else:
                            print(f" Not Found in DB {skill_name}")
                finally:
                    sync.close()
                registry_snapshot = SubgraphRegistry.list_all()

    tasks = []
    total_skills = len(selected)
    for idx, skill_name in enumerate(selected):
        entry = registry_snapshot.get(skill_name)
        if not entry:
            async def create_error_result(name):
                return{
                    "skill":name,
                    "error":f"skill '{name}' not registered",
                    "executed": False,
                    "duration_ms": 0,
                    "timestamp": time.time()
                }
            tasks.append(create_error_result(skill_name))
        else:
            tasks.append(_execute_single_skill(
                skill_name=skill_name,
                entry=entry,
                state=state,
                skill_index=idx,
                total_skills=total_skills))
    results = await asyncio.gather(*tasks,return_exceptions=True)

    executions: List[Dict[str,Any]] = []
    executed_names: List[str] = []
    tool_summaries: List[str] = []

    for result in results:
        if isinstance(result, Exception):
            executions.append(
                {
                "skill": "unknown",
                "error": str(result),
                "executed": False
                }
            )
            continue
        executions.append(result)
        if result.get("executed",False):
            executed_names.append(result["skill"])

        result_payload = result.get("result",{}) or {}
        sub_summaries = result_payload.get("tool_results_summary") or []
        if isinstance(sub_summaries, str):
            tool_summaries.append(sub_summaries)
        elif isinstance(sub_summaries,list):
            tool_summaries.extend(str(s) for s in sub_summaries)

        successful_count = len([e for e in executions if e.get("executed",False)])
        failed_count = len(executions) - successful_count
        total_duration = sum(e.get("duration_ms",0) for e in executions)

        return {
            "executions": executions,
            "executed_skills": executed_names,
            "tool_results_summary": tool_summaries
        }