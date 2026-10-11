from typing import Any, Dict, List, Optional
from MAFI.subgraphs.subgraph_registry import SubgraphRegistry
from MAFI.subgraphs.registry_persistence import SubgraphRegistryPersistance as SubgraphPersistence

class RegistrySync:
    def __init__(self, db_path: str = "subgraph_registry.db"):
        self.persistence = SubgraphPersistence(db_path)

    def save_to_db(self, name: str) -> bool:
        metadata = SubgraphRegistry.get_metadata(name)
        if not metadata:
            return False
        tool_names = metadata.get("tools", [])

        state_schema = metadata.get("state_schema")
        if state_schema is not None and not isinstance(state_schema, str):
            state_schema = f"{state_schema.__module__}.{state_schema.__name__}" if hasattr(state_schema, "__module__") and hasattr(state_schema, "__name__") else str(state_schema)
        self.persistence.save_subgraph(
            name=metadata["name"],
            description=metadata["description"],
            prompt=metadata["prompt"],
            tools=tool_names,
            state_schema=state_schema,
            scope=metadata["scope"],
            tags=metadata.get("tags", []),
            aliases=metadata.get("aliases", []),
            exposed=metadata.get("exposed", True),
        )
        return True
    
    def save_all_to_db(self):
        all_subgraphs = SubgraphRegistry.list_all()
        count = 0
        for name in all_subgraphs.keys():
            self.save_to_db(name)
            count += 1
        return count
    
    def load_from_db(self, name: str) -> Optional[Dict[str, Any]]:
        return self.persistence.load_subgraph(name)
    
    def load_all_from_db(self) -> Dict[str, Dict[str, Any]]:
        return self.persistence.load_all_subgraphs()
    def recreate_subgraph_from_db(
            self, 
            name: str,
            SubgraphFactory,
            tools_registry: Dict[str, Any],
            state_mgr = None,
            ) -> bool:
        config = self.load_from_db(name)
        if not config:
            return False
        
        tool_names = set(config.get("tools", []))
        if isinstance (tools_registry, dict):
            tools = [t for name,t in tools_registry.items() if name in tool_names]
        else:
            tools = [t for t in tools_registry if getattr(t, "name", None) in tool_names]

            SubgraphFactory(
                name=config["name"],
                description=config["description"],
                prpompt=config["prompt"],
                tools=tools,
                state_mgr=state_mgr,
                state_schema=config.get("state_schema"),
                scope=config.get("scope"),
            )
        return True
    

    def recreate_all_from_db(
            self,
            SubgraphFactory,
            tools_registry: Dict[str, Any],
            state_mgr = None,
    ) -> int:
        config = self.load_all_from_db()
        count = 0
        for name in config.keys():
            if self.recreate_subgraph_from_db(name, SubgraphFactory, tools_registry, state_mgr):
                count += 1
        return count
    

    def close(self):
        self.persistence.close()