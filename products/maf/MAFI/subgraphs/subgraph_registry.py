from typing import Any, Dict, Optional
import threading

class SubgraphRegistry:
    _registry: Dict[str, Dict[str, Any]] = {}
    _lock = threading.RLock()

    @classmethod
    def register(cls, name: str, subgraph_info: Dict[str, Any]):
        with cls._lock:
            if name in cls._registry:
                raise ValueError(f"Subgraph with name '{name}' is already registered.")
            cls._registry[name] = subgraph_info

    @classmethod
    def get(cls, name: str) -> Optional[Dict[str, Any]]:
        with cls._lock:
            return cls._registry.get(name)

    @classmethod
    def list_all(cls) -> Dict[str, Dict[str, Any]]:
        with cls._lock:
            return dict(cls._registry)
        
    @classmethod
    def update(cls, name: str, updates: Dict[str, Any]):
        with cls._lock:
            if name not in cls._registry:
                raise ValueError(f"Subgraph with name '{name}' is not registered.")
            cls._registry[name].update(updates)

    @classmethod
    def get(cls, name: str) -> Optional[Dict[str, Any]]:
        with cls._lock:
            return dict(cls._registry.get(name)) if name in cls._registry else None
        
    @classmethod
    def remove(cls, name: str) -> None:
        with cls._lock:
            if name in cls._registry:
                del cls._registry[name]
    
    @classmethod
    def exists(cls, name: str) -> bool:
        with cls._lock:
            return name in cls._registry
        
    @classmethod
    def clear_all(cls) -> None:
        with cls._lock:
            cls._registry.clear()

    @classmethod
    def get_metadata(cls, name: str) -> Optional[Dict[str, Any]]:
        with cls._lock:
            subgraph = cls._registry.get(name)
            if subgraph is not None:
                return {
                    "name": subgraph.get("name"),
                    "description": subgraph.get("description"),
                    "prompt": subgraph.get("prompt"),
                    "tools": subgraph.get("tools"),
                    "state_schema": subgraph.get("state_schema"),
                    "scope": subgraph.get("scope"),
                }
            return None