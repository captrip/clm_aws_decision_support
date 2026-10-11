from typing import Any, Dict, List, Optional
import threading
from MAFI.agent_factory.types.context_state import Context

class StateManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._global_state : Dict[str, Any] = {}
        self._scopes : Dict[str, Dict[str, Any]] = {}


    def context(self)-> Context:
        with self._lock:
            return Context(self._global_state)
    
    def update(self,scope:str, data: Dict[str,Any]):
        if not isinstance(data, dict):
            raise TypeError("Data should be a dictionary.")
        with self._lock:
            if scope not in self._scopes:
                self._scopes[scope] = {}
            self._scopes[scope].update(data)

            self._global_state[scope] = self._scopes[scope]

    def get_scope(self,scope:str)->Dict[str,Any]:
        with self._lock:
            return dict(self._scopes.get(scope, {}))
    
    def set_scope(self,scope:str, data: Dict[str,Any]):
        if not isinstance(data, dict):
            raise TypeError("Data should be a dictionary.")
        with self._lock:
            self._scopes[scope] = dict(data)
            self._global_state[scope] = self._scopes[scope]

    def clear_scope(self,scope:str):
        with self._lock:
            if scope in self._scopes:
                del self._scopes[scope]
            if scope in self._global_state:
                del self._global_state[scope]

    def list_scopes(self)->List[str]:
        with self._lock:
            return list(self._scopes.keys())
        
    def get_global(self)->Dict[str,Any]:
        with self._lock:
            return dict(self._global_state)
    def merge_global(self, data: Dict[str,Any]):
        if not isinstance(data, dict):
            raise TypeError("Data should be a dictionary.")
        with self._lock:
            for k,v in data.items():
                if k in self._scopes and isinstance(v,dict):
                    raise TypeError(f"Cannot merge key '{k}' into global state because it conflicts with an existing scope.")
            self._global_state.update(data)


    def __repr__(self):
        with self._lock:
            return f"StateManager(global_state={self._global_state}, scopes={self._scopes})"