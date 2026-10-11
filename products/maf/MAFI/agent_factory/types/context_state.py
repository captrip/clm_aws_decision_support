from typing import Dict,Any,Iterable,Tuple
from copy import deepcopy

class Context:

    def __init__(self,initial: Dict[str,Any]):
        self._state:Dict[str,Any] = deepcopy(initial or {})

    def get(self,key:str,default:Any=None)->Any:
        return self._state.get(key,default)

    def set(self,key:str,value:Any)->"Context":
        self._state[key] = value
        return self
    def update(self,updates:Dict[str,Any])->"Context":
        if not isinstance(updates, dict):
            raise TypeError("Updates should be a dictionary.")
        self._state.update(updates)
        return self
    
    def items(self)->Iterable[Tuple[str,Any]]:
        return self._state.items()
    def to_dict(self)->Dict[str,Any]:
        return deepcopy(self._state)
    def keys(self):
        return self._state.keys()
    def values(self):
        return self._state.values()
    def __getitem__(self,key:str)->Any:
        return self._state[key]
    def __setitem__(self,key:str,value:Any):
        self._state[key] = value
    def __contains__(self,key:str)->bool:
        return key in self._state
    def __repr__(self):        
        return f"Context({self._state})"
    
    def get_scope(self,scope:str, default:Any=None)->Dict[str,Any]:
        value = self._state.get(scope, default)
        if isinstance(value, dict):
            return dict(value)
        return value
    def set_scope(self,scope:str, value:Dict[str,Any])->"Context":
        if not isinstance(value, dict):
            raise TypeError("Scope value should be a dictionary.")
        self._state[scope] = dict(value)
        return self