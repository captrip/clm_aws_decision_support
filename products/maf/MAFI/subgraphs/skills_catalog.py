from typing import Any, Dict, List, Optional,Tuple
import hashlib
from MAFI.subgraphs.subgraph_registry import SubgraphRegistry
from MAFI.subgraphs.registry_persistence import SubgraphRegistryPersistance

def _slug(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")

def _make_skill_id(name: str) -> str:
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]

def _shorten(text: str, max_length: int = 220) -> str:
    if not text:
        return ""
    text = str(text).strip()
    return (text[:max_length] + "…") if len(text) > max_length else text

def get_public_registry_snapshot() -> Dict[str, Dict[str, Any]]:
    raw = SubgraphRegistry.list_all()
    visible = {}
    for name, metadata in raw.items():
        if metadata.get("exposed", True):
            visible[name] = metadata
    return visible

def build_llm_catalog(
        limit: Optional[int] = None,
        include_tags: bool = True,
        max_description_length: int = 220,
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    snapshot = get_public_registry_snapshot()
    items:List[tuple[str, Dict[str, Any]]] = list(snapshot.items())
    items.sort(key=lambda x: x[0].lower())
    if limit is not None:
        items = items[:limit]
    
    catalog = []
    id_to_name = {}
    for name, metadata in items:
        skill_id = _make_skill_id(name)
        id_to_name[skill_id] = name
        catalog.append({
            "id": skill_id,
            "name": name,
            "description": _shorten(metadata.get("description", ""), max_description_length),
            "tags": metadata.get("tags", []) if include_tags else [],
            "scope": metadata.get("scope", name),
        })
    return catalog, id_to_name

def get_public_db_snapshot(db_path: str = "subgraph_registry.db") -> Dict[str, Dict[str, Any]]:
    persistence = SubgraphRegistryPersistance(db_path)
    try:
        all_subgraphs = persistence.load_all_subgraphs()
        visible = {}
        for name, metadata in all_subgraphs.items():
            if metadata.get("exposed", True):
                visible[name] = metadata
        return visible
    except Exception as e:
        print(f"Error loading from database: {e}")
        return {}
    finally:
        persistence.close()

def build_llm_catalog_from_db(
        db_path: str = "subgraph_registry.db",
        limit: Optional[int] = None,
        include_tags: bool = True,
        max_description_length: int = 220,
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    snapshot = get_public_db_snapshot(db_path)
    items:List[tuple[str, Dict[str, Any]]] = list(snapshot.items())
    items.sort(key=lambda x: x[0].lower())
    if limit is not None:
        items = items[:limit]
    
    catalog = []
    id_to_name = {}
    for name, metadata in items:
        skill_id = _make_skill_id(name)
        id_to_name[skill_id] = name
        catalog.append({
            "id": skill_id,
            "name": name,
            "description": _shorten(metadata.get("description", ""), max_description_length),
            "tags": metadata.get("tags", []) if include_tags else [],
            "scope": metadata.get("scope", name),
        })
    return catalog, id_to_name

def canonicalize_selections(
        selections: List[str],
        id_to_name: Dict[str, str],
        allow_name_fallback: bool = True,
        ) -> List[str]:
    
    if not selections:
        return []
    
    name_to_name = {name.lower(): name for name in id_to_name.values()}
    alias_to_name: Dict[str, str] = {}
    snapshot = SubgraphRegistry.list_all()
    for name, metadata in snapshot.items():
        for alias in metadata.get("aliases", []):
            alias_to_name[_slug(alias)] = name
        alias_to_name[_slug(alias)] = name

    chosen: List[str] = []
    for sel in selections:
        if sel in id_to_name:
            chosen.append(id_to_name[sel])
            continue

        if allow_name_fallback:
            key = sel.lower()
            if key in name_to_name:
                chosen.append(name_to_name[key])
                continue
            key = _slug(sel)
            if key in alias_to_name:
                chosen.append(alias_to_name[key])
                continue

    seen = set()
    out = []
    for name in chosen:
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out

def canonicalize_selction_from_db(
        selections: List[str],
        id_to_name: Dict[str, str],
        db_path: str = "subgraph_registry.db",
        allow_name_fallback: bool = True,
        ) -> List[str]:
    if not selections:
        return []
    
    name_to_name = {name.lower(): name for name in id_to_name.values()}
    alias_to_name: Dict[str, str] = {}
    persistence = SubgraphRegistryPersistance(db_path)
    try:
        all_subgraphs = persistence.load_all_subgraphs()
        for name, metadata in all_subgraphs.items():
            for alias in metadata.get("aliases", []):
                alias_to_name[_slug(alias)] = name
            alias_to_name[_slug(alias)] = name
    except Exception as e:
        print(f"Error loading from database: {e}")
    finally:
        persistence.close()

    chosen: List[str] = []
    for sel in selections:
        if sel in id_to_name:
            chosen.append(id_to_name[sel])
            continue

        if allow_name_fallback:
            key = sel.lower()
            if key in name_to_name:
                chosen.append(name_to_name[key])
                continue
            key = _slug(sel)
            if key in alias_to_name:
                chosen.append(alias_to_name[key])
                continue

    seen = set()
    out = []
    for name in chosen:
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out