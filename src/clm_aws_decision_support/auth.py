from __future__ import annotations
import os
from dataclasses import dataclass
from functools import lru_cache
from fastapi import Header, HTTPException

class SupabaseconfigError(RuntimeError):
    """SUPABASE not configured"""

@dataclass(frozen=True, slots=True)
class SupabaseUser:
    id: str
    email: str | None

@lru_cache(maxsize=1)
def _admin_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise SupabaseconfigError(
            "set Supabase credentials"
        )
    from supabase import create_client

    return create_client(url, key)

def _parse_bearer(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer auth header")
    return authorization.split(" ",1)[1].strip()

def required_supabase_user(authorization: str | None = Header(default=None)) -> SupabaseUser:
    token = _parse_bearer(authorization=authorization)
    try:
        client = _admin_client()
        response = client.auth.get_user(token)
    except SupabaseconfigError:
        raise
    except Exception as error:
        raise HTTPException(status_code=401, detail=f"invalid session : {error}") from error

    user = getattr(response, "user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid_session")
    return SupabaseUser(id=user.id, email=user.email)