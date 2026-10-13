"""SQLite persistence for the MAF lifecycle (requests, catalogues, members, audit log)."""

from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import threading
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Iterator

from .models import (
    CatalogueAgent,
    CatalogueWorkflow,
    RequestDoc,
    Role,
    name_key,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS maf_requests (
    id         TEXT PRIMARY KEY,
    doc        TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS maf_workflows (
    key TEXT PRIMARY KEY,
    doc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS maf_agents (
    key TEXT PRIMARY KEY,
    doc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS maf_members (
    email TEXT PRIMARY KEY,
    roles TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS maf_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id  TEXT,
    at          TEXT NOT NULL,
    actor       TEXT NOT NULL,
    role        TEXT,
    action      TEXT NOT NULL,
    detail      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_maf_events_request ON maf_events (request_id, id);
"""


class DomainError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MafStore:
    def __init__(self, db_path: str) -> None:
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.executescript(_SCHEMA)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        # BEGIN IMMEDIATE serialises writers across worker processes as well as threads.
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            else:
                self.conn.execute("COMMIT")

    # -- audit log -------------------------------------------------------------------

    def log_event(
        self,
        request_id: str | None,
        actor: str,
        role: str | None,
        action: str,
        detail: str = "",
    ) -> None:
        self.conn.execute(
            "INSERT INTO maf_events (request_id, at, actor, role, action, detail) VALUES (?,?,?,?,?,?)",
            (request_id, now(), actor, role, action, detail),
        )

    def version(self) -> int:
        with self._lock:
            row = self.conn.execute("SELECT COALESCE(MAX(id), 0) AS v FROM maf_events").fetchone()
        return int(row["v"])

    def list_events(self, request_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT at, actor, role, action, detail FROM maf_events WHERE request_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (request_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    # -- requests --------------------------------------------------------------------

    def create_request(self, doc: RequestDoc, actor: str, detail: str) -> None:
        with self.transaction():
            self.conn.execute(
                "INSERT INTO maf_requests (id, doc, created_at, updated_at) VALUES (?,?,?,?)",
                (doc.id, doc.model_dump_json(), doc.created_at, doc.updated_at),
            )
            self.log_event(doc.id, actor, Role.CHATBOT_BUILDER.value, "request.created", detail)

    def get_request(self, request_id: str) -> RequestDoc | None:
        with self._lock:
            row = self.conn.execute("SELECT doc FROM maf_requests WHERE id = ?", (request_id,)).fetchone()
        return RequestDoc.model_validate_json(row["doc"]) if row else None

    def list_requests(self) -> list[RequestDoc]:
        with self._lock:
            rows = self.conn.execute("SELECT doc FROM maf_requests ORDER BY created_at DESC, id DESC").fetchall()
        return [RequestDoc.model_validate_json(row["doc"]) for row in rows]

    def mutate(
        self,
        request_id: str,
        change: Callable[["MafStore", RequestDoc], tuple[str, str | None, str, str]],
    ) -> RequestDoc:
        """Apply ``change`` atomically. It returns (action, role, actor, detail) for the audit log."""
        with self.transaction():
            row = self.conn.execute("SELECT doc FROM maf_requests WHERE id = ?", (request_id,)).fetchone()
            if row is None:
                raise DomainError(404, "Request not found.")
            doc = RequestDoc.model_validate_json(row["doc"])
            action, role, actor, detail = change(self, doc)
            doc.updated_at = now()
            self.conn.execute(
                "UPDATE maf_requests SET doc = ?, updated_at = ? WHERE id = ?",
                (doc.model_dump_json(), doc.updated_at, request_id),
            )
            self.log_event(request_id, actor, role, action, detail)
        return doc

    # -- catalogues ------------------------------------------------------------------

    def get_workflow(self, name: str) -> CatalogueWorkflow | None:
        with self._lock:
            row = self.conn.execute("SELECT doc FROM maf_workflows WHERE key = ?", (name_key(name),)).fetchone()
        return CatalogueWorkflow.model_validate_json(row["doc"]) if row else None

    def get_agent(self, name: str) -> CatalogueAgent | None:
        with self._lock:
            row = self.conn.execute("SELECT doc FROM maf_agents WHERE key = ?", (name_key(name),)).fetchone()
        return CatalogueAgent.model_validate_json(row["doc"]) if row else None

    def list_workflows(self) -> list[CatalogueWorkflow]:
        with self._lock:
            rows = self.conn.execute("SELECT doc FROM maf_workflows ORDER BY key").fetchall()
        return [CatalogueWorkflow.model_validate_json(row["doc"]) for row in rows]

    def list_agents(self) -> list[CatalogueAgent]:
        with self._lock:
            rows = self.conn.execute("SELECT doc FROM maf_agents ORDER BY key").fetchall()
        return [CatalogueAgent.model_validate_json(row["doc"]) for row in rows]

    def publish_workflow(self, entry: CatalogueWorkflow) -> CatalogueWorkflow:
        """Call inside a transaction. Republishing a name bumps its version."""
        existing = self.get_workflow(entry.name)
        entry.version = existing.version + 1 if existing else 1
        self.conn.execute(
            "INSERT INTO maf_workflows (key, doc) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET doc = excluded.doc",
            (name_key(entry.name), entry.model_dump_json()),
        )
        return entry

    def publish_agent(self, entry: CatalogueAgent) -> CatalogueAgent:
        existing = self.get_agent(entry.name)
        entry.version = existing.version + 1 if existing else 1
        self.conn.execute(
            "INSERT INTO maf_agents (key, doc) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET doc = excluded.doc",
            (name_key(entry.name), entry.model_dump_json()),
        )
        return entry

    # -- members ---------------------------------------------------------------------

    def get_roles(self, email: str) -> set[Role]:
        with self._lock:
            row = self.conn.execute("SELECT roles FROM maf_members WHERE email = ?", (email.lower(),)).fetchone()
        return {Role(value) for value in json.loads(row["roles"])} if row else set()

    def list_members(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute("SELECT email, roles FROM maf_members ORDER BY email").fetchall()
        return [{"email": row["email"], "roles": json.loads(row["roles"])} for row in rows]

    def has_admin(self) -> bool:
        return any(Role.ADMIN.value in member["roles"] for member in self.list_members())

    def set_roles(self, email: str, roles: list[Role], actor: str) -> None:
        email = email.lower()
        unique = sorted({role.value for role in roles})
        with self.transaction():
            if Role.ADMIN.value not in unique:
                others = [m for m in self.list_members() if m["email"] != email and Role.ADMIN.value in m["roles"]]
                current = self.conn.execute("SELECT roles FROM maf_members WHERE email = ?", (email,)).fetchone()
                was_admin = bool(current and Role.ADMIN.value in json.loads(current["roles"]))
                if was_admin and not others:
                    raise DomainError(409, "At least one admin is required.")
            if unique:
                self.conn.execute(
                    "INSERT INTO maf_members (email, roles) VALUES (?,?) "
                    "ON CONFLICT(email) DO UPDATE SET roles = excluded.roles",
                    (email, json.dumps(unique)),
                )
            else:
                self.conn.execute("DELETE FROM maf_members WHERE email = ?", (email,))
            self.log_event(None, actor, Role.ADMIN.value, "member.updated", f"{email}: {', '.join(unique) or 'no roles'}")

    def bootstrap_admin(self, email: str) -> bool:
        """Grant admin to the first user when no admin exists. Returns True if granted."""
        email = email.lower()
        with self.transaction():
            if self.has_admin():
                return False
            roles = sorted(self.get_roles(email) | {Role.ADMIN}, key=lambda role: role.value)
            self.conn.execute(
                "INSERT INTO maf_members (email, roles) VALUES (?,?) "
                "ON CONFLICT(email) DO UPDATE SET roles = excluded.roles",
                (email, json.dumps([role.value for role in roles])),
            )
            self.log_event(None, email, Role.ADMIN.value, "member.bootstrap", "first user became admin")
        return True


@lru_cache(maxsize=1)
def get_maf_store() -> MafStore:
    data_dir = pathlib.Path(os.environ.get("CLM_DATA_DIR", "data")).resolve()
    return MafStore(str(data_dir / "maf.db"))
