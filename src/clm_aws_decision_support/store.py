from __future__ import annotations
import hashlib
import os
import secrets
import sqlite3
import threading
import uuid
from typing import Any
import numpy as np


_SCHEMA = """
CREATE TABLE IF NOT EXISTS deployments (
    logical_name            TEXT    PRIMARY_KEY,
    provier                 TEXT    NOT NULL,
    model_id                TEXT,
    deployment              TEXT,
    input_cost_per_million  REAL    NOT NULL,
    output_cost_per_million REAL    NOT NULL,
    sample_accuracy         REAL,
    active                  INTEGER NOT NULL DEFAULT 1,
    created_at              TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    question    TEXT NOT NULL,
    model       TEXT NOT NULL,
    policy      TEXT NOT NULL,
    features    BLOB NOT NULL,
    cost        REAL NOT NULL,
    reward      REAL,
    source      TEXT NOT NULL DEFAULT 'ask',
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS api_keys (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    label       TEXT,
    key_hash    TEXT NOT NULL UNIQUE,
    key_prefix  TEXT NOT NULL,
    revoked     INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

class Store:
    def __init__(
            self,
            db_path: str
    ) -> None:
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory,exist_ok=True)

        self._lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()


    def add_deployment(
            self,
            logical_name: str,
            provider: str,
            *,
            model_id: str | None,
            deployment: str | None,
            input_cost_per_million: float,
            output_cost_per_million: float,
            sample_accuracy: float | None = None
    ) -> None:
        with self._lock:
            existing = self.conn.execute(
                "SELECT active FROM deployments WHERE logical_name = ?", (logical_name,)
            ).fetchone()
            if existing is not None and existing["active"]:
                raise ValueError(f"Deployment {logical_name!r} already exists")
            self.conn.execute(
                """
                INSERT INTO deployments (
                    logical_name, provider, model_id, deployment,
                    input_cost_per_million, output_cost_per_million,
                    sample_accuracy, active
                ) VALUES (?,?,?,?,?,?,?,1)
                ON CONFLICT(logical_name) DO UPDATE SET
                    provider = excluded.provider,
                    model_id = excluded.model_id,
                    deployment = excluded.deployment,
                    input_cost_per_million = excluded.input_cost_per_million,
                    output_cost_per_million = excluded.output_cost_per_million,
                    sample_accuracy = excluded.sample_accuracy,
                    active = 1
                """,
                (
                    logical_name,
                    provider,
                    model_id,
                    deployment,
                    float(input_cost_per_million),
                    float(output_cost_per_million),
                    sample_accuracy
                ),
            )
            self.conn.commit()

    def list_deployments(self, active_only: bool = True) -> list[dict[str, Any]]:
        with self._lock:
            query = "SELECT * FROM deployments"
            if active_only:
                query += "WHERE active = 1"
            query += " ORDER BY created_at"
            rows = self.conn.execute(query).fetchall()
            return [dict(row) for row in rows]

    def deactivate_deployment(self, logical_name: str) -> bool:
        with self._lock:
            cursor = self.conn.execute(
                "UPDATE deployments SET active = 0 WHERE logical_name = ? AND active = 1",
                (logical_name,),
            )
            self.conn.commit()
            return cursor.rowcount>0

    def log_decision(
            self,
            decision_id: str,
            question: str,
            model: str,
            policy: str,
            features: np.ndarray,
            cost: float,
            reward: float | None = None,
            source: str = "ask"
    ) -> None:
        vector = np.asarray(features, dtype=np.float64).reshape(-1)
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO decisions (decision_id, question, model, policy, features, cost, reward, source)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (decision_id, question, model, policy, vector.tobytes(). float(cost), reward, source),
            )
            self.conn.commit()


    def _row_to_decision(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        data = dict(row)
        data["features"] = np.frombuffer(data["features"], dtype=np.float64)
        return data

    def get_decision(self, decision_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM decision WHERE decision_id = ?", (decision_id,)
            ).fetchone()
            return self._row_to_decision(row)

    def attach_reward(self, decision_id: str, reward: float) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM decisions WHERE decision_id = ? AND reward IS NULL", (decision_id,)
            ).fetchone()
            if row is None:
                return None
            self.conn.execute(
                "UPDATE decisions SET reward = ? WHERE decision_id = ?", (float(reward), decision_id)
            )
            self.conn.commit()
            return self._row_to_decision(row)

    def create_api_key(self, user_id: str, label:str | None = None) -> tuple[dict[str, Any],str]:
        raw_key = f"sk-{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        key_id = str(uuid.uuid4())
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO api_keys (id, usr_id, label, key_hash, key_prefix)
                VALUES (?,?,?,?,?)
                """,
                (key_id, user_id, label, key_hash, raw_key[:11]),
            )
            self.conn.commit()
            row = self.conn.execute("SELECT * FROM api_keys WHERE id =?", (key_id,)).fetchone()
        return self._row_to_api_key(row), raw_key

    def _row_to_api_key(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        del data["key_hash"]
        data["revoked"] = bool(data["revoked"])
        return data

    def list_api_keys(self, user_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM api_keys WHERE user_id = ? ORDER BY created_at", (user_id)
            ).fetchall()
            return [self._row_to_api_key(row) for row in rows]

    def revoke_api_key(self, user_id: str, key_id: str) -> bool:
        with self._lock:
            cursor = self.conn.execute(
                "UPDATE api_keys SET revoked = 1 WHERE id = ? AND user_id = ? AND revoked = 0",
                (key_id, user_id)
            )
            self.conn.commit()
            return cursor.rowcount > 0

    def verify_api_key(self, raw_key: str) -> dict[str, Any] | None:

        key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM api_keys WHERE key_hash = ? AND revoked = 0", (key_hash,)
            ).fetchone()
            return self._row_to_api_key(row) if row is not None else None