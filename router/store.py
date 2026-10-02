"""SQLite persistence for decisions, rewards, and conversation spend.

The context vector is stored with each decision so a reward arriving later --
from the async judge, seconds or minutes after the response -- can update the
bandit with the vector that arm actually saw at selection time (DESIGN.md
section 6). Without this, delayed rewards silently corrupt learning.
"""

import os
import sqlite3
import threading

import numpy as np

from router import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT    NOT NULL,
    turn_index      INTEGER NOT NULL,
    arm             TEXT    NOT NULL,
    context_vector  BLOB    NOT NULL,
    retry_depth     INTEGER NOT NULL,
    task_type       TEXT    NOT NULL,
    cost            REAL    NOT NULL,
    reward          INTEGER,
    needs_judge     INTEGER NOT NULL DEFAULT 0,
    query           TEXT,
    response_text   TEXT,
    -- Which projection built context_vector. Without it the documented reset
    -- (bump PROJECTION_VERSION, delete the snapshot) leaves this table holding
    -- vectors in the old feature space, and the next judge batch feeds them to
    -- the fresh bandit. The snapshot is versioned; these rows must be too.
    projection_version TEXT NOT NULL,
    created_at      TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_pending_judge
    ON decisions (needs_judge, reward, projection_version);
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    spent           REAL NOT NULL DEFAULT 0.0
);
"""


class Store:
    def __init__(self, db_path=config.DB_PATH):
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        # check_same_thread=False because Starlette runs every sync handler on
        # its anyio threadpool, so one Store is shared across ~40 threads. That
        # is only safe with the mutex below: sqlite3.Connection is not safe for
        # concurrent use, and unguarded it drops rows and undercounts spend
        # silently. A plain Lock suffices -- Store calls nothing that re-enters
        # it, so there is no re-entrancy or lock-ordering hazard.
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # --- decisions

    def log_decision(
        self,
        conversation_id,
        turn_index,
        arm,
        context_vector,
        retry_depth,
        task_type,
        cost,
        reward,
        needs_judge,
        query,
        response_text,
    ):
        vector = np.asarray(context_vector, dtype=np.float64).reshape(-1)
        if vector.shape[0] != config.CONTEXT_DIM:
            raise ValueError(
                f"context_vector has {vector.shape[0]} dims, expected "
                f"{config.CONTEXT_DIM}"
            )
        with self._lock:
            cursor = self.conn.execute(
                """
                INSERT INTO decisions (
                    conversation_id, turn_index, arm, context_vector,
                    retry_depth, task_type, cost, reward, needs_judge, query,
                    response_text, projection_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    int(turn_index),
                    arm,
                    vector.tobytes(),
                    int(retry_depth),
                    task_type,
                    float(cost),
                    reward,
                    1 if needs_judge else 0,
                    query,
                    response_text,
                    config.PROJECTION_VERSION,
                ),
            )
            self.conn.commit()
            return int(cursor.lastrowid)

    def _row_to_dict(self, row):
        if row is None:
            return None
        data = dict(row)
        data["context_vector"] = np.frombuffer(
            data["context_vector"], dtype=np.float64
        )
        return data

    def get_decision(self, decision_id):
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM decisions WHERE id = ?", (int(decision_id),)
            ).fetchone()
            return self._row_to_dict(row)

    def attach_reward(self, decision_id, reward):
        with self._lock:
            cursor = self.conn.execute(
                "UPDATE decisions SET reward = ?, needs_judge = 0 WHERE id = ?",
                (int(reward), int(decision_id)),
            )
            self.conn.commit()
            if cursor.rowcount == 0:
                raise KeyError(f"no decision with id {decision_id}")

    def pending_judge(self, limit):
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT * FROM decisions
                WHERE needs_judge = 1 AND reward IS NULL
                  AND projection_version = ?
                ORDER BY id ASC LIMIT ?
                """,
                (config.PROJECTION_VERSION, int(limit)),
            ).fetchall()
            return [self._row_to_dict(row) for row in rows]

    # --- conversation spend

    def get_spent(self, conversation_id):
        with self._lock:
            row = self.conn.execute(
                "SELECT spent FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            return float(row["spent"]) if row else 0.0

    def add_spend(self, conversation_id, cost):
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO conversations (conversation_id, spent) VALUES (?, ?)
                ON CONFLICT (conversation_id)
                DO UPDATE SET spent = spent + excluded.spent
                """,
                (conversation_id, float(cost)),
            )
            self.conn.commit()

    def close(self):
        with self._lock:
            self.conn.close()
