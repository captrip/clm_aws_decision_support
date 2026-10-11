import sqlite3
import json
from typing import Any, Dict, List,Optional


class SubgraphRegistryPersistance:
    def __init__(self, db_path: str = "subgraph_registry.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path,check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS subgraphs (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    prompt TEXT,
                    scope TEXT,
                    state_schema TEXT,
                    exposed INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS subgraph_tools (
                    subgraph_name TEXT,
                    tool_name TEXT,
                    tool_order INTEGER,
                    PRIMARY KEY (subgraph_name, tool_name),
                    FOREIGN KEY (subgraph_name) REFERENCES subgraphs(name) ON DELETE CASCADE
                )
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS subgraph_tags (
                    subgraph_name TEXT,
                    tag_name TEXT,
                    PRIMARY KEY (subgraph_name, tag_name),
                    FOREIGN KEY (subgraph_name) REFERENCES subgraphs(name) ON DELETE CASCADE
                )
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS subgraph_aliases (
                    subgraph_name TEXT,
                    alias_name TEXT,
                    PRIMARY KEY (subgraph_name, alias_name),
                    FOREIGN KEY (subgraph_name) REFERENCES subgraphs(name) ON DELETE CASCADE
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_approval_requests(
                    request_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    description TEXT,
                    prompt TEXT,
                    tool_names TEXT,
                    state_schema TEXT,
                    tags TEXT,
                    status TEXT DEFAULT 'pending',
                    aliasses TEXT,
                    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    reviewed_at TIMESTAMP,
                    submitted_by TEXT,
                    reviewed_by TEXT,
                    feedback TEXT
                )
            """)

    def save_subgraph(
            self,
            name: str,
            description: str,
            prompt: str,
            tools: List[str],
            state_schema: Optional[str] = None,
            scope: Optional[str] = None,
            tags: Optional[List[str]] = None,
            aliases: Optional[List[str]] = None,
            exposed: bool = True,
            ):
        scope = scope or name
        tags = tags or []
        aliases = aliases or []
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO subgraphs (name, description, prompt, scope, state_schema, exposed, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (name, description, prompt, scope, state_schema, int(exposed)))

            self.conn.execute("DELETE FROM subgraph_tools WHERE subgraph_name = ?", (name,))
            for order, tool in enumerate(tools):
                self.conn.execute("""
                    INSERT INTO subgraph_tools (subgraph_name, tool_name, tool_order)
                    VALUES (?, ?, ?)
                """, (name, tool, order))

            self.conn.execute("DELETE FROM subgraph_tags WHERE subgraph_name = ?", (name,))
            for tag in tags:
                self.conn.execute("""
                    INSERT INTO subgraph_tags (subgraph_name, tag_name)
                    VALUES (?, ?)
                """, (name, tag))

            self.conn.execute("DELETE FROM subgraph_aliases WHERE subgraph_name = ?", (name,))
            for alias in aliases:
                self.conn.execute("""
                    INSERT INTO subgraph_aliases (subgraph_name, alias_name)
                    VALUES (?, ?)
                """, (name, alias))

            for idx,tool_name in enumerate(tools):
                self.conn.execute("""
                    UPDATE subgraph_tools SET tool_order = ? WHERE subgraph_name = ? AND tool_name = ?
                """, (name, tool_name,idx))

            for tag in tags:
                self.conn.execute("""
                    UPDATE subgraph_tags SET tag_name = ? WHERE subgraph_name = ? AND tag_name = ?
                """, (name, tag))

            for alias in aliases:
                self.conn.execute("""
                    UPDATE subgraph_aliases SET alias_name = ? WHERE subgraph_name = ? AND alias_name = ?
                """, (name, alias))


    def load_subgraph(self, name: str) -> Optional[Dict[str, Any]]:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT data FROM subgraph_registry WHERE name = ?
            """, (name,))
            result = cursor.fetchone()
            if not result:
                return None
            cursor.execute("""
                SELECT tool_name FROM subgraph_tools WHERE subgraph_name = ? ORDER BY tool_order
            """, (name,))
            tools = [row[0] for row in cursor.fetchall()]
            cursor.execute("""
                SELECT tag_name FROM subgraph_tags WHERE subgraph_name = ?
            """, (name,))
            tags = [row[0] for row in cursor.fetchall()]
            cursor.execute("""
                SELECT alias_name FROM subgraph_aliases WHERE subgraph_name = ?
            """, (name,))
            aliases = [row[0] for row in cursor.fetchall()]

            return {
                "name": name,
                "description": result[0],
                "prompt": result[1],
                "scope": result[2],
                "state_schema": result[3],
                "exposed": bool(result[4]),
                "created_at": result[5],
                "updated_at": result[6],
                "tools": tools,
                "tags": tags,
                "aliases": aliases,
            }

    def load_all_subgraphs(self) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM subgraph_registry")
        names = [row[0] for row in cursor.fetchall()]
        return [self.load_subgraph(name) for name in names]
    
    def delete_subgraph(self, name: str):
        with self.conn:
            cur = self.conn.execute("DELETE FROM subgraph_registry WHERE name = ?", (name,))
            return cur.rowcount > 0

    def subgraph_exists(self, name: str) -> bool:
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM subgraph_registry WHERE name = ?", (name,))
        return cursor.fetchone() is not None

    def list_subgraph_names(self) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM subgraph_registry")
        return [row[0] for row in cursor.fetchall()]

    def close(self):
        if self.conn:
            self.conn.close()


    def submit_agent_for_approval(
            self,
            agent_name: str,
            description: str,
            prompt: str,
            tool_names: List[str],
            submitted_by: str,
            state_schema: Optional[str] = None,
            tags: Optional[List[str]] = None,
            aliasses: Optional[List[str]] = None,
            exposed: bool = True,
            scope: Optional[str] = None,
    ) ->str:
        scope = scope or agent_name
        tags = tags or []
        aliasses = aliasses or []

        with self.conn:
            self.conn.execute("""
                INSERT INTO agent_approval_requests (agent_name, description, prompt, tool_names, scope, state_schema, tags, aliasses, exposed, submitted_by, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """, (
                agent_name,
                description,
                prompt,
                json.dumps(tool_names),
                scope,
                state_schema,
                json.dumps(tags),
                json.dumps(aliasses),
                int(exposed),
                submitted_by
            )) 