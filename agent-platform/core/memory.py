"""
Memory Module
=============
Two-level memory system:
  - Short-term: conversation history + current task state (SQLite)
  - Long-term: past results and task logs (Chroma vector store)

Each agent has its own isolated memory namespace.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agentforge.memory")


# ═══════════════════════════════════════════════════════════════════════
# Short-Term Memory (SQLite)
# ═══════════════════════════════════════════════════════════════════════

class ShortTermMemory:
    """
    Stores conversation history and current task state in SQLite.
    Each agent_id gets its own table namespace.
    """

    def __init__(self, db_path: str = "./data/memory.db", max_history: int = 50):
        self.db_path = db_path
        self.max_history = max_history
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS task_state (
                agent_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                state TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (agent_id, task_id)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_agent
            ON conversations(agent_id, created_at)
        """)
        conn.commit()
        conn.close()

    def add_message(self, agent_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the conversation history."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO conversations (agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
            (agent_id, role, content, json.dumps(metadata or {}), time.time()),
        )
        conn.commit()

        # Trim old messages beyond max_history
        count = conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE agent_id = ?", (agent_id,)
        ).fetchone()[0]
        if count > self.max_history:
            conn.execute(
                """DELETE FROM conversations WHERE id IN (
                    SELECT id FROM conversations WHERE agent_id = ?
                    ORDER BY created_at ASC LIMIT ?
                )""",
                (agent_id, count - self.max_history),
            )
            conn.commit()
        conn.close()

    def get_history(self, agent_id: str, limit: int = 50) -> List[Dict[str, str]]:
        """Retrieve conversation history for an agent."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT role, content FROM conversations WHERE agent_id = ? ORDER BY created_at ASC LIMIT ?",
            (agent_id, limit),
        ).fetchall()
        conn.close()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def save_task_state(self, agent_id: str, task_id: str, state: Dict[str, Any]):
        """Save the current task state."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT OR REPLACE INTO task_state (agent_id, task_id, state, updated_at)
               VALUES (?, ?, ?, ?)""",
            (agent_id, task_id, json.dumps(state), time.time()),
        )
        conn.commit()
        conn.close()

    def get_task_state(self, agent_id: str, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the current task state."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT state FROM task_state WHERE agent_id = ? AND task_id = ?",
            (agent_id, task_id),
        ).fetchone()
        conn.close()
        return json.loads(row[0]) if row else None

    def clear(self, agent_id: str):
        """Clear all memory for an agent."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM conversations WHERE agent_id = ?", (agent_id,))
        conn.execute("DELETE FROM task_state WHERE agent_id = ?", (agent_id,))
        conn.commit()
        conn.close()

    def get_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get memory stats for an agent."""
        conn = sqlite3.connect(self.db_path)
        msg_count = conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE agent_id = ?", (agent_id,)
        ).fetchone()[0]
        task_count = conn.execute(
            "SELECT COUNT(*) FROM task_state WHERE agent_id = ?", (agent_id,)
        ).fetchone()[0]
        conn.close()
        return {"messages": msg_count, "active_tasks": task_count, "backend": "sqlite"}


# ═══════════════════════════════════════════════════════════════════════
# Long-Term Memory (Chroma Vector Store)
# ═══════════════════════════════════════════════════════════════════════

class LongTermMemory:
    """
    Vector store for past results, task logs, and knowledge.
    Uses ChromaDB for embedding + retrieval.
    Each agent gets its own collection.
    """

    def __init__(self, persist_dir: str = "./data/chroma_store", collection_prefix: str = "agent"):
        self.persist_dir = persist_dir
        self.collection_prefix = collection_prefix
        self._client = None
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

    def _get_client(self):
        if self._client is None:
            try:
                import chromadb
                self._client = chromadb.PersistentClient(path=self.persist_dir)
            except ImportError:
                logger.warning("chromadb not installed — long-term memory disabled")
                return None
        return self._client

    def _collection_name(self, agent_id: str) -> str:
        # ChromaDB collection names must be 3-63 chars, alphanumeric + underscores
        name = f"{self.collection_prefix}_{agent_id}".replace("-", "_")
        return name[:63]

    def _get_collection(self, agent_id: str):
        client = self._get_client()
        if client is None:
            return None
        return client.get_or_create_collection(name=self._collection_name(agent_id))

    def store(self, agent_id: str, text: str, metadata: Optional[Dict[str, Any]] = None, doc_id: Optional[str] = None):
        """Store a document in long-term memory."""
        collection = self._get_collection(agent_id)
        if collection is None:
            return
        doc_id = doc_id or f"doc_{int(time.time() * 1000)}"
        collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id],
        )
        logger.debug("Stored document %s for agent %s", doc_id, agent_id)

    def search(self, agent_id: str, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search long-term memory by semantic similarity."""
        collection = self._get_collection(agent_id)
        if collection is None:
            return []
        try:
            results = collection.query(query_texts=[query], n_results=n_results)
            docs = []
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results.get("distances") else 0
                docs.append({"text": doc, "metadata": meta, "distance": dist})
            return docs
        except Exception as exc:
            logger.warning("Long-term memory search failed: %s", exc)
            return []

    def clear(self, agent_id: str):
        """Clear all long-term memory for an agent."""
        client = self._get_client()
        if client is None:
            return
        try:
            client.delete_collection(name=self._collection_name(agent_id))
        except Exception:
            pass

    def get_stats(self, agent_id: str) -> Dict[str, Any]:
        collection = self._get_collection(agent_id)
        count = collection.count() if collection else 0
        return {"documents": count, "backend": "chroma"}


# ═══════════════════════════════════════════════════════════════════════
# Unified Memory Manager
# ═══════════════════════════════════════════════════════════════════════

class MemoryManager:
    """
    Unified interface for both short-term and long-term memory.
    Provides per-agent memory isolation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        st_config = config.get("short_term", {})
        lt_config = config.get("long_term", {})

        self.short_term = ShortTermMemory(
            db_path=st_config.get("db_path", "./data/memory.db"),
            max_history=st_config.get("max_history", 50),
        )
        self.long_term = LongTermMemory(
            persist_dir=lt_config.get("persist_dir", "./data/chroma_store"),
            collection_prefix=lt_config.get("collection_prefix", "agent"),
        )

    def get_conversation_history(self, agent_id: str, limit: int = 50) -> List[Dict[str, str]]:
        return self.short_term.get_history(agent_id, limit)

    def add_message(self, agent_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        self.short_term.add_message(agent_id, role, content, metadata)

    def store_long_term(self, agent_id: str, text: str, metadata: Optional[Dict] = None):
        self.long_term.store(agent_id, text, metadata)

    def search_long_term(self, agent_id: str, query: str, n: int = 5) -> List[Dict]:
        return self.long_term.search(agent_id, query, n)

    def clear_all(self, agent_id: str):
        self.short_term.clear(agent_id)
        self.long_term.clear(agent_id)

    def get_stats(self, agent_id: str) -> Dict[str, Any]:
        return {
            "short_term": self.short_term.get_stats(agent_id),
            "long_term": self.long_term.get_stats(agent_id),
        }
