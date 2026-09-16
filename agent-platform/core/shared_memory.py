"""
Shared Memory Bus
=================
Cross-agent shared memory system for v3 multi-agent support.

Provides named channels where any agent can read/write data.
Subscribers receive real-time notifications when channels are updated.
Backed by SQLite for persistence and in-memory cache for speed.

This is distinct from per-agent memory (core/memory.py) — shared memory
is explicitly shared across agents and used for inter-agent data passing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

logger = logging.getLogger("agentforge.shared_memory")


@dataclass
class SharedMemoryEntry:
    """A single entry in a shared memory channel."""
    id: str
    channel: str
    key: str
    value: Any
    written_by: str  # agent_id that wrote this
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "channel": self.channel,
            "key": self.key,
            "value": self.value,
            "written_by": self.written_by,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class ChannelInfo:
    """Metadata about a shared memory channel."""
    name: str
    created_at: float
    entry_count: int
    writers: List[str]  # agent_ids that have written to this channel
    last_updated: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "entry_count": self.entry_count,
            "writers": self.writers,
            "last_updated": self.last_updated,
        }


class SharedMemoryBus:
    """
    Shared memory system for inter-agent data exchange.

    Agents write to named channels. Other agents can read from those
    channels or subscribe for real-time updates.

    Storage: SQLite for durability + in-memory cache for hot reads.
    """

    def __init__(self, db_path: str = "./data/shared_memory.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        # In-memory cache: channel -> {key -> SharedMemoryEntry}
        self._cache: Dict[str, Dict[str, SharedMemoryEntry]] = {}

        # Subscribers: channel -> list of async queues
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._global_subscribers: List[asyncio.Queue] = []

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shared_memory (
                id TEXT PRIMARY KEY,
                channel TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                written_by TEXT NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sm_channel
            ON shared_memory(channel, key)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sm_writer
            ON shared_memory(written_by)
        """)
        conn.commit()
        conn.close()

    # ── Write Operations ─────────────────────────────────────────

    async def write(
        self,
        channel: str,
        key: str,
        value: Any,
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SharedMemoryEntry:
        """Write a value to a shared memory channel."""
        entry = SharedMemoryEntry(
            id=uuid4().hex[:16],
            channel=channel,
            key=key,
            value=value,
            written_by=agent_id,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        # Persist to SQLite
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT OR REPLACE INTO shared_memory
               (id, channel, key, value, written_by, timestamp, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.id,
                entry.channel,
                entry.key,
                json.dumps(entry.value),
                entry.written_by,
                entry.timestamp,
                json.dumps(entry.metadata),
            ),
        )
        conn.commit()
        conn.close()

        # Update cache
        if channel not in self._cache:
            self._cache[channel] = {}
        self._cache[channel][key] = entry

        # Notify subscribers
        await self._notify(channel, entry)

        logger.debug(
            "Shared memory write: channel=%s key=%s by=%s",
            channel, key, agent_id,
        )
        return entry

    # ── Read Operations ──────────────────────────────────────────

    def read(self, channel: str, key: str) -> Optional[SharedMemoryEntry]:
        """Read a specific key from a channel."""
        # Check cache first
        if channel in self._cache and key in self._cache[channel]:
            return self._cache[channel][key]

        # Fall back to DB
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM shared_memory WHERE channel = ? AND key = ? ORDER BY timestamp DESC LIMIT 1",
            (channel, key),
        ).fetchone()
        conn.close()

        if row:
            entry = SharedMemoryEntry(
                id=row["id"],
                channel=row["channel"],
                key=row["key"],
                value=json.loads(row["value"]),
                written_by=row["written_by"],
                timestamp=row["timestamp"],
                metadata=json.loads(row["metadata"]),
            )
            # Populate cache
            if channel not in self._cache:
                self._cache[channel] = {}
            self._cache[channel][key] = entry
            return entry
        return None

    def read_channel(self, channel: str, limit: int = 100) -> List[SharedMemoryEntry]:
        """Read all entries from a channel."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM shared_memory WHERE channel = ? ORDER BY timestamp DESC LIMIT ?",
            (channel, limit),
        ).fetchall()
        conn.close()

        entries = []
        for row in rows:
            entries.append(SharedMemoryEntry(
                id=row["id"],
                channel=row["channel"],
                key=row["key"],
                value=json.loads(row["value"]),
                written_by=row["written_by"],
                timestamp=row["timestamp"],
                metadata=json.loads(row["metadata"]),
            ))
        return entries

    def read_by_agent(self, agent_id: str, limit: int = 100) -> List[SharedMemoryEntry]:
        """Read all entries written by a specific agent."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM shared_memory WHERE written_by = ? ORDER BY timestamp DESC LIMIT ?",
            (agent_id, limit),
        ).fetchall()
        conn.close()

        return [
            SharedMemoryEntry(
                id=row["id"],
                channel=row["channel"],
                key=row["key"],
                value=json.loads(row["value"]),
                written_by=row["written_by"],
                timestamp=row["timestamp"],
                metadata=json.loads(row["metadata"]),
            )
            for row in rows
        ]

    # ── Channel Management ───────────────────────────────────────

    def list_channels(self) -> List[ChannelInfo]:
        """List all channels with metadata."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT
                channel,
                MIN(timestamp) as created_at,
                COUNT(*) as entry_count,
                GROUP_CONCAT(DISTINCT written_by) as writers,
                MAX(timestamp) as last_updated
            FROM shared_memory
            GROUP BY channel
            ORDER BY last_updated DESC
        """).fetchall()
        conn.close()

        return [
            ChannelInfo(
                name=row["channel"],
                created_at=row["created_at"],
                entry_count=row["entry_count"],
                writers=row["writers"].split(",") if row["writers"] else [],
                last_updated=row["last_updated"],
            )
            for row in rows
        ]

    def clear_channel(self, channel: str):
        """Clear all entries in a channel."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM shared_memory WHERE channel = ?", (channel,))
        conn.commit()
        conn.close()
        self._cache.pop(channel, None)

    def clear_all(self):
        """Clear all shared memory."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM shared_memory")
        conn.commit()
        conn.close()
        self._cache.clear()

    # ── Subscriptions ────────────────────────────────────────────

    def subscribe(self, channel: Optional[str] = None) -> asyncio.Queue:
        """
        Subscribe to shared memory updates.
        If channel is None, subscribes to ALL channel updates.
        """
        q: asyncio.Queue = asyncio.Queue()
        if channel:
            self._subscribers.setdefault(channel, []).append(q)
        else:
            self._global_subscribers.append(q)
        return q

    def unsubscribe(self, queue: asyncio.Queue, channel: Optional[str] = None):
        """Unsubscribe from shared memory updates."""
        if channel and channel in self._subscribers:
            self._subscribers[channel] = [
                q for q in self._subscribers[channel] if q is not q
            ]
        else:
            self._global_subscribers = [
                q for q in self._global_subscribers if q is not queue
            ]

    async def _notify(self, channel: str, entry: SharedMemoryEntry):
        """Notify all subscribers of a channel update."""
        event_data = entry.to_dict()

        # Channel-specific
        for q in self._subscribers.get(channel, []):
            await q.put(event_data)

        # Global
        for q in self._global_subscribers:
            await q.put(event_data)

    # ── Stats ────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get overall shared memory statistics."""
        conn = sqlite3.connect(self.db_path)
        total = conn.execute("SELECT COUNT(*) FROM shared_memory").fetchone()[0]
        channels = conn.execute("SELECT COUNT(DISTINCT channel) FROM shared_memory").fetchone()[0]
        writers = conn.execute("SELECT COUNT(DISTINCT written_by) FROM shared_memory").fetchone()[0]
        conn.close()

        return {
            "total_entries": total,
            "channel_count": channels,
            "unique_writers": writers,
            "cached_channels": len(self._cache),
            "active_subscribers": sum(
                len(subs) for subs in self._subscribers.values()
            ) + len(self._global_subscribers),
        }
