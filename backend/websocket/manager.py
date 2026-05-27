"""
backend.websocket.manager — WebSocket connection manager.

DESIGN:
Manages all active WebSocket connections with topic-based broadcasting.

Topics:
  - "training"  — training progress, loss curves, metrics
  - "jobs"      — background job status updates
  - "system"    — GPU utilization, VRAM, queue depth
  - "global"    — general notifications

Usage:
    manager = WebSocketManager()

    # In WebSocket endpoint:
    await manager.connect(websocket, client_id, topic="training")

    # Broadcast to all clients on a topic:
    await manager.broadcast_to_topic("training", {
        "epoch": 50,
        "train_loss": 0.234,
        "val_map50": 0.821,
    })
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import WebSocket

from shared.logging.logger import get_logger

logger = get_logger(__name__)


class WebSocketManager:
    """
    Topic-based WebSocket connection manager.

    Thread-safe: all operations use asyncio locks.
    """

    def __init__(self) -> None:
        # {client_id: (websocket, set_of_topics)}
        self._connections: dict[str, tuple[WebSocket, set[str]]] = {}
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        client_id: str,
        topic: str = "global",
    ) -> None:
        """Accept and register a WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            if client_id in self._connections:
                # Update topics for existing connection
                _, topics = self._connections[client_id]
                topics.add(topic)
            else:
                self._connections[client_id] = (websocket, {topic})

        logger.info(
            "WebSocket connected",
            client_id=client_id,
            topic=topic,
            total_connections=len(self._connections),
        )

        # Send welcome message
        await self.send_personal(client_id, {
            "type": "connected",
            "client_id": client_id,
            "topic": topic,
            "timestamp": time.time(),
        })

    async def disconnect(self, client_id: str) -> None:
        """Remove a connection."""
        async with self._lock:
            self._connections.pop(client_id, None)
        logger.info("WebSocket disconnected", client_id=client_id)

    async def subscribe(self, client_id: str, topic: str) -> None:
        """Subscribe an existing connection to an additional topic."""
        async with self._lock:
            if client_id in self._connections:
                _, topics = self._connections[client_id]
                topics.add(topic)

    async def unsubscribe(self, client_id: str, topic: str) -> None:
        """Remove a topic subscription."""
        async with self._lock:
            if client_id in self._connections:
                _, topics = self._connections[client_id]
                topics.discard(topic)

    async def send_personal(
        self,
        client_id: str,
        data: dict[str, Any],
    ) -> bool:
        """
        Send message to a specific client.

        Returns True if sent successfully, False if client not found.
        """
        async with self._lock:
            conn = self._connections.get(client_id)

        if conn is None:
            return False

        websocket, _ = conn
        try:
            await websocket.send_text(json.dumps(data, default=str))
            return True
        except Exception as e:
            logger.debug("Failed to send to client", client_id=client_id, error=str(e))
            await self.disconnect(client_id)
            return False

    async def broadcast_to_topic(
        self,
        topic: str,
        data: dict[str, Any],
    ) -> int:
        """
        Broadcast message to all connections subscribed to a topic.

        Returns number of clients messaged.
        """
        # Snapshot connections to avoid lock contention during send
        async with self._lock:
            subscribers = [
                (cid, ws)
                for cid, (ws, topics) in self._connections.items()
                if topic in topics or "global" == topic
            ]

        if not subscribers:
            return 0

        message = json.dumps({**data, "_topic": topic, "_ts": time.time()}, default=str)
        disconnected: list[str] = []
        sent = 0

        for client_id, websocket in subscribers:
            try:
                await websocket.send_text(message)
                sent += 1
            except Exception:
                disconnected.append(client_id)

        # Clean up dead connections
        for cid in disconnected:
            await self.disconnect(cid)

        return sent

    async def broadcast_global(self, data: dict[str, Any]) -> int:
        """Broadcast to ALL connected clients."""
        async with self._lock:
            all_clients = [
                (cid, ws)
                for cid, (ws, _) in self._connections.items()
            ]

        if not all_clients:
            return 0

        message = json.dumps({**data, "_ts": time.time()}, default=str)
        disconnected: list[str] = []
        sent = 0

        for client_id, websocket in all_clients:
            try:
                await websocket.send_text(message)
                sent += 1
            except Exception:
                disconnected.append(client_id)

        for cid in disconnected:
            await self.disconnect(cid)

        return sent

    async def send_training_update(
        self,
        epoch: int,
        metrics: dict[str, float],
        run_id: str = "",
    ) -> None:
        """Convenience: send training epoch metrics."""
        await self.broadcast_to_topic("training", {
            "type": "training_metrics",
            "run_id": run_id,
            "epoch": epoch,
            "metrics": metrics,
        })

    async def send_job_update(
        self,
        job_uuid: str,
        status: str,
        progress: float,
        message: str = "",
    ) -> None:
        """Convenience: send job status update."""
        await self.broadcast_to_topic("jobs", {
            "type": "job_update",
            "job_uuid": job_uuid,
            "status": status,
            "progress": progress,
            "progress_pct": progress * 100,
            "message": message,
        })

    async def send_gpu_stats(self, stats: dict[str, Any]) -> None:
        """Convenience: broadcast GPU stats."""
        await self.broadcast_to_topic("system", {
            "type": "gpu_stats",
            **stats,
        })

    async def heartbeat(self, client_id: str) -> bool:
        """Send heartbeat ping to keep connection alive."""
        return await self.send_personal(client_id, {
            "type": "heartbeat",
            "timestamp": time.time(),
        })

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    @property
    def client_ids(self) -> list[str]:
        return list(self._connections.keys())


# Global manager instance (shared across all routes)
ws_manager = WebSocketManager()
