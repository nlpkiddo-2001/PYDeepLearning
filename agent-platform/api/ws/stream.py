"""
WebSocket Stream Endpoint
=========================
Streams real-time execution logs for a specific agent.
The UI connects here to get live plan/tool_call/tool_result/done events.

v3: Added multi-agent stream endpoints for inter-agent events,
shared memory updates, and agent hierarchy changes.

WS /agents/{agent_id}/stream      — single agent events
WS /stream/all                    — all agent events
WS /multi-agent/stream            — NEW: inter-agent communication events
WS /shared-memory/stream          — NEW: shared memory update events
WS /shared-memory/{channel}/stream — NEW: per-channel updates
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.streaming import event_bus, StreamEvent

logger = logging.getLogger("agentforge.ws")

router = APIRouter(tags=["websocket"])

# References injected at startup
_orchestrator = None


def set_ws_orchestrator(orchestrator):
    global _orchestrator
    _orchestrator = orchestrator


@router.websocket("/agents/{agent_id}/stream")
async def agent_stream(websocket: WebSocket, agent_id: str):
    """
    WebSocket endpoint that streams execution events for a specific agent.

    On connect, any buffered (missed) events are replayed first so the
    client never has a gap in its event log.  A `last_event_id` query
    parameter can be supplied to replay only events after that ID.

    Events are JSON objects with fields:
    - type: plan | tool_call | tool_result | memory_read | done | error | retry | info
            | agent_spawn | agent_message | agent_delegate | agent_delegate_result
    - data: event-type-specific payload
    - step: current ReAct step number
    - timestamp: Unix timestamp
    """
    await websocket.accept()
    logger.info("WebSocket connected for agent: %s", agent_id)

    # Replay buffered events the client may have missed
    last_event_id = websocket.query_params.get("last_event_id")
    missed = event_bus.get_event_history(agent_id, since_event_id=last_event_id)
    for ev in missed:
        try:
            await websocket.send_text(ev.to_json())
        except Exception:
            return  # client already gone

    # Subscribe to live events for this agent
    queue = event_bus.subscribe(agent_id)

    try:
        while True:
            try:
                # Wait for events with a timeout so we can send heartbeats
                event: StreamEvent = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_text(event.to_json())

                # "done" is NOT terminal for the WebSocket — the connection
                # stays open so the UI can stream subsequent runs without
                # needing to reconnect.

            except asyncio.TimeoutError:
                # Send heartbeat ping to keep connection alive
                try:
                    await websocket.send_json({"type": "heartbeat", "data": {}})
                except Exception:
                    break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for agent: %s", agent_id)
    except Exception as exc:
        logger.exception("WebSocket error for agent %s: %s", agent_id, exc)
    finally:
        event_bus.unsubscribe(queue, agent_id)


@router.websocket("/stream/all")
async def global_stream(websocket: WebSocket):
    """
    WebSocket endpoint that streams ALL agent events.
    Useful for the dashboard/sidebar to show global activity.
    """
    await websocket.accept()
    queue = event_bus.subscribe()  # global subscription

    try:
        while True:
            try:
                event: StreamEvent = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_text(event.to_json())
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"type": "heartbeat", "data": {}})
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        event_bus.unsubscribe(queue)


# ═══════════════════════════════════════════════════════════════════════
# v3: Multi-Agent Streaming Endpoints
# ═══════════════════════════════════════════════════════════════════════

@router.websocket("/multi-agent/stream")
async def multi_agent_stream(websocket: WebSocket):
    """
    WebSocket endpoint that streams ALL inter-agent communication events.

    Events include:
    - agent_spawn: when a sub-agent is created
    - agent_message: agent-to-agent messages
    - agent_delegate: task delegation events
    - agent_delegate_result: sub-agent task completion
    - agent_terminate: sub-agent shutdown
    - shared_memory_write: shared memory updates

    The UI uses this for the multi-agent visualization panel.
    """
    await websocket.accept()
    logger.info("Multi-agent WebSocket connected")

    # Subscribe to global events + message router events
    event_queue = event_bus.subscribe()  # all events

    # Also subscribe to message router if available
    msg_queue = None
    if _orchestrator:
        msg_queue = _orchestrator.router.subscribe_global()

    try:
        while True:
            # Check both queues with timeout
            done = set()
            pending = set()

            tasks = [asyncio.create_task(event_queue.get())]
            if msg_queue:
                tasks.append(asyncio.create_task(msg_queue.get()))

            try:
                done, pending = await asyncio.wait(
                    tasks,
                    timeout=30.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except Exception:
                break

            # Cancel any pending tasks
            for task in pending:
                task.cancel()

            if not done:
                # Timeout — send heartbeat
                try:
                    await websocket.send_json({"type": "heartbeat", "data": {}})
                except Exception:
                    break
                continue

            for task in done:
                try:
                    result = task.result()
                    if isinstance(result, StreamEvent):
                        # Filter to only multi-agent relevant events
                        multi_agent_types = {
                            "agent_spawn", "agent_message", "agent_delegate",
                            "agent_delegate_result", "agent_terminate",
                            "shared_memory_write", "shared_memory_read", "info",
                        }
                        if result.type.value in multi_agent_types:
                            # Check if it's a multi-agent INFO event
                            if result.type.value == "info":
                                if "multi_agent_event" in result.data:
                                    await websocket.send_text(result.to_json())
                            else:
                                await websocket.send_text(result.to_json())
                    elif isinstance(result, dict):
                        # Message router event
                        await websocket.send_json({
                            "type": "agent_message",
                            "data": result,
                        })
                except Exception as exc:
                    logger.debug("Error processing multi-agent event: %s", exc)

    except WebSocketDisconnect:
        logger.info("Multi-agent WebSocket disconnected")
    except Exception as exc:
        logger.exception("Multi-agent WebSocket error: %s", exc)
    finally:
        event_bus.unsubscribe(event_queue)
        if msg_queue and _orchestrator:
            _orchestrator.router.unsubscribe_global(msg_queue)


@router.websocket("/shared-memory/stream")
async def shared_memory_stream(websocket: WebSocket):
    """
    WebSocket endpoint that streams ALL shared memory updates.
    Useful for the shared memory inspector in the UI.
    """
    await websocket.accept()
    logger.info("Shared memory WebSocket connected")

    if not _orchestrator:
        await websocket.send_json({"type": "error", "data": {"message": "Orchestrator not initialized"}})
        await websocket.close()
        return

    queue = _orchestrator.shared_memory.subscribe()

    try:
        while True:
            try:
                entry = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json({
                    "type": "shared_memory_update",
                    "data": entry,
                })
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"type": "heartbeat", "data": {}})
                except Exception:
                    break
    except WebSocketDisconnect:
        logger.info("Shared memory WebSocket disconnected")
    finally:
        _orchestrator.shared_memory.unsubscribe(queue)


@router.websocket("/shared-memory/{channel}/stream")
async def channel_stream(websocket: WebSocket, channel: str):
    """
    WebSocket endpoint that streams updates for a specific shared memory channel.
    """
    await websocket.accept()
    logger.info("Channel WebSocket connected: %s", channel)

    if not _orchestrator:
        await websocket.send_json({"type": "error", "data": {"message": "Orchestrator not initialized"}})
        await websocket.close()
        return

    queue = _orchestrator.shared_memory.subscribe(channel)

    try:
        while True:
            try:
                entry = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json({
                    "type": "channel_update",
                    "channel": channel,
                    "data": entry,
                })
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"type": "heartbeat", "data": {}})
                except Exception:
                    break
    except WebSocketDisconnect:
        logger.info("Channel WebSocket disconnected: %s", channel)
    finally:
        _orchestrator.shared_memory.unsubscribe(queue, channel)
