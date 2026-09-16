import { useState, useEffect, useRef, useCallback } from 'react';

export interface StreamEvent {
  type: 'plan' | 'tool_call' | 'tool_result' | 'memory_read' | 'done' | 'error' | 'retry' | 'info' | 'heartbeat';
  data: Record<string, any>;
  agent_id: string;
  task_id: string;
  step: number;
  timestamp: number;
  event_id: string;
}

interface UseAgentStreamOptions {
  agentId: string;
  autoConnect?: boolean;
  onEvent?: (event: StreamEvent) => void;
}

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const MAX_RECONNECT_DELAY = 10000; // 10s ceiling
const INITIAL_RECONNECT_DELAY = 1000; // 1s

export function useAgentStream({ agentId, autoConnect = false, onEvent }: UseAgentStreamOptions) {
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectDelayRef = useRef(INITIAL_RECONNECT_DELAY);
  const intentionalCloseRef = useRef(false);
  // Track the last event_id we received so reconnects can request replay
  const lastEventIdRef = useRef<string | null>(null);
  // Deduplicate events across replays using a Set of event_ids
  const seenEventIdsRef = useRef<Set<string>>(new Set());

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  const addEvent = useCallback((event: StreamEvent) => {
    // Deduplicate: skip events we've already seen
    if (seenEventIdsRef.current.has(event.event_id)) return;
    seenEventIdsRef.current.add(event.event_id);
    lastEventIdRef.current = event.event_id;

    setEvents((prev) => [...prev, event]);
    onEventRef.current?.(event);

    if (event.type === 'plan' || event.type === 'tool_call') {
      setIsRunning(true);
    }
    if (event.type === 'done' || event.type === 'error') {
      setIsRunning(false);
    }
  }, []);

  const connect = useCallback(() => {
    // Don't open a duplicate connection
    if (wsRef.current?.readyState === WebSocket.OPEN ||
        wsRef.current?.readyState === WebSocket.CONNECTING) return;

    clearReconnectTimer();
    intentionalCloseRef.current = false;

    // Include last_event_id so the server replays missed events
    let url = `${WS_BASE}/agents/${agentId}/stream`;
    if (lastEventIdRef.current) {
      url += `?last_event_id=${lastEventIdRef.current}`;
    }

    const ws = new WebSocket(url);

    ws.onopen = () => {
      setConnected(true);
      reconnectDelayRef.current = INITIAL_RECONNECT_DELAY; // reset backoff
      console.log(`[WS] Connected to agent: ${agentId}`);
    };

    ws.onmessage = (msg) => {
      try {
        const event: StreamEvent = JSON.parse(msg.data);
        if (event.type === 'heartbeat') return;
        addEvent(event);
      } catch (e) {
        console.warn('[WS] Failed to parse:', msg.data);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
      console.log(`[WS] Disconnected from agent: ${agentId}`);

      // Auto-reconnect with exponential backoff (unless intentionally closed)
      if (!intentionalCloseRef.current) {
        const delay = reconnectDelayRef.current;
        console.log(`[WS] Reconnecting in ${delay}ms...`);
        reconnectTimerRef.current = setTimeout(() => {
          reconnectDelayRef.current = Math.min(delay * 2, MAX_RECONNECT_DELAY);
          connect();
        }, delay);
      }
    };

    ws.onerror = (err) => {
      console.error('[WS] Error:', err);
    };

    wsRef.current = ws;
  }, [agentId, clearReconnectTimer, addEvent]);

  const disconnect = useCallback(() => {
    intentionalCloseRef.current = true;
    clearReconnectTimer();
    wsRef.current?.close();
    wsRef.current = null;
    setConnected(false);
  }, [clearReconnectTimer]);

  const clearEvents = useCallback(() => {
    setEvents([]);
    seenEventIdsRef.current.clear();
    lastEventIdRef.current = null;
  }, []);

  /** Poll the REST event-history endpoint as a fallback (useful when WS was down). */
  const pollEvents = useCallback(async () => {
    try {
      const since = lastEventIdRef.current ? `?since=${lastEventIdRef.current}` : '';
      const resp = await fetch(`${API_BASE}/agents/${agentId}/events${since}`);
      if (!resp.ok) return;
      const data = await resp.json();
      const eventsArr: StreamEvent[] = (data.events || []).map((raw: string) => JSON.parse(raw));
      for (const ev of eventsArr) {
        addEvent(ev);
      }
    } catch {
      // polling is best-effort
    }
  }, [agentId, addEvent]);

  useEffect(() => {
    if (autoConnect) connect();
    return () => disconnect();
  }, [agentId, autoConnect, connect, disconnect]);

  return { events, connected, isRunning, connect, disconnect, clearEvents, pollEvents };
}
