import { useState, useEffect, useRef, useCallback } from 'react';

export interface MultiAgentEvent {
  type: string;
  data: Record<string, any>;
  agent_id?: string;
  timestamp?: number;
}

interface UseMultiAgentStreamOptions {
  autoConnect?: boolean;
  onEvent?: (event: MultiAgentEvent) => void;
}

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

export function useMultiAgentStream({ autoConnect = false, onEvent }: UseMultiAgentStreamOptions = {}) {
  const [events, setEvents] = useState<MultiAgentEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(`${WS_BASE}/multi-agent/stream`);

    ws.onopen = () => {
      setConnected(true);
      console.log('[WS] Multi-agent stream connected');
    };

    ws.onmessage = (msg) => {
      try {
        const event: MultiAgentEvent = JSON.parse(msg.data);
        if (event.type === 'heartbeat') return;

        setEvents((prev) => [...prev.slice(-200), event]); // Keep last 200 events
        onEventRef.current?.(event);
      } catch (e) {
        console.warn('[WS] Failed to parse multi-agent event:', msg.data);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      console.log('[WS] Multi-agent stream disconnected');
    };

    ws.onerror = (err) => {
      console.error('[WS] Multi-agent stream error:', err);
    };

    wsRef.current = ws;
  }, []);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setConnected(false);
  }, []);

  const clearEvents = useCallback(() => {
    setEvents([]);
  }, []);

  useEffect(() => {
    if (autoConnect) connect();
    return () => disconnect();
  }, [autoConnect, connect, disconnect]);

  return { events, connected, connect, disconnect, clearEvents };
}
