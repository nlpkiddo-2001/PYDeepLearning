import { useState, useCallback, useEffect, useRef } from 'react';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  /** Tool execution steps (populated when using /run streaming) */
  toolSteps?: ToolStep[];
  isStreaming?: boolean;
}

export interface ToolStep {
  step: number;
  type: 'plan' | 'tool_call' | 'tool_result' | 'error' | 'retry' | 'info';
  thought?: string;
  toolName?: string;
  toolArgs?: Record<string, any>;
  toolResult?: string;
  error?: string;
  isRetry?: boolean;
  timestamp: number;
}

export interface ChatSession {
  id: string;
  title: string;
  agentId: string;
  messages: ChatMessage[];
  createdAt: number;
  updatedAt: number;
}

const STORAGE_KEY = 'agentforge_chat_sessions';

function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function loadSessions(): ChatSession[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveSessions(sessions: ChatSession[]) {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  } catch {}
}

export function useChatSessions(agentId: string | null) {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  // Keep a ref that always points to current sessions to avoid stale closures
  const sessionsRef = useRef<ChatSession[]>(sessions);
  sessionsRef.current = sessions;

  // Load from localStorage on mount
  useEffect(() => {
    const loaded = loadSessions();
    setSessions(loaded);
    sessionsRef.current = loaded;
  }, []);

  // Filter sessions for current agent
  const agentSessions = sessions
    .filter((s) => !agentId || s.agentId === agentId)
    .sort((a, b) => b.updatedAt - a.updatedAt);

  const activeSession = sessions.find((s) => s.id === activeSessionId) || null;

  const persist = useCallback((updated: ChatSession[]) => {
    sessionsRef.current = updated;
    setSessions(updated);
    saveSessions(updated);
  }, []);

  const createSession = useCallback(
    (agId: string, title?: string): ChatSession => {
      const session: ChatSession = {
        id: generateId(),
        title: title || 'New Chat',
        agentId: agId,
        messages: [],
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };
      const updated = [session, ...sessionsRef.current];
      persist(updated);
      setActiveSessionId(session.id);
      return session;
    },
    [persist],
  );

  const deleteSession = useCallback(
    (sessionId: string) => {
      const updated = sessionsRef.current.filter((s) => s.id !== sessionId);
      persist(updated);
      setActiveSessionId((prev) => {
        if (prev === sessionId) {
          return updated.length > 0 ? updated[0].id : null;
        }
        return prev;
      });
    },
    [persist],
  );

  const addMessage = useCallback(
    (sessionId: string, message: ChatMessage) => {
      const updated = sessionsRef.current.map((s) => {
        if (s.id !== sessionId) return s;
        const msgs = [...s.messages, message];
        // Auto-title from first user message
        const title =
          s.messages.length === 0 && message.role === 'user'
            ? message.content.slice(0, 50) + (message.content.length > 50 ? '…' : '')
            : s.title;
        return { ...s, messages: msgs, title, updatedAt: Date.now() };
      });
      persist(updated);
    },
    [persist],
  );

  const updateLastMessage = useCallback(
    (sessionId: string, updater: (msg: ChatMessage) => ChatMessage) => {
      const updated = sessionsRef.current.map((s) => {
        if (s.id !== sessionId || s.messages.length === 0) return s;
        const msgs = [...s.messages];
        msgs[msgs.length - 1] = updater(msgs[msgs.length - 1]);
        return { ...s, messages: msgs, updatedAt: Date.now() };
      });
      persist(updated);
    },
    [persist],
  );

  const clearSessions = useCallback(
    (agId?: string) => {
      const updated = agId ? sessionsRef.current.filter((s) => s.agentId !== agId) : [];
      persist(updated);
      setActiveSessionId(null);
    },
    [persist],
  );

  return {
    sessions: agentSessions,
    activeSession,
    activeSessionId,
    setActiveSessionId,
    createSession,
    deleteSession,
    addMessage,
    updateLastMessage,
    clearSessions,
  };
}
