import React, { useState, useEffect, useCallback, useRef } from 'react';
import NavHeader from '../components/NavHeader';
import ChatSidebar from '../components/ChatSidebar';
import ChatMessageComponent from '../components/ChatMessage';
import MarkdownRenderer from '../components/MarkdownRenderer';
import { useChatSessions, ChatMessage, ToolStep } from '../hooks/useChatSessions';
import { useAgentStream, StreamEvent } from '../hooks/useAgentStream';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const MODEL_OPTIONS = [
  { key: 'gemini', label: 'Gemini 3 Flash', provider: 'gemini', model: 'gemini-3-flash-preview', api_key: '' },
  { key: 'vllm', label: 'GLM-5 (vLLM)', provider: 'vllm', model: 'glm-5', base_url: '', jwt_secret: '' },
];

interface Agent {
  id: string;
  name: string;
  description: string;
  status: string;
  tools: string[];
  llm: { provider: string; model: string };
}

export default function ChatPage() {
  // ── Agent state ────────────────────────────────────────
  const [agents, setAgents] = useState<Agent[]>([]);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [activeModel, setActiveModel] = useState('gemini');
  const [switching, setSwitching] = useState(false);
  const [useTools, setUseTools] = useState(false);

  // ── Chat sessions ──────────────────────────────────────
  const {
    sessions,
    activeSession,
    activeSessionId,
    setActiveSessionId,
    createSession,
    deleteSession,
    addMessage,
    updateLastMessage,
  } = useChatSessions(selectedAgentId);

  // ── Input state ────────────────────────────────────────
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // ── WebSocket stream (for tool execution mode) ─────────
  const {
    events: wsEvents,
    connected: wsConnected,
    connect: wsConnect,
    clearEvents: wsClearEvents,
  } = useAgentStream({
    agentId: selectedAgentId || '',
    autoConnect: !!selectedAgentId && useTools,
  });

  // Track current streaming message ID so we can update it with WS events
  const streamingMsgIdRef = useRef<string | null>(null);

  // ── Fetch agents ───────────────────────────────────────
  const fetchAgents = useCallback(async () => {
    try {
      const resp = await fetch(`${API_BASE}/agents`);
      if (resp.ok) {
        const data = await resp.json();
        setAgents(data.agents || []);
        if (!selectedAgentId && data.agents?.length > 0) {
          setSelectedAgentId(data.agents[0].id);
        }
      }
    } catch {}
  }, [selectedAgentId]);

  useEffect(() => {
    fetchAgents();
    const interval = setInterval(fetchAgents, 15000);
    return () => clearInterval(interval);
  }, [fetchAgents]);

  // ── Detect current model ───────────────────────────────
  useEffect(() => {
    if (!selectedAgentId) return;
    (async () => {
      try {
        const resp = await fetch(`${API_BASE}/agents/${selectedAgentId}`);
        if (resp.ok) {
          const data = await resp.json();
          const prov = data.llm?.provider;
          if (prov) {
            const match = MODEL_OPTIONS.find((m) => m.provider === prov);
            if (match) setActiveModel(match.key);
          }
        }
      } catch {}
    })();
  }, [selectedAgentId]);

  // ── Auto-scroll ────────────────────────────────────────
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeSession?.messages]);

  // ── Handle WS events for tool streaming ────────────────
  useEffect(() => {
    if (!streamingMsgIdRef.current || !activeSessionId) return;

    const msgId = streamingMsgIdRef.current;
    const sessionId = activeSessionId;

    // Convert WS events to tool steps
    const toolSteps: ToolStep[] = [];
    let finalResult = '';
    let isDone = false;

    for (const ev of wsEvents) {
      if (ev.type === 'info' && ev.step === 0) continue;

      if (ev.type === 'plan') {
        toolSteps.push({
          step: ev.step,
          type: 'plan',
          thought: ev.data.thought,
          timestamp: ev.timestamp,
        });
      } else if (ev.type === 'tool_call') {
        toolSteps.push({
          step: ev.step,
          type: 'tool_call',
          toolName: ev.data.tool,
          toolArgs: ev.data.args,
          timestamp: ev.timestamp,
        });
      } else if (ev.type === 'tool_result') {
        toolSteps.push({
          step: ev.step,
          type: 'tool_result',
          toolResult: ev.data.result,
          timestamp: ev.timestamp,
        });
      } else if (ev.type === 'error') {
        toolSteps.push({
          step: ev.step,
          type: 'error',
          error: ev.data.message,
          timestamp: ev.timestamp,
        });
      } else if (ev.type === 'retry') {
        toolSteps.push({
          step: ev.step,
          type: 'retry',
          error: ev.data.error,
          isRetry: true,
          timestamp: ev.timestamp,
        });
      } else if (ev.type === 'done') {
        finalResult = ev.data.result || '';
        isDone = true;
      }
    }

    updateLastMessage(sessionId, (msg) => {
      if (msg.id !== msgId) return msg;
      return {
        ...msg,
        toolSteps,
        content: finalResult,
        isStreaming: !isDone,
      };
    });

    if (isDone) {
      streamingMsgIdRef.current = null;
      setLoading(false);
    }
  }, [wsEvents, activeSessionId, updateLastMessage]);

  // ── Model switch ───────────────────────────────────────
  const handleModelSwitch = useCallback(
    async (key: string) => {
      if (key === activeModel || switching || !selectedAgentId) return;
      const opt = MODEL_OPTIONS.find((m) => m.key === key);
      if (!opt) return;

      setSwitching(true);
      try {
        const body: Record<string, any> = { provider: opt.provider, model: opt.model };
        if ((opt as any).api_key) body.api_key = (opt as any).api_key;
        if ((opt as any).base_url) body.base_url = (opt as any).base_url;
        if ((opt as any).jwt_secret) body.jwt_secret = (opt as any).jwt_secret;

        const resp = await fetch(`${API_BASE}/agents/${selectedAgentId}/config`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (resp.ok) setActiveModel(key);
      } catch {}
      setSwitching(false);
    },
    [activeModel, selectedAgentId, switching],
  );

  // ── Send message ───────────────────────────────────────
  const sendMessage = useCallback(async () => {
    if (!input.trim() || loading || !selectedAgentId) return;

    const userText = input.trim();
    setInput('');

    // Ensure session exists
    let sessionId = activeSessionId;
    if (!sessionId) {
      const s = createSession(selectedAgentId);
      sessionId = s.id;
    }

    // Add user message
    const userMsg: ChatMessage = {
      id: Date.now().toString(36) + 'u',
      role: 'user',
      content: userText,
      timestamp: Date.now(),
    };
    addMessage(sessionId, userMsg);
    setLoading(true);

    if (useTools) {
      // ── Tool mode: use /run endpoint + WebSocket streaming ──
      const assistantMsgId = Date.now().toString(36) + 'a';
      const assistantMsg: ChatMessage = {
        id: assistantMsgId,
        role: 'assistant',
        content: '',
        timestamp: Date.now(),
        toolSteps: [],
        isStreaming: true,
      };
      addMessage(sessionId, assistantMsg);
      streamingMsgIdRef.current = assistantMsgId;
      wsClearEvents();

      if (!wsConnected) wsConnect();

      try {
        await fetch(`${API_BASE}/agents/${selectedAgentId}/run`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ goal: userText }),
        });
      } catch (err) {
        updateLastMessage(sessionId, (msg) => ({
          ...msg,
          content: `Network error: ${err}`,
          isStreaming: false,
        }));
        streamingMsgIdRef.current = null;
        setLoading(false);
      }
    } else {
      // ── Chat mode: SSE streaming via /chat/stream ──
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      const timeout = setTimeout(() => controller.abort(), 120_000);

      // Add streaming placeholder
      const assistantMsgId = Date.now().toString(36) + 'a';
      const placeholderMsg: ChatMessage = {
        id: assistantMsgId,
        role: 'assistant',
        content: '',
        timestamp: Date.now(),
        isStreaming: true,
      };
      addMessage(sessionId, placeholderMsg);

      try {
        const resp = await fetch(`${API_BASE}/agents/${selectedAgentId}/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: userText }),
          signal: controller.signal,
        });

        if (resp.ok && resp.body) {
          const reader = resp.body.getReader();
          const decoder = new TextDecoder();
          let accumulated = '';

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const text = decoder.decode(value, { stream: true });
            const lines = text.split('\n');

            for (const line of lines) {
              if (!line.startsWith('data: ')) continue;
              try {
                const payload = JSON.parse(line.slice(6));
                if (payload.chunk) {
                  accumulated += payload.chunk;
                  updateLastMessage(sessionId, (msg) => ({
                    ...msg,
                    content: accumulated,
                    isStreaming: true,
                  }));
                }
                if (payload.done) {
                  updateLastMessage(sessionId, (msg) => ({
                    ...msg,
                    isStreaming: false,
                  }));
                }
                if (payload.error) {
                  updateLastMessage(sessionId, (msg) => ({
                    ...msg,
                    content: accumulated + `\n\nError: ${payload.error}`,
                    isStreaming: false,
                  }));
                }
              } catch {}
            }
          }

          // Ensure streaming flag is off
          updateLastMessage(sessionId, (msg) => ({
            ...msg,
            isStreaming: false,
          }));
        } else if (resp.ok) {
          // Fallback if body is not streamable
          const data = await resp.json();
          updateLastMessage(sessionId, (msg) => ({
            ...msg,
            content: data.response || 'No response',
            isStreaming: false,
          }));
        } else {
          let detail = 'Request failed';
          try {
            const err = await resp.json();
            detail = err.detail || `HTTP ${resp.status}`;
          } catch {}
          updateLastMessage(sessionId, (msg) => ({
            ...msg,
            content: `Error: ${detail}`,
            isStreaming: false,
          }));
        }
      } catch (err: any) {
        const errMsg =
          err?.name === 'AbortError'
            ? 'Request timed out — the LLM server may be unreachable.'
            : `Network error: ${err}`;
        updateLastMessage(sessionId, (msg) => ({
          ...msg,
          content: errMsg,
          isStreaming: false,
        }));
      } finally {
        clearTimeout(timeout);
        abortRef.current = null;
        setLoading(false);
      }
    }
  }, [
    input,
    loading,
    selectedAgentId,
    activeSessionId,
    useTools,
    createSession,
    addMessage,
    updateLastMessage,
    wsClearEvents,
    wsConnect,
    wsConnected,
  ]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleNewChat = () => {
    if (!selectedAgentId) return;
    createSession(selectedAgentId);
  };

  const selectedAgent = agents.find((a) => a.id === selectedAgentId);
  const messages = activeSession?.messages || [];

  return (
    <div className="chat-page-layout">
      {/* Nav header */}
      <NavHeader agentCount={agents.length} />

      {/* Chat sessions sidebar */}
      <ChatSidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelect={setActiveSessionId}
        onNew={handleNewChat}
        onDelete={deleteSession}
        agentName={selectedAgent?.name}
      />

      {/* Main chat area */}
      <div className="chat-main">
        {/* Top bar: agent selector + model selector */}
        <div className="chat-topbar">
          <div className="chat-topbar-left">
            <select
              className="chat-agent-select"
              value={selectedAgentId || ''}
              onChange={(e) => setSelectedAgentId(e.target.value)}
            >
              {agents.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.name}
                </option>
              ))}
            </select>
            <div className="chat-model-pills">
              {MODEL_OPTIONS.map((opt) => (
                <button
                  key={opt.key}
                  className={`chat-model-pill ${activeModel === opt.key ? 'active' : ''}`}
                  onClick={() => handleModelSwitch(opt.key)}
                  disabled={switching}
                >
                  {opt.key === 'gemini' ? '✦' : '⚡'} {opt.label}
                </button>
              ))}
              {switching && <span className="chat-model-switching">Switching…</span>}
            </div>
          </div>
          <div className="chat-topbar-right">
            <label className="chat-tool-toggle-label">
              <input
                type="checkbox"
                checked={useTools}
                onChange={(e) => setUseTools(e.target.checked)}
              />
              <span>Use Tools</span>
            </label>
          </div>
        </div>

        {/* Messages area */}
        <div className="chat-messages-area">
          <div className="chat-messages-container">
            {messages.length === 0 && (
              <div className="chat-welcome">
                <div className="chat-welcome-icon">⚡</div>
                <h2>How can I help you today?</h2>
                <p>
                  Chat with{' '}
                  <strong>{selectedAgent?.name || 'an agent'}</strong>. Your
                  conversation is saved as a session.
                </p>
                {useTools && (
                  <p className="chat-welcome-tools">
                    🔧 <strong>Tool mode is ON</strong> — the agent will use tools and
                    you&apos;ll see each step of its reasoning process.
                  </p>
                )}
              </div>
            )}

            {messages.map((msg) => (
              <ChatMessageComponent key={msg.id} message={msg} />
            ))}

            <div ref={bottomRef} />
          </div>
        </div>

        {/* Input area */}
        <div className="chat-input-container">
          <div className="chat-input-box">
            <textarea
              ref={inputRef}
              className="chat-textarea"
              placeholder={
                useTools
                  ? 'Enter a goal... the agent will use tools to complete it'
                  : 'Message AgentForge...'
              }
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
              rows={1}
              onInput={(e) => {
                const el = e.target as HTMLTextAreaElement;
                el.style.height = 'auto';
                el.style.height = Math.min(el.scrollHeight, 200) + 'px';
              }}
            />
            <button
              className="chat-send-btn"
              onClick={sendMessage}
              disabled={!input.trim() || loading}
              title="Send message"
            >
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <path
                  d="M3 10l7-7m0 0l7 7m-7-7v14"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  transform="rotate(-90 10 10)"
                />
              </svg>
            </button>
          </div>
          <div className="chat-input-footer">
            <span>
              {selectedAgent?.name} · {selectedAgent?.llm?.provider}/{selectedAgent?.llm?.model}
              {useTools && ' · 🔧 Tools enabled'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
