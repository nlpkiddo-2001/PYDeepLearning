import React, { useState, useCallback, useRef, useEffect } from 'react';
import MarkdownRenderer from './MarkdownRenderer';
import ThinkingBlock from './ThinkingBlock';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface ChatTabProps {
  agentId: string;
}

/* Provider presets matching agent.yaml */
const MODEL_OPTIONS = [
  { key: 'gemini', label: 'Gemini 3 Flash', provider: 'gemini', model: 'gemini-3-flash-preview', api_key: 'AIzaSyABiD612PXe2QKSnbfVbNHAzKcOmSCJd90' },
  { key: 'vllm',   label: 'GLM-5 (vLLM)',   provider: 'vllm',   model: 'glm-5', base_url: 'http://103.42.51.233:443/llm/text/api/glm/v1', jwt_secret: 'eyJhbGciOiJI' },
];

export default function ChatTab({ agentId }: ChatTabProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeModel, setActiveModel] = useState('gemini');
  const [switching, setSwitching] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Detect current model from agent info
  useEffect(() => {
    (async () => {
      try {
        const resp = await fetch(`${API_BASE}/agents/${agentId}`);
        if (resp.ok) {
          const data = await resp.json();
          const prov = data.llm?.provider;
          if (prov) {
            const match = MODEL_OPTIONS.find(m => m.provider === prov);
            if (match) setActiveModel(match.key);
          }
        }
      } catch {}
    })();
  }, [agentId]);

  // Load latest history from memory on mount
  useEffect(() => {
    (async () => {
      try {
        const resp = await fetch(`${API_BASE}/agents/${agentId}/memory`);
        if (resp.ok) {
          const data = await resp.json();
          if (data.recent_history?.length) {
            setMessages(data.recent_history.map((m: any) => ({
              role: m.role as 'user' | 'assistant',
              content: m.content,
            })));
          }
        }
      } catch {}
    })();
  }, [agentId]);

  /* Switch the backend LLM provider when the user picks a different model */
  const handleModelSwitch = useCallback(async (key: string) => {
    if (key === activeModel || switching) return;
    const opt = MODEL_OPTIONS.find(m => m.key === key);
    if (!opt) return;

    setSwitching(true);
    try {
      const body: Record<string, any> = { provider: opt.provider, model: opt.model };
      if ((opt as any).api_key) body.api_key = (opt as any).api_key;
      if ((opt as any).base_url) body.base_url = (opt as any).base_url;
      if ((opt as any).jwt_secret) body.jwt_secret = (opt as any).jwt_secret;

      const resp = await fetch(`${API_BASE}/agents/${agentId}/config`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (resp.ok) {
        setActiveModel(key);
      } else {
        const err = await resp.json();
        const detail = typeof err.detail === 'object' ? JSON.stringify(err.detail) : err.detail;
        alert(`Failed to switch model: ${detail}`);
      }
    } catch (err) {
      alert(`Network error switching model: ${err}`);
    } finally {
      setSwitching(false);
    }
  }, [activeModel, agentId, switching]);

  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(async () => {
    if (!input.trim() || loading) return;

    const userMsg = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: userMsg }]);
    setLoading(true);

    // Abort any previous in-flight request
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    // 90-second timeout so the UI never hangs forever
    const timeout = setTimeout(() => controller.abort(), 90_000);

    try {
      const resp = await fetch(`${API_BASE}/agents/${agentId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg }),
        signal: controller.signal,
      });
      if (resp.ok) {
        const data = await resp.json();
        setMessages((prev) => [...prev, { role: 'assistant', content: data.response }]);
      } else {
        let detail = 'Request failed';
        try {
          const err = await resp.json();
          detail = err.detail || `HTTP ${resp.status}`;
        } catch {}
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: `Error: ${detail}` },
        ]);
      }
    } catch (err: any) {
      const msg = err?.name === 'AbortError'
        ? 'Request timed out — the LLM server may be unreachable. Check the backend logs.'
        : `Network error: ${err}`;
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: msg },
      ]);
    } finally {
      clearTimeout(timeout);
      abortRef.current = null;
      setLoading(false);
    }
  }, [input, agentId, loading]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="chat-tab">
      {/* Model Selector */}
      <div className="model-selector-bar">
        <span className="model-selector-label">Model:</span>
        <div className="model-selector-buttons">
          {MODEL_OPTIONS.map((opt) => (
            <button
              key={opt.key}
              className={`model-btn ${activeModel === opt.key ? 'active' : ''}`}
              onClick={() => handleModelSwitch(opt.key)}
              disabled={switching}
              title={`${opt.provider} — ${opt.model}`}
            >
              {opt.key === 'gemini' ? '✦' : '⚡'} {opt.label}
            </button>
          ))}
        </div>
        {switching && <span className="model-switching">Switching…</span>}
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon">💬</div>
            <p>Start a conversation with this agent.<br />Chat history feeds into the agent&apos;s short-term memory.</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`chat-message ${msg.role}`}>
            <div className="msg-role">{msg.role}</div>
            <div className="msg-bubble">
              {msg.role === 'assistant' ? (
                <ThinkingBlock content={msg.content} />
              ) : (
                <MarkdownRenderer content={msg.content} />
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="chat-message assistant">
            <div className="msg-role">assistant</div>
            <div className="msg-bubble" style={{ opacity: 0.6 }}>Thinking...</div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <div className="chat-input-area">
        <input
          className="chat-input"
          type="text"
          placeholder="Send a message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />
        <button
          className="send-btn"
          onClick={sendMessage}
          disabled={!input.trim() || loading}
        >
          Send
        </button>
      </div>
    </div>
  );
}
