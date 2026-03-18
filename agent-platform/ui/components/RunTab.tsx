import React, { useState, useCallback, useEffect, useRef } from 'react';
import ThinkingRunLog from './ThinkingRunLog';
import { useAgentStream } from '../hooks/useAgentStream';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const MODEL_OPTIONS = [
  { key: 'gemini', label: 'Gemini 3 Flash', provider: 'gemini', model: 'gemini-3-flash-preview', api_key: 'AIzaSyABiD612PXe2QKSnbfVbNHAzKcOmSCJd90' },
  { key: 'vllm',   label: 'GLM-5 (vLLM)',   provider: 'vllm',   model: 'glm-5', base_url: '', jwt_secret: '' },
];

interface RunTabProps {
  agentId: string;
}

export default function RunTab({ agentId }: RunTabProps) {
  const [goal, setGoal] = useState('');
  const [activeGoal, setActiveGoal] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeModel, setActiveModel] = useState('gemini');
  const [switching, setSwitching] = useState(false);

  const { events, connected, isRunning, connect, clearEvents, pollEvents } = useAgentStream({
    agentId,
    autoConnect: true,
  });

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

  // Model switch handler
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
      if (resp.ok) setActiveModel(key);
    } catch {}
    setSwitching(false);
  }, [activeModel, agentId, switching]);

  // Polling fallback
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const runActiveRef = useRef(false);

  useEffect(() => {
    if (runActiveRef.current && !pollRef.current) {
      pollRef.current = setInterval(() => { pollEvents(); }, 3000);
    }
    const lastEvent = events[events.length - 1];
    if (lastEvent && (lastEvent.type === 'done' || lastEvent.type === 'error')) {
      runActiveRef.current = false;
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    }
    return () => {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    };
  }, [events, pollEvents]);

  const handleRun = useCallback(async () => {
    if (!goal.trim() || loading) return;

    setLoading(true);
    setActiveGoal(goal.trim());
    clearEvents();

    if (!connected) connect();

    try {
      const resp = await fetch(`${API_BASE}/agents/${agentId}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ goal: goal.trim() }),
      });
      if (!resp.ok) {
        const err = await resp.json();
        alert(`Error: ${err.detail || 'Failed to start run'}`);
      } else {
        runActiveRef.current = true;
      }
    } catch (err) {
      alert(`Network error: ${err}`);
    } finally {
      setLoading(false);
    }
  }, [goal, agentId, loading, connected, connect, clearEvents]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleRun();
    }
  };

  return (
    <div className="run-tab">
      {/* Model selector bar */}
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

      <div className="goal-input-area">
        <div className="goal-input-row">
          <input
            className="goal-input"
            type="text"
            placeholder="Enter a goal... e.g., 'Find ML engineer jobs posted this week'"
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isRunning}
          />
          <button
            className="run-btn"
            onClick={handleRun}
            disabled={!goal.trim() || isRunning}
          >
            {isRunning ? '⟳ Running...' : '▸ Run'}
          </button>
        </div>
        <div style={{
          marginTop: '8px',
          fontSize: '11px',
          color: connected ? 'var(--accent-green)' : 'var(--accent-red)',
        }}>
          {connected ? '● WebSocket connected' : '○ WebSocket disconnected'}
        </div>
      </div>
      <ThinkingRunLog events={events} goal={activeGoal || undefined} />
    </div>
  );
}
