import React, { useState, useEffect, useCallback } from 'react';
import NavHeader from '../components/NavHeader';
import ThinkingRunLog from '../components/ThinkingRunLog';
import { useAgentStream } from '../hooks/useAgentStream';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Agent {
  id: string;
  name: string;
  description: string;
  status: string;
  tools: string[];
  llm: { provider: string; model: string };
  runs_count: number;
}

export default function RunPage() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [goal, setGoal] = useState('');
  const [activeGoal, setActiveGoal] = useState('');
  const [loading, setLoading] = useState(false);

  // Fetch agents
  useEffect(() => {
    const fetchAgents = async () => {
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
    };
    fetchAgents();
    const interval = setInterval(fetchAgents, 15000);
    return () => clearInterval(interval);
  }, [selectedAgentId]);

  const { events, connected, isRunning, connect, clearEvents, pollEvents } =
    useAgentStream({
      agentId: selectedAgentId || '',
      autoConnect: !!selectedAgentId,
    });

  // Polling fallback
  useEffect(() => {
    if (!isRunning) return;
    const id = setInterval(pollEvents, 3000);
    return () => clearInterval(id);
  }, [isRunning, pollEvents]);

  // Stop loading when done/error
  useEffect(() => {
    const last = events[events.length - 1];
    if (last && (last.type === 'done' || last.type === 'error')) {
      setLoading(false);
    }
  }, [events]);

  const handleRun = useCallback(async () => {
    if (!goal.trim() || loading || !selectedAgentId) return;

    setLoading(true);
    setActiveGoal(goal.trim());
    clearEvents();

    if (!connected) connect();

    try {
      const resp = await fetch(`${API_BASE}/agents/${selectedAgentId}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ goal: goal.trim() }),
      });
      if (!resp.ok) {
        const err = await resp.json();
        alert(`Error: ${err.detail || 'Failed to start run'}`);
        setLoading(false);
      }
    } catch (err) {
      alert(`Network error: ${err}`);
      setLoading(false);
    }
  }, [goal, selectedAgentId, loading, connected, connect, clearEvents]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleRun();
    }
  };

  const selectedAgent = agents.find((a) => a.id === selectedAgentId);

  return (
    <div className="run-page-layout">
      <NavHeader agentCount={agents.length} />

      <div className="run-page-body">
        {/* Left: Agent selector */}
        <div className="run-sidebar">
          <div className="run-sidebar-title">Select Agent</div>
          {agents.map((agent) => (
            <div
              key={agent.id}
              className={`run-agent-card ${selectedAgentId === agent.id ? 'active' : ''}`}
              onClick={() => {
                setSelectedAgentId(agent.id);
                clearEvents();
                setActiveGoal('');
              }}
            >
              <div className={`agent-dot ${agent.status}`} />
              <div>
                <div className="run-agent-name">{agent.name}</div>
                <div className="run-agent-meta">
                  {agent.llm?.provider}/{agent.llm?.model} · {agent.tools.length} tools
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Right: Run area */}
        <div className="run-content">
          {/* Agent info bar */}
          {selectedAgent && (
            <div className="run-agent-bar">
              <span className="run-agent-bar-name">{selectedAgent.name}</span>
              <span className="run-agent-bar-badge">
                {selectedAgent.llm?.provider}/{selectedAgent.llm?.model}
              </span>
              <span className="run-agent-bar-tools">
                {selectedAgent.tools.length} tools available
              </span>
              <span
                className={`run-ws-status ${connected ? 'connected' : 'disconnected'}`}
              >
                {connected ? '● Connected' : '○ Disconnected'}
              </span>
            </div>
          )}

          {/* Goal input */}
          <div className="run-goal-area">
            <div className="run-goal-row">
              <input
                className="run-goal-input"
                type="text"
                placeholder="Enter a goal... e.g., 'Research the latest AI frameworks and compare them'"
                value={goal}
                onChange={(e) => setGoal(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isRunning}
              />
              <button
                className="run-goal-btn"
                onClick={handleRun}
                disabled={!goal.trim() || isRunning}
              >
                {isRunning ? '⟳ Running...' : '▸ Run Agent'}
              </button>
            </div>
          </div>

          {/* Results */}
          <ThinkingRunLog events={events} goal={activeGoal || undefined} />
        </div>
      </div>
    </div>
  );
}
