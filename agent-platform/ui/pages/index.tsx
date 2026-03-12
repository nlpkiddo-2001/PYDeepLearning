import React, { useState, useEffect, useCallback } from 'react';
import AgentSidebar from '../components/AgentSidebar';
import RunTab from '../components/RunTab';
import ChatTab from '../components/ChatTab';
import ConfigTab from '../components/ConfigTab';
import RightPanel from '../components/RightPanel';
import WorkflowsPage from './workflows';
import MultiAgentPanel from '../components/MultiAgentPanel';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

type TabName = 'run' | 'chat' | 'config' | 'multi-agent';
type AppView = 'agents' | 'workflows';

interface Agent {
  id: string;
  name: string;
  description: string;
  status: string;
  tools: string[];
  llm: { provider: string; model: string };
  runs_count: number;
  total_tokens: number;
  total_tool_calls: number;
}

export default function Home() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabName>('run');
  const [error, setError] = useState<string | null>(null);
  const [appView, setAppView] = useState<AppView>('agents');

  // Fetch agents on mount and periodically
  const fetchAgents = useCallback(async () => {
    try {
      const resp = await fetch(`${API_BASE}/agents`);
      if (resp.ok) {
        const data = await resp.json();
        setAgents(data.agents || []);
        setError(null);

        // Auto-select first agent if none selected
        if (!selectedAgentId && data.agents?.length > 0) {
          setSelectedAgentId(data.agents[0].id);
        }
      }
    } catch (err) {
      setError('Cannot connect to AgentForge backend. Is it running on port 8000?');
    }
  }, [selectedAgentId]);

  useEffect(() => {
    fetchAgents();
    const interval = setInterval(fetchAgents, 10000);
    return () => clearInterval(interval);
  }, [fetchAgents]);

  const selectedAgent = agents.find((a) => a.id === selectedAgentId);

  return (
    <div className={`app-layout ${appView === 'workflows' ? 'full-width' : ''}`}>
      {/* ── Header ──────────────────────────────── */}
      <header className="app-header">
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <h1>⚡ AgentForge</h1>
          <span className="subtitle">Agentic AI Platform</span>
          {/* View Switcher */}
          <div className="view-switcher">
            <button
              className={`view-btn ${appView === 'agents' ? 'active' : ''}`}
              onClick={() => setAppView('agents')}
            >
              🤖 Agents
            </button>
            <button
              className={`view-btn ${appView === 'workflows' ? 'active' : ''}`}
              onClick={() => setAppView('workflows')}
            >
              🔧 Workflows
            </button>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          {error && (
            <span style={{ color: 'var(--accent-red)', fontSize: '12px' }}>{error}</span>
          )}
          <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
            {agents.length} agent{agents.length !== 1 ? 's' : ''} registered
          </span>
        </div>
      </header>

      {/* ── Agents View ─────────────────────────── */}
      {appView === 'agents' && (
        <>
          {/* ── Sidebar ─────────────────────────────── */}
          <AgentSidebar
            agents={agents}
            selectedId={selectedAgentId}
            onSelect={(id) => {
              setSelectedAgentId(id);
              setActiveTab('run');
            }}
          />

      {/* ── Main Content ────────────────────────── */}
      <div className="main-content">
        {selectedAgent ? (
          <>
            {/* Tab bar */}
            <div className="tab-bar">
              <div
                className={`tab ${activeTab === 'run' ? 'active' : ''}`}
                onClick={() => setActiveTab('run')}
              >
                ▸ Run
              </div>
              <div
                className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
                onClick={() => setActiveTab('chat')}
              >
                💬 Chat
              </div>
              <div
                className={`tab ${activeTab === 'config' ? 'active' : ''}`}
                onClick={() => setActiveTab('config')}
              >
                ⚙ Config
              </div>
              <div
                className={`tab ${activeTab === 'multi-agent' ? 'active' : ''}`}
                onClick={() => setActiveTab('multi-agent')}
              >
                🌐 Multi-Agent
              </div>
              <div style={{
                marginLeft: 'auto',
                display: 'flex',
                alignItems: 'center',
                fontSize: '12px',
                color: 'var(--text-secondary)',
              }}>
                {selectedAgent.name}
                <span style={{
                  marginLeft: '8px',
                  padding: '2px 8px',
                  borderRadius: '10px',
                  fontSize: '10px',
                  background: 'var(--bg-tertiary)',
                  color: 'var(--text-muted)',
                }}>
                  {selectedAgent.llm?.provider}/{selectedAgent.llm?.model}
                </span>
              </div>
            </div>

            {/* Tab content */}
            {activeTab === 'run' && <RunTab agentId={selectedAgent.id} />}
            {activeTab === 'chat' && <ChatTab agentId={selectedAgent.id} />}
            {activeTab === 'config' && <ConfigTab agentId={selectedAgent.id} />}
            {activeTab === 'multi-agent' && <MultiAgentPanel selectedAgentId={selectedAgent.id} />}
          </>
        ) : (
          <div className="empty-state">
            <div className="empty-icon">⚡</div>
            <p>
              <strong>Welcome to AgentForge</strong>
              <br /><br />
              {agents.length === 0
                ? 'No agents found. Start the backend (uvicorn api.server:app) to auto-load agent templates.'
                : 'Select an agent from the sidebar to begin.'}
            </p>
          </div>
        )}
      </div>

      {/* ── Right Panel ─────────────────────────── */}
      <RightPanel agentId={selectedAgentId} />
        </>
      )}

      {/* ── Workflows View ──────────────────────── */}
      {appView === 'workflows' && (
        <WorkflowsPage />
      )}
    </div>
  );
}
