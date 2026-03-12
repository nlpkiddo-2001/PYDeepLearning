import React, { useEffect, useState } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface StatsData {
  runs_count: number;
  total_tokens: number;
  total_tool_calls: number;
  status: string;
  is_sub_agent?: boolean;
  parent_id?: string;
  sub_agents?: string[];
  can_spawn?: boolean;
  memory: {
    short_term: { messages: number; backend: string };
    long_term: { documents: number; backend: string };
  };
}

interface MultiAgentStats {
  total_agents: number;
  sub_agents: number;
  active_delegations: number;
  total_messages: number;
  shared_memory_channels: number;
  shared_memory_entries: number;
}

interface RightPanelProps {
  agentId: string | null;
}

export default function RightPanel({ agentId }: RightPanelProps) {
  const [stats, setStats] = useState<StatsData | null>(null);
  const [maStats, setMaStats] = useState<MultiAgentStats | null>(null);

  useEffect(() => {
    if (!agentId) return;

    const fetchStats = async () => {
      try {
        const resp = await fetch(`${API_BASE}/agents/${agentId}`);
        if (resp.ok) {
          const data = await resp.json();
          setStats(data);
        }
      } catch {}
    };

    const fetchMultiAgentStats = async () => {
      try {
        const resp = await fetch(`${API_BASE}/multi-agent/stats`);
        if (resp.ok) {
          const data = await resp.json();
          setMaStats(data);
        }
      } catch {}
    };

    fetchStats();
    fetchMultiAgentStats();
    const interval = setInterval(() => { fetchStats(); fetchMultiAgentStats(); }, 5000);
    return () => clearInterval(interval);
  }, [agentId]);

  if (!agentId || !stats) {
    return (
      <div className="right-panel">
        <div className="stats-title">Agent Stats</div>
        <div style={{ color: 'var(--text-muted)', fontSize: '12px', marginTop: '20px' }}>
          Select an agent to view stats.
        </div>
      </div>
    );
  }

  return (
    <div className="right-panel">
      <div className="stats-title">Live Stats</div>

      <div className="stat-card">
        <div className="stat-label">Status</div>
        <div className={`stat-value ${stats.status === 'running' ? 'running' : ''}`}>
          {stats.status?.toUpperCase() || 'IDLE'}
        </div>
      </div>

      <div className="stat-card">
        <div className="stat-label">Runs This Session</div>
        <div className="stat-value">{stats.runs_count || 0}</div>
      </div>

      <div className="stat-card">
        <div className="stat-label">Total Tokens Used</div>
        <div className="stat-value">{(stats.total_tokens || 0).toLocaleString()}</div>
      </div>

      <div className="stat-card">
        <div className="stat-label">Tool Calls Made</div>
        <div className="stat-value">{stats.total_tool_calls || 0}</div>
      </div>

      <div className="stats-title" style={{ marginTop: '20px' }}>Memory Layers</div>

      <div className="memory-layers">
        <div className="memory-layer">
          <span className="layer-name">
            SQLite (short-term)
          </span>
          <span className={`layer-status ${stats.memory?.short_term ? 'active' : 'inactive'}`}>
            {stats.memory?.short_term
              ? `${stats.memory.short_term.messages} msgs`
              : 'inactive'}
          </span>
        </div>
        <div className="memory-layer">
          <span className="layer-name">
            Chroma (long-term)
          </span>
          <span className={`layer-status ${stats.memory?.long_term ? 'active' : 'inactive'}`}>
            {stats.memory?.long_term
              ? `${stats.memory.long_term.documents} docs`
              : 'inactive'}
          </span>
        </div>
      </div>

      {/* Multi-Agent Section */}
      {(stats.is_sub_agent || (stats.sub_agents && stats.sub_agents.length > 0) || maStats) && (
        <>
          <div className="stats-title" style={{ marginTop: '20px' }}>Multi-Agent</div>

          {stats.is_sub_agent && (
            <div className="stat-card">
              <div className="stat-label">Role</div>
              <div className="stat-value" style={{ color: '#a78bfa', fontSize: '14px' }}>
                Sub-Agent of {stats.parent_id || 'unknown'}
              </div>
            </div>
          )}

          {stats.sub_agents && stats.sub_agents.length > 0 && (
            <div className="stat-card">
              <div className="stat-label">Active Sub-Agents</div>
              <div className="stat-value">{stats.sub_agents.length}</div>
            </div>
          )}

          {maStats && (
            <div className="memory-layers">
              <div className="memory-layer">
                <span className="layer-name">Total Agents</span>
                <span className="layer-status active">{maStats.total_agents}</span>
              </div>
              <div className="memory-layer">
                <span className="layer-name">Sub-Agents</span>
                <span className="layer-status active">{maStats.sub_agents}</span>
              </div>
              <div className="memory-layer">
                <span className="layer-name">Active Delegations</span>
                <span className={`layer-status ${maStats.active_delegations > 0 ? 'active' : 'inactive'}`}>
                  {maStats.active_delegations}
                </span>
              </div>
              <div className="memory-layer">
                <span className="layer-name">Messages Sent</span>
                <span className="layer-status active">{maStats.total_messages}</span>
              </div>
              <div className="memory-layer">
                <span className="layer-name">Shared Memory</span>
                <span className={`layer-status ${maStats.shared_memory_entries > 0 ? 'active' : 'inactive'}`}>
                  {maStats.shared_memory_channels} ch / {maStats.shared_memory_entries} entries
                </span>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
