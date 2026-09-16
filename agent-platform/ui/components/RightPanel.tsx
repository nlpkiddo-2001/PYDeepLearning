import React, { useEffect, useState, useCallback } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface StatsData {
  runs_count: number;
  total_tokens: number;
  total_tool_calls: number;
  status: string;
  memory: {
    short_term: { messages: number; backend: string };
    long_term: { documents: number; backend: string };
  };
}

interface RightPanelProps {
  agentId: string | null;
}

export default function RightPanel({ agentId }: RightPanelProps) {
  const [stats, setStats] = useState<StatsData | null>(null);
  const [resetting, setResetting] = useState(false);

  const fetchStats = useCallback(async () => {
    if (!agentId) return;
    try {
      const resp = await fetch(`${API_BASE}/agents/${agentId}`);
      if (resp.ok) setStats(await resp.json());
    } catch {}
  }, [agentId]);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, [fetchStats]);

  const handleReset = useCallback(async () => {
    if (!agentId || resetting) return;
    if (!confirm('Reset all runs, tokens, and memory for this agent?')) return;
    setResetting(true);
    try {
      const resp = await fetch(`${API_BASE}/agents/${agentId}/reset`, { method: 'POST' });
      if (resp.ok) {
        await fetchStats();
      }
    } catch {}
    setResetting(false);
  }, [agentId, resetting, fetchStats]);

  if (!agentId || !stats) {
    return (
      <div className="right-panel">
        <div className="stats-title">Agent Info</div>
        <div style={{ color: 'var(--text-muted)', fontSize: '12px', marginTop: '20px' }}>
          Select an agent to view info.
        </div>
      </div>
    );
  }

  return (
    <div className="right-panel">
      <div className="stats-title">Agent Info</div>

      <div className="stat-card">
        <div className="stat-label">Status</div>
        <div className={`stat-value ${stats.status === 'running' ? 'running' : ''}`}>
          {stats.status?.toUpperCase() || 'IDLE'}
        </div>
      </div>

      <div className="stat-card">
        <div className="stat-label">Runs</div>
        <div className="stat-value">{stats.runs_count || 0}</div>
      </div>

      <div className="stat-card">
        <div className="stat-label">Tokens Used</div>
        <div className="stat-value">{(stats.total_tokens || 0).toLocaleString()}</div>
      </div>

      <div className="stat-card">
        <div className="stat-label">Tool Calls</div>
        <div className="stat-value">{stats.total_tool_calls || 0}</div>
      </div>

      {/* Long-term memory only */}
      {stats.memory?.long_term && stats.memory.long_term.documents > 0 && (
        <>
          <div className="stats-title" style={{ marginTop: '16px' }}>Long-term Memory</div>
          <div className="memory-layers">
            <div className="memory-layer">
              <span className="layer-name">Chroma</span>
              <span className="layer-status active">{stats.memory.long_term.documents} docs</span>
            </div>
          </div>
        </>
      )}

      {/* Reset button */}
      <button
        className="reset-stats-btn"
        onClick={handleReset}
        disabled={resetting}
        style={{
          marginTop: '20px',
          width: '100%',
          padding: '8px 12px',
          fontSize: '12px',
          background: 'transparent',
          border: '1px solid var(--accent-red)',
          color: 'var(--accent-red)',
          borderRadius: '6px',
          cursor: resetting ? 'wait' : 'pointer',
          opacity: resetting ? 0.5 : 1,
          transition: 'all 0.2s',
        }}
        onMouseEnter={(e) => {
          if (!resetting) {
            (e.target as HTMLButtonElement).style.background = 'var(--accent-red)';
            (e.target as HTMLButtonElement).style.color = '#fff';
          }
        }}
        onMouseLeave={(e) => {
          (e.target as HTMLButtonElement).style.background = 'transparent';
          (e.target as HTMLButtonElement).style.color = 'var(--accent-red)';
        }}
      >
        {resetting ? 'Resetting...' : '⟳ Reset Runs & Memory'}
      </button>
    </div>
  );
}
