import React from 'react';

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
  is_sub_agent?: boolean;
  parent_id?: string | null;
  sub_agents?: string[];
}

interface AgentSidebarProps {
  agents: Agent[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export default function AgentSidebar({ agents, selectedId, onSelect }: AgentSidebarProps) {
  // Separate root agents and sub-agents
  const rootAgents = agents.filter(a => !a.is_sub_agent);
  const subAgents = agents.filter(a => a.is_sub_agent);

  return (
    <div className="sidebar">
      <div className="sidebar-title">Agents</div>
      {agents.length === 0 && (
        <div style={{ padding: '16px', color: 'var(--text-muted)', fontSize: '12px' }}>
          No agents registered yet.
          <br />
          Use POST /agents to create one.
        </div>
      )}
      {rootAgents.map((agent) => (
        <div key={agent.id}>
          <div
            className={`agent-card ${selectedId === agent.id ? 'active' : ''}`}
            onClick={() => onSelect(agent.id)}
          >
            <div className={`agent-dot ${agent.status}`} />
            <div>
              <div className="agent-name">{agent.name}</div>
              <div className="agent-tools">
                {agent.tools.slice(0, 3).map((t) => (
                  <span key={t} className="tool-badge">{t}</span>
                ))}
                {agent.tools.length > 3 && (
                  <span className="tool-badge">+{agent.tools.length - 3}</span>
                )}
              </div>
              {/* v3: Show sub-agent count */}
              {(agent.sub_agents?.length ?? 0) > 0 && (
                <div style={{
                  marginTop: 4, fontSize: 10, color: 'var(--accent-purple)',
                }}>
                  🌐 {agent.sub_agents!.length} sub-agent{agent.sub_agents!.length > 1 ? 's' : ''}
                </div>
              )}
            </div>
          </div>
          {/* Render child sub-agents indented */}
          {subAgents
            .filter(sa => sa.parent_id === agent.id)
            .map(sa => (
              <div
                key={sa.id}
                className={`agent-card ${selectedId === sa.id ? 'active' : ''}`}
                onClick={() => onSelect(sa.id)}
                style={{ marginLeft: 16, borderLeft: '2px solid var(--accent-purple)', paddingLeft: 8 }}
              >
                <div className={`agent-dot ${sa.status}`} />
                <div>
                  <div className="agent-name" style={{ fontSize: 12 }}>
                    ↳ {sa.name}
                    <span style={{
                      marginLeft: 6, padding: '0 4px', borderRadius: 4,
                      fontSize: 9, background: 'var(--accent-purple)', color: '#fff',
                    }}>sub</span>
                  </div>
                </div>
              </div>
            ))}
        </div>
      ))}
    </div>
  );
}
