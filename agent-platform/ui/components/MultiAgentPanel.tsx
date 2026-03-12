import React, { useState, useEffect, useCallback } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface HierarchyNode {
  agent_id: string;
  parent_id: string | null;
  children: string[];
  children_detail: HierarchyNode[];
  spawned_at: number;
  is_sub_agent: boolean;
  delegation_id: string | null;
  status: string;
}

interface HierarchyData {
  roots: HierarchyNode[];
  total_agents: number;
  sub_agents: number;
}

interface Delegation {
  id: string;
  parent_id: string;
  child_id: string;
  goal: string;
  status: string;
  result: string | null;
  created_at: number;
  completed_at: number | null;
}

interface AgentMessage {
  id: string;
  type: string;
  sender_id: string;
  receiver_id: string;
  content: string;
  timestamp: number;
}

interface MultiAgentStats {
  total_agents: number;
  root_agents: number;
  sub_agents: number;
  active_agents: number;
  message_router: {
    total_messages: number;
    total_delegations: number;
    active_delegations: number;
  };
  shared_memory: {
    total_entries: number;
    channel_count: number;
  };
}

interface MultiAgentPanelProps {
  selectedAgentId: string | null;
}

export default function MultiAgentPanel({ selectedAgentId }: MultiAgentPanelProps) {
  const [hierarchy, setHierarchy] = useState<HierarchyData | null>(null);
  const [stats, setStats] = useState<MultiAgentStats | null>(null);
  const [delegations, setDelegations] = useState<{ delegated_to_others: Delegation[]; received_from_parent: Delegation[] } | null>(null);
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [activeSubTab, setActiveSubTab] = useState<'hierarchy' | 'delegations' | 'messages' | 'shared-memory'>('hierarchy');
  const [spawnGoal, setSpawnGoal] = useState('');
  const [spawnName, setSpawnName] = useState('');
  const [spawning, setSpawning] = useState(false);
  const [channels, setChannels] = useState<any[]>([]);
  const [sendMsg, setSendMsg] = useState('');
  const [sendTo, setSendTo] = useState('');

  // Fetch hierarchy
  const fetchHierarchy = useCallback(async () => {
    try {
      const resp = await fetch(`${API_BASE}/multi-agent/hierarchy`);
      if (resp.ok) {
        const data = await resp.json();
        setHierarchy(data);
      }
    } catch {}
  }, []);

  // Fetch stats
  const fetchStats = useCallback(async () => {
    try {
      const resp = await fetch(`${API_BASE}/multi-agent/stats`);
      if (resp.ok) {
        setStats(await resp.json());
      }
    } catch {}
  }, []);

  // Fetch delegations for selected agent
  const fetchDelegations = useCallback(async () => {
    if (!selectedAgentId) return;
    try {
      const resp = await fetch(`${API_BASE}/multi-agent/agents/${selectedAgentId}/delegations`);
      if (resp.ok) {
        setDelegations(await resp.json());
      }
    } catch {}
  }, [selectedAgentId]);

  // Fetch messages for selected agent
  const fetchMessages = useCallback(async () => {
    if (!selectedAgentId) return;
    try {
      const resp = await fetch(`${API_BASE}/multi-agent/agents/${selectedAgentId}/messages?limit=30`);
      if (resp.ok) {
        const data = await resp.json();
        setMessages(data.messages || []);
      }
    } catch {}
  }, [selectedAgentId]);

  // Fetch shared memory channels
  const fetchChannels = useCallback(async () => {
    try {
      const resp = await fetch(`${API_BASE}/shared-memory/channels`);
      if (resp.ok) {
        const data = await resp.json();
        setChannels(data.channels || []);
      }
    } catch {}
  }, []);

  useEffect(() => {
    fetchHierarchy();
    fetchStats();
    const interval = setInterval(() => {
      fetchHierarchy();
      fetchStats();
    }, 5000);
    return () => clearInterval(interval);
  }, [fetchHierarchy, fetchStats]);

  useEffect(() => {
    if (selectedAgentId) {
      fetchDelegations();
      fetchMessages();
    }
  }, [selectedAgentId, fetchDelegations, fetchMessages]);

  useEffect(() => {
    if (activeSubTab === 'shared-memory') {
      fetchChannels();
      const interval = setInterval(fetchChannels, 5000);
      return () => clearInterval(interval);
    }
  }, [activeSubTab, fetchChannels]);

  // Spawn sub-agent
  const handleSpawn = async () => {
    if (!selectedAgentId || !spawnGoal.trim()) return;
    setSpawning(true);
    try {
      const resp = await fetch(`${API_BASE}/multi-agent/spawn`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          parent_agent_id: selectedAgentId,
          goal: spawnGoal.trim(),
          agent_name: spawnName.trim() || undefined,
        }),
      });
      if (resp.ok) {
        setSpawnGoal('');
        setSpawnName('');
        fetchHierarchy();
        fetchDelegations();
      } else {
        const err = await resp.json();
        alert(`Spawn failed: ${err.detail || 'Unknown error'}`);
      }
    } catch (err) {
      alert(`Network error: ${err}`);
    } finally {
      setSpawning(false);
    }
  };

  // Send message
  const handleSendMessage = async () => {
    if (!selectedAgentId || !sendTo.trim() || !sendMsg.trim()) return;
    try {
      await fetch(`${API_BASE}/multi-agent/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sender_id: selectedAgentId,
          receiver_id: sendTo.trim(),
          content: sendMsg.trim(),
        }),
      });
      setSendMsg('');
      fetchMessages();
    } catch {}
  };

  // Terminate sub-agent
  const handleTerminate = async (agentId: string) => {
    if (!confirm(`Terminate agent ${agentId}?`)) return;
    try {
      await fetch(`${API_BASE}/multi-agent/agents/${agentId}/terminate`, { method: 'POST' });
      fetchHierarchy();
    } catch {}
  };

  const statusColor = (status: string) => {
    switch (status) {
      case 'active': return 'var(--accent-green)';
      case 'completed': return 'var(--accent-blue)';
      case 'failed': return 'var(--accent-red)';
      case 'terminated': return 'var(--text-muted)';
      default: return 'var(--accent-yellow)';
    }
  };

  // Render hierarchy tree
  const renderTree = (node: HierarchyNode, depth: number = 0) => (
    <div key={node.agent_id} style={{ marginLeft: depth * 20, marginTop: 4 }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '4px 8px',
        borderRadius: 4,
        background: node.agent_id === selectedAgentId ? 'var(--bg-tertiary)' : 'transparent',
        fontSize: 12,
      }}>
        <span style={{
          width: 8, height: 8, borderRadius: '50%',
          background: statusColor(node.status),
          flexShrink: 0,
        }} />
        <span style={{ fontWeight: node.is_sub_agent ? 'normal' : 600 }}>
          {node.agent_id}
        </span>
        {node.is_sub_agent && (
          <span style={{
            padding: '1px 6px', borderRadius: 8, fontSize: 9,
            background: 'var(--accent-purple)', color: '#fff',
          }}>sub</span>
        )}
        <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>
          [{node.status}]
        </span>
        {node.is_sub_agent && (
          <button
            onClick={() => handleTerminate(node.agent_id)}
            style={{
              marginLeft: 'auto', padding: '1px 6px', fontSize: 10,
              background: 'var(--accent-red)', color: '#fff',
              border: 'none', borderRadius: 4, cursor: 'pointer',
            }}
          >
            ✕
          </button>
        )}
      </div>
      {node.children_detail?.map(child => renderTree(child, depth + 1))}
    </div>
  );

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Sub-tab bar */}
      <div style={{
        display: 'flex', gap: 0, borderBottom: '1px solid var(--border)',
        background: 'var(--bg-secondary)', flexShrink: 0,
      }}>
        {(['hierarchy', 'delegations', 'messages', 'shared-memory'] as const).map((tab) => (
          <button key={tab} onClick={() => setActiveSubTab(tab)} style={{
            padding: '8px 16px', fontSize: 12, border: 'none', cursor: 'pointer',
            background: activeSubTab === tab ? 'var(--bg-primary)' : 'transparent',
            color: activeSubTab === tab ? 'var(--text-primary)' : 'var(--text-muted)',
            borderBottom: activeSubTab === tab ? '2px solid var(--accent-blue)' : '2px solid transparent',
          }}>
            {tab === 'hierarchy' ? '🌳 Hierarchy' :
             tab === 'delegations' ? '📋 Delegations' :
             tab === 'messages' ? '💬 Messages' : '🗄️ Shared Memory'}
          </button>
        ))}
      </div>

      {/* Stats bar */}
      {stats && (
        <div style={{
          display: 'flex', gap: 16, padding: '8px 16px',
          fontSize: 11, color: 'var(--text-muted)',
          borderBottom: '1px solid var(--border)',
          background: 'var(--bg-secondary)',
        }}>
          <span>Agents: <strong>{stats.total_agents}</strong></span>
          <span>Sub-agents: <strong>{stats.sub_agents}</strong></span>
          <span>Messages: <strong>{stats.message_router?.total_messages || 0}</strong></span>
          <span>Delegations: <strong>{stats.message_router?.total_delegations || 0}</strong></span>
          <span>Shared Memory: <strong>{stats.shared_memory?.total_entries || 0}</strong> entries</span>
        </div>
      )}

      {/* Content */}
      <div style={{ flex: 1, overflow: 'auto', padding: 16 }}>
        {/* ── Hierarchy Tab ── */}
        {activeSubTab === 'hierarchy' && (
          <div>
            {/* Spawn sub-agent form */}
            {selectedAgentId && (
              <div style={{
                padding: 12, border: '1px solid var(--border)',
                borderRadius: 8, marginBottom: 16, background: 'var(--bg-secondary)',
              }}>
                <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>
                  Spawn Sub-Agent from: {selectedAgentId}
                </div>
                <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
                  <input
                    placeholder="Sub-agent name (optional)"
                    value={spawnName}
                    onChange={(e) => setSpawnName(e.target.value)}
                    style={{
                      flex: '0 0 200px', padding: '6px 10px', fontSize: 12,
                      border: '1px solid var(--border)', borderRadius: 4,
                      background: 'var(--bg-primary)', color: 'var(--text-primary)',
                    }}
                  />
                  <input
                    placeholder="Goal for the sub-agent..."
                    value={spawnGoal}
                    onChange={(e) => setSpawnGoal(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSpawn()}
                    style={{
                      flex: 1, padding: '6px 10px', fontSize: 12,
                      border: '1px solid var(--border)', borderRadius: 4,
                      background: 'var(--bg-primary)', color: 'var(--text-primary)',
                    }}
                  />
                  <button
                    onClick={handleSpawn}
                    disabled={!spawnGoal.trim() || spawning}
                    style={{
                      padding: '6px 16px', fontSize: 12,
                      background: 'var(--accent-green)', color: '#fff',
                      border: 'none', borderRadius: 4, cursor: 'pointer',
                      opacity: !spawnGoal.trim() || spawning ? 0.5 : 1,
                    }}
                  >
                    {spawning ? '...' : '+ Spawn'}
                  </button>
                </div>
              </div>
            )}

            {/* Hierarchy tree */}
            <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>
              Agent Hierarchy
            </div>
            {hierarchy?.roots?.length ? (
              hierarchy.roots.map(root => renderTree(root))
            ) : (
              <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>
                No agents in hierarchy. Register agents to see the tree.
              </div>
            )}
          </div>
        )}

        {/* ── Delegations Tab ── */}
        {activeSubTab === 'delegations' && (
          <div>
            {!selectedAgentId ? (
              <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>
                Select an agent to view delegations.
              </div>
            ) : (
              <>
                <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>
                  Tasks Delegated to Others
                </div>
                {delegations?.delegated_to_others?.length ? (
                  delegations.delegated_to_others.map((d) => (
                    <div key={d.id} style={{
                      padding: '8px 12px', borderRadius: 6, marginBottom: 8,
                      border: '1px solid var(--border)', background: 'var(--bg-secondary)',
                      fontSize: 12,
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                        <span style={{ fontWeight: 600 }}>→ {d.child_id}</span>
                        <span style={{
                          padding: '1px 8px', borderRadius: 8, fontSize: 10,
                          background: d.status === 'completed' ? 'var(--accent-green)' :
                                     d.status === 'failed' ? 'var(--accent-red)' : 'var(--accent-yellow)',
                          color: '#fff',
                        }}>{d.status}</span>
                      </div>
                      <div style={{ color: 'var(--text-secondary)' }}>{d.goal}</div>
                      {d.result && (
                        <div style={{ marginTop: 4, color: 'var(--accent-green)', fontSize: 11 }}>
                          Result: {d.result.slice(0, 200)}{d.result.length > 200 ? '...' : ''}
                        </div>
                      )}
                    </div>
                  ))
                ) : (
                  <div style={{ color: 'var(--text-muted)', fontSize: 12, marginBottom: 16 }}>
                    No tasks delegated to other agents.
                  </div>
                )}

                <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, marginTop: 16 }}>
                  Tasks Received from Parent
                </div>
                {delegations?.received_from_parent?.length ? (
                  delegations.received_from_parent.map((d) => (
                    <div key={d.id} style={{
                      padding: '8px 12px', borderRadius: 6, marginBottom: 8,
                      border: '1px solid var(--border)', background: 'var(--bg-secondary)',
                      fontSize: 12,
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                        <span style={{ fontWeight: 600 }}>← {d.parent_id}</span>
                        <span style={{
                          padding: '1px 8px', borderRadius: 8, fontSize: 10,
                          background: d.status === 'completed' ? 'var(--accent-green)' :
                                     d.status === 'failed' ? 'var(--accent-red)' : 'var(--accent-yellow)',
                          color: '#fff',
                        }}>{d.status}</span>
                      </div>
                      <div style={{ color: 'var(--text-secondary)' }}>{d.goal}</div>
                    </div>
                  ))
                ) : (
                  <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>
                    No tasks received from a parent agent.
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* ── Messages Tab ── */}
        {activeSubTab === 'messages' && (
          <div>
            {!selectedAgentId ? (
              <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>
                Select an agent to view messages.
              </div>
            ) : (
              <>
                {/* Send message form */}
                <div style={{
                  display: 'flex', gap: 8, marginBottom: 16,
                  padding: 12, border: '1px solid var(--border)',
                  borderRadius: 8, background: 'var(--bg-secondary)',
                }}>
                  <input
                    placeholder="Receiver agent ID"
                    value={sendTo}
                    onChange={(e) => setSendTo(e.target.value)}
                    style={{
                      width: 160, padding: '6px 10px', fontSize: 12,
                      border: '1px solid var(--border)', borderRadius: 4,
                      background: 'var(--bg-primary)', color: 'var(--text-primary)',
                    }}
                  />
                  <input
                    placeholder="Message content..."
                    value={sendMsg}
                    onChange={(e) => setSendMsg(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                    style={{
                      flex: 1, padding: '6px 10px', fontSize: 12,
                      border: '1px solid var(--border)', borderRadius: 4,
                      background: 'var(--bg-primary)', color: 'var(--text-primary)',
                    }}
                  />
                  <button
                    onClick={handleSendMessage}
                    disabled={!sendTo.trim() || !sendMsg.trim()}
                    style={{
                      padding: '6px 16px', fontSize: 12,
                      background: 'var(--accent-blue)', color: '#fff',
                      border: 'none', borderRadius: 4, cursor: 'pointer',
                      opacity: !sendTo.trim() || !sendMsg.trim() ? 0.5 : 1,
                    }}
                  >
                    Send
                  </button>
                </div>

                {/* Message list */}
                <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>
                  Message History
                </div>
                {messages.length ? (
                  messages.map((m) => (
                    <div key={m.id} style={{
                      padding: '6px 12px', borderRadius: 6, marginBottom: 6,
                      border: '1px solid var(--border)', fontSize: 12,
                      background: m.sender_id === selectedAgentId ? 'var(--bg-tertiary)' : 'var(--bg-secondary)',
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                        <span>
                          <strong>{m.sender_id}</strong>
                          <span style={{ color: 'var(--text-muted)' }}> → </span>
                          <strong>{m.receiver_id || 'ALL'}</strong>
                        </span>
                        <span style={{
                          padding: '0 6px', borderRadius: 6, fontSize: 9,
                          background: 'var(--bg-tertiary)', color: 'var(--text-muted)',
                        }}>{m.type}</span>
                      </div>
                      <div style={{ color: 'var(--text-secondary)' }}>
                        {m.content.slice(0, 300)}
                      </div>
                    </div>
                  ))
                ) : (
                  <div style={{ color: 'var(--text-muted)' }}>No messages yet.</div>
                )}
              </>
            )}
          </div>
        )}

        {/* ── Shared Memory Tab ── */}
        {activeSubTab === 'shared-memory' && (
          <div>
            <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>
              Shared Memory Channels
            </div>
            {channels.length ? (
              channels.map((ch: any) => (
                <div key={ch.name} style={{
                  padding: '8px 12px', borderRadius: 6, marginBottom: 8,
                  border: '1px solid var(--border)', background: 'var(--bg-secondary)',
                  fontSize: 12,
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span style={{ fontWeight: 600 }}>📁 {ch.name}</span>
                    <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>
                      {ch.entry_count} entries
                    </span>
                  </div>
                  <div style={{ color: 'var(--text-secondary)', fontSize: 11 }}>
                    Writers: {ch.writers?.join(', ') || 'none'}
                  </div>
                  <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>
                    Last updated: {ch.last_updated ? new Date(ch.last_updated * 1000).toLocaleString() : 'never'}
                  </div>
                </div>
              ))
            ) : (
              <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>
                No shared memory channels. Agents will create channels when they write shared data.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
