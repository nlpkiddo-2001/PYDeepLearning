import React from 'react';
import { ChatSession } from '../hooks/useChatSessions';

interface ChatSidebarProps {
  sessions: ChatSession[];
  activeSessionId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  agentName?: string;
}

function formatDate(ts: number): string {
  const d = new Date(ts);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays === 0) return 'Today';
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return `${diffDays} days ago`;
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

export default function ChatSidebar({
  sessions,
  activeSessionId,
  onSelect,
  onNew,
  onDelete,
  agentName,
}: ChatSidebarProps) {
  // Group sessions by date category
  const today: ChatSession[] = [];
  const yesterday: ChatSession[] = [];
  const thisWeek: ChatSession[] = [];
  const older: ChatSession[] = [];

  const now = Date.now();
  for (const s of sessions) {
    const diffDays = Math.floor((now - s.updatedAt) / (1000 * 60 * 60 * 24));
    if (diffDays === 0) today.push(s);
    else if (diffDays === 1) yesterday.push(s);
    else if (diffDays < 7) thisWeek.push(s);
    else older.push(s);
  }

  const groups = [
    { label: 'Today', items: today },
    { label: 'Yesterday', items: yesterday },
    { label: 'This Week', items: thisWeek },
    { label: 'Older', items: older },
  ].filter((g) => g.items.length > 0);

  return (
    <div className="cs-sidebar">
      {/* Header */}
      <div className="cs-sidebar-header">
        <div className="cs-sidebar-title">
          <span className="cs-brand-icon">⚡</span>
          <span>AgentForge</span>
        </div>
        <button className="cs-new-chat-btn" onClick={onNew} title="New chat">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </button>
      </div>

      {/* Agent label */}
      {agentName && (
        <div className="cs-agent-label">
          <span className="cs-agent-dot" />
          <span>{agentName}</span>
        </div>
      )}

      {/* Session list */}
      <div className="cs-session-list">
        {sessions.length === 0 && (
          <div className="cs-empty">
            <p>No conversations yet.</p>
            <p>Start a new chat to begin.</p>
          </div>
        )}
        {groups.map((group) => (
          <div key={group.label} className="cs-group">
            <div className="cs-group-label">{group.label}</div>
            {group.items.map((session) => (
              <div
                key={session.id}
                className={`cs-session-card ${activeSessionId === session.id ? 'active' : ''}`}
                onClick={() => onSelect(session.id)}
              >
                <div className="cs-session-title">{session.title}</div>
                <div className="cs-session-meta">
                  {session.messages.length} message{session.messages.length !== 1 ? 's' : ''}
                </div>
                <button
                  className="cs-session-delete"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(session.id);
                  }}
                  title="Delete conversation"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
