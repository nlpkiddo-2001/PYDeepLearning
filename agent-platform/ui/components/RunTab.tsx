import React, { useState, useCallback, useEffect, useRef } from 'react';
import ThinkingRunLog from './ThinkingRunLog';
import { useAgentStream } from '../hooks/useAgentStream';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface RunTabProps {
  agentId: string;
}

export default function RunTab({ agentId }: RunTabProps) {
  const [goal, setGoal] = useState('');
  const [activeGoal, setActiveGoal] = useState('');
  const [loading, setLoading] = useState(false);

  const { events, connected, isRunning, connect, clearEvents, pollEvents } = useAgentStream({
    agentId,
    autoConnect: true,
  });

  // ── Polling fallback: while a run is active, poll every 3s in case WS missed events ──
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const runActiveRef = useRef(false);

  useEffect(() => {
    // Start polling when we know a run was submitted but haven't received a "done" event
    if (runActiveRef.current && !pollRef.current) {
      pollRef.current = setInterval(() => {
        pollEvents();
      }, 3000);
    }

    // Stop polling once "done" or "error" event arrives
    const lastEvent = events[events.length - 1];
    if (lastEvent && (lastEvent.type === 'done' || lastEvent.type === 'error')) {
      runActiveRef.current = false;
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    }

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [events, pollEvents]);

  const handleRun = useCallback(async () => {
    if (!goal.trim() || loading) return;

    setLoading(true);
    setActiveGoal(goal.trim());
    clearEvents();

    // Ensure WebSocket is connected
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
        // Mark run as active to enable polling fallback
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
