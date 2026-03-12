import React, { useState, useEffect, useRef } from 'react';
import { StreamEvent } from '../hooks/useAgentStream';

interface ThinkingRunLogProps {
  events: StreamEvent[];
  goal?: string;
}

/** Group consecutive events into numbered "steps" keyed by step number. */
interface StepGroup {
  step: number;
  events: StreamEvent[];
  type: 'plan' | 'tool_call' | 'tool_result' | 'error' | 'retry' | 'info' | 'done' | 'other';
  thought?: string;
  toolName?: string;
  toolArgs?: Record<string, any>;
  toolResult?: string;
  error?: string;
  isRetry?: boolean;
  isDone?: boolean;
  finalResult?: string;
}

function groupEvents(events: StreamEvent[]): StepGroup[] {
  const groups: StepGroup[] = [];
  let current: StepGroup | null = null;

  for (const ev of events) {
    if (ev.type === 'info' && ev.step === 0) {
      // Starting info — skip from grouping, rendered as header
      continue;
    }

    if (ev.type === 'done') {
      groups.push({
        step: ev.step,
        events: [ev],
        type: 'done',
        isDone: true,
        finalResult: ev.data.result,
      });
      continue;
    }

    if (ev.type === 'plan') {
      // Start a new step group
      if (current) groups.push(current);
      current = {
        step: ev.step,
        events: [ev],
        type: 'plan',
        thought: ev.data.thought,
      };
    } else if (current && ev.step === current.step) {
      current.events.push(ev);
      if (ev.type === 'tool_call') {
        current.toolName = ev.data.tool;
        current.toolArgs = ev.data.args;
      } else if (ev.type === 'tool_result') {
        current.toolResult = ev.data.result;
      } else if (ev.type === 'retry') {
        current.isRetry = true;
        current.error = ev.data.error;
      } else if (ev.type === 'error') {
        current.error = ev.data.message;
      }
    } else {
      // New step number
      if (current) groups.push(current);
      current = {
        step: ev.step,
        events: [ev],
        type: ev.type as any,
      };
      if (ev.type === 'tool_call') {
        current.toolName = ev.data.tool;
        current.toolArgs = ev.data.args;
      } else if (ev.type === 'tool_result') {
        current.toolResult = ev.data.result;
      }
    }
  }

  if (current) groups.push(current);
  return groups;
}

function StepCard({ group, isLatest }: { group: StepGroup; isLatest: boolean }) {
  const [expanded, setExpanded] = useState(false);

  // Auto-expand the latest step
  useEffect(() => {
    if (isLatest && !group.isDone) setExpanded(true);
    else if (!isLatest) setExpanded(false);
  }, [isLatest, group.isDone]);

  if (group.isDone) return null; // Rendered separately as final result

  const hasResult = group.toolResult !== undefined;
  const isError = group.events.some(e => e.type === 'error' && !e.data.retry);
  const hasRetry = group.isRetry;

  // Status indicator
  let statusIcon = '💭';
  let statusClass = 'thinking';
  if (group.toolName) {
    statusIcon = '🔧';
    statusClass = 'tool';
  }
  if (hasResult) {
    statusIcon = '✓';
    statusClass = 'success';
  }
  if (isError) {
    statusIcon = '✕';
    statusClass = 'error';
  }
  if (hasRetry && !hasResult) {
    statusIcon = '⟳';
    statusClass = 'retrying';
  }
  if (isLatest && !hasResult && !isError) {
    statusIcon = '';
    statusClass = 'running';
  }

  return (
    <div className={`thinking-step ${statusClass} ${expanded ? 'expanded' : ''}`}>
      <div className="thinking-step-header" onClick={() => setExpanded(!expanded)}>
        <div className="thinking-step-left">
          <span className={`thinking-status-icon ${statusClass}`}>
            {statusClass === 'running' ? (
              <span className="thinking-spinner" />
            ) : statusIcon}
          </span>
          <span className="thinking-step-num">Step {group.step}</span>
          {group.toolName && (
            <span className="thinking-tool-badge">{group.toolName}</span>
          )}
          {!group.toolName && group.thought && (
            <span className="thinking-thought-preview">
              {group.thought.length > 80 ? group.thought.slice(0, 80) + '…' : group.thought}
            </span>
          )}
          {hasRetry && <span className="thinking-retry-badge">retry</span>}
          {isError && <span className="thinking-error-badge">failed</span>}
        </div>
        <span className="thinking-expand-icon">{expanded ? '▾' : '▸'}</span>
      </div>

      {expanded && (
        <div className="thinking-step-body">
          {/* Thought */}
          {group.thought && (
            <div className="thinking-section">
              <div className="thinking-section-label">Reasoning</div>
              <div className="thinking-section-content thought">{group.thought}</div>
            </div>
          )}

          {/* Tool call */}
          {group.toolName && (
            <div className="thinking-section">
              <div className="thinking-section-label">Tool Call</div>
              <div className="thinking-section-content tool-call">
                <span className="thinking-fn-name">{group.toolName}</span>
                <span className="thinking-fn-args">
                  ({JSON.stringify(group.toolArgs || {}, null, 0)})
                </span>
              </div>
            </div>
          )}

          {/* Retries */}
          {group.events.filter(e => e.type === 'retry').map((ev, i) => (
            <div key={i} className="thinking-section">
              <div className="thinking-section-content retry">
                ⚠ Retry {ev.data.attempt}/{ev.data.max_retries}: {ev.data.error}
              </div>
            </div>
          ))}

          {/* Tool result */}
          {group.toolResult !== undefined && (
            <div className="thinking-section">
              <div className="thinking-section-label">Result</div>
              <div className={`thinking-section-content result ${group.toolResult.startsWith('ERROR') ? 'error-text' : ''}`}>
                {group.toolResult.length > 500 ? group.toolResult.slice(0, 500) + '…' : group.toolResult}
              </div>
            </div>
          )}

          {/* Error (non-retry) */}
          {group.events.filter(e => e.type === 'error').map((ev, i) => (
            <div key={i} className="thinking-section">
              <div className="thinking-section-content error-msg">
                {ev.data.message}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function ThinkingRunLog({ events, goal }: ThinkingRunLogProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const groups = groupEvents(events);
  const doneGroup = groups.find(g => g.isDone);
  const stepGroups = groups.filter(g => !g.isDone);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [events]);

  if (events.length === 0) {
    return (
      <div className="thinking-log">
        <div className="empty-state">
          <div className="empty-icon">▸</div>
          <p>
            Enter a goal above and click <strong>Run</strong> to start.
            <br />
            The agent will reason, use tools, and stream results in real time.
          </p>
        </div>
      </div>
    );
  }

  const isRunning = !doneGroup && events.length > 0;

  return (
    <div className="thinking-log">
      {/* Goal banner */}
      {goal && (
        <div className="thinking-goal">
          <span className="thinking-goal-label">Goal</span>
          <span className="thinking-goal-text">{goal}</span>
        </div>
      )}

      {/* Thinking section (collapsible overview) */}
      <div className="thinking-container">
        <div className="thinking-header">
          {isRunning ? (
            <>
              <span className="thinking-spinner" />
              <span>Thinking…</span>
            </>
          ) : doneGroup ? (
            <>
              <span className="thinking-done-icon">✓</span>
              <span>Completed in {stepGroups.length} steps</span>
            </>
          ) : (
            <span>Processing…</span>
          )}
        </div>

        <div className="thinking-steps">
          {stepGroups.map((group, i) => (
            <StepCard
              key={`${group.step}-${i}`}
              group={group}
              isLatest={i === stepGroups.length - 1 && !doneGroup}
            />
          ))}
        </div>
      </div>

      {/* Final result — chatbot-style bubble */}
      {doneGroup && doneGroup.finalResult && (
        <div className="thinking-final-result">
          <div className="thinking-result-avatar">🤖</div>
          <div className="thinking-result-bubble">
            {doneGroup.finalResult.split('\n').map((line, j) => (
              <React.Fragment key={j}>
                {line}
                {j < doneGroup.finalResult!.split('\n').length - 1 && <br />}
              </React.Fragment>
            ))}
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
