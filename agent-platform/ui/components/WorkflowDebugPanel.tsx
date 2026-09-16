import React, { useEffect, useRef } from 'react';
import { StreamEvent } from '../hooks/useAgentStream';

interface WorkflowDebugPanelProps {
  events: StreamEvent[];
  onClose: () => void;
}

function formatWorkflowEvent(event: StreamEvent): React.ReactNode {
  const data = event.data;
  const wfEvent = data.workflow_event;

  switch (wfEvent) {
    case 'workflow_start':
      return (
        <span className="debug-content">
          <span className="debug-icon">🚀</span>
          Starting workflow: <strong>{data.workflow_name}</strong> ({data.total_steps} steps)
        </span>
      );

    case 'workflow_step_start':
      return (
        <span className="debug-content">
          <span className="debug-icon">▸</span>
          <span className="debug-step-name">{data.step_name}</span>
          → calling <span className="debug-tool">{data.tool}</span>
          {data.attempt > 1 && <span className="debug-retry"> (attempt {data.attempt})</span>}
          {data.inputs && Object.keys(data.inputs).length > 0 && (
            <div className="debug-inputs">
              {Object.entries(data.inputs).map(([k, v]) => (
                <span key={k} className="debug-input-badge">{k}={String(v).slice(0, 50)}</span>
              ))}
            </div>
          )}
        </span>
      );

    case 'workflow_step_complete':
      return (
        <span className="debug-content">
          <span className="debug-icon success">✓</span>
          <span className="debug-step-name">{data.step_name}</span>
          completed in {data.duration_ms}ms
          {data.output && (
            <div className="debug-output">
              {data.output.length > 200 ? data.output.slice(0, 200) + '...' : data.output}
            </div>
          )}
        </span>
      );

    case 'workflow_step_error':
      return (
        <span className="debug-content">
          <span className="debug-icon error">✕</span>
          <span className="debug-step-name">{data.step_name}</span>
          {data.will_retry ? (
            <span className="debug-retry">
              ⚠ Retry {data.attempt}/{data.max_retries}: {data.error}
            </span>
          ) : (
            <span className="debug-error">FAILED: {data.error}</span>
          )}
        </span>
      );

    case 'workflow_step_skip':
      return (
        <span className="debug-content">
          <span className="debug-icon skip">⊘</span>
          <span className="debug-step-name">{data.step_name}</span>
          skipped (condition: {data.condition})
        </span>
      );

    case 'workflow_step_paused':
      return (
        <span className="debug-content">
          <span className="debug-icon paused">⏸</span>
          <span className="debug-step-name">{data.step_name}</span>
          <span className="debug-breakpoint">BREAKPOINT</span>
        </span>
      );

    case 'workflow_complete':
      return (
        <span className="debug-content">
          <span className="debug-icon success">✓</span>
          Workflow completed: {data.completed_steps}/{data.total_steps} steps
          in {data.duration_seconds?.toFixed(1)}s
        </span>
      );

    case 'workflow_error':
      return (
        <span className="debug-content">
          <span className="debug-icon error">✕</span>
          Workflow failed: {data.error}
          {data.failed_step && <span> (at step: {data.failed_step})</span>}
        </span>
      );

    default:
      // Regular agent events (plan, tool_call, etc.)
      if (event.type === 'plan') {
        return <span className="debug-content"><span className="debug-icon">💭</span> {data.thought}</span>;
      }
      return <span className="debug-content">{JSON.stringify(data)}</span>;
  }
}

function getEventClass(event: StreamEvent): string {
  const wfEvent = event.data.workflow_event;
  if (wfEvent === 'workflow_step_complete' || wfEvent === 'workflow_complete') return 'success';
  if (wfEvent === 'workflow_step_error' || wfEvent === 'workflow_error') return 'error';
  if (wfEvent === 'workflow_step_paused') return 'paused';
  if (wfEvent === 'workflow_step_skip') return 'skip';
  if (wfEvent === 'workflow_step_start' || wfEvent === 'workflow_start') return 'running';
  return '';
}

export default function WorkflowDebugPanel({ events, onClose }: WorkflowDebugPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [events]);

  return (
    <div className="workflow-debug-panel">
      <div className="debug-panel-header">
        <span className="debug-panel-title">🐛 Step Debugger</span>
        <span className="debug-event-count">{events.length} events</span>
        <button className="debug-close-btn" onClick={onClose}>✕</button>
      </div>
      <div className="debug-panel-body">
        {events.length === 0 ? (
          <div className="debug-empty">
            Run the workflow to see step-by-step execution logs here.
            <br /><br />
            Set breakpoints by clicking the ● button on any step node.
          </div>
        ) : (
          events.map((event, i) => (
            <div key={event.event_id || i} className={`debug-entry ${getEventClass(event)}`}>
              <span className="debug-timestamp">
                {new Date(event.timestamp * 1000).toLocaleTimeString()}
              </span>
              {formatWorkflowEvent(event)}
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
