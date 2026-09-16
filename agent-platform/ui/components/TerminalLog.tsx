import React, { useEffect, useRef } from 'react';
import { StreamEvent } from '../hooks/useAgentStream';

interface TerminalLogProps {
  events: StreamEvent[];
}

function formatEventContent(event: StreamEvent): React.ReactNode {
  const { type, data } = event;

  switch (type) {
    case 'plan':
      return <span className="log-content">{data.thought}</span>;

    case 'tool_call':
      return (
        <span className="log-content">
          <span className="tool-name">{data.tool}</span>
          <span className="tool-args">({JSON.stringify(data.args)})</span>
        </span>
      );

    case 'tool_result':
      const result = data.result?.length > 300 ? data.result.slice(0, 300) + '...' : data.result;
      return (
        <span className="log-content">
          <span className="tool-name">{data.tool}</span> → {result}
        </span>
      );

    case 'error':
      return (
        <span className="log-content" style={{ color: 'var(--accent-red)' }}>
          {data.message}
          {data.retry > 0 && ` (retry ${data.retry}/${data.max_retries})`}
        </span>
      );

    case 'retry':
      return (
        <span className="log-content" style={{ color: 'var(--accent-orange)' }}>
          ⚠ Retry {data.attempt}/{data.max_retries} for {data.tool}: {data.error}
        </span>
      );

    case 'done':
      return (
        <span className="log-content" style={{ color: 'var(--accent-green)' }}>
          ✓ {data.result}
        </span>
      );

    case 'info':
      return <span className="log-content">{data.message}</span>;

    default:
      return <span className="log-content">{JSON.stringify(data)}</span>;
  }
}

export default function TerminalLog({ events }: TerminalLogProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [events]);

  if (events.length === 0) {
    return (
      <div className="terminal-log">
        <div className="empty-state">
          <div className="empty-icon">▸</div>
          <p>
            Enter a goal above and click <strong>Run</strong> to start.
            <br />
            Execution steps will stream here in real time.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="terminal-log">
      {events.map((event, i) => (
        <div key={event.event_id || i} className="log-entry">
          <span className="log-step">#{event.step}</span>
          <span className={`log-type ${event.type}`}>{event.type}</span>
          {formatEventContent(event)}
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
