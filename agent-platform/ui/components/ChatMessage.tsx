import React, { useState } from 'react';
import MarkdownRenderer from './MarkdownRenderer';
import ThinkingBlock from './ThinkingBlock';
import { ChatMessage as ChatMessageType, ToolStep } from '../hooks/useChatSessions';

interface ChatMessageProps {
  message: ChatMessageType;
}

function ToolStepCard({ step }: { step: ToolStep }) {
  const [expanded, setExpanded] = useState(false);

  let icon = '💭';
  let label = 'Thinking';
  let statusClass = 'thinking';

  if (step.type === 'tool_call' || step.toolName) {
    icon = '🔧';
    label = step.toolName || 'Tool';
    statusClass = 'tool';
  }
  if (step.type === 'tool_result') {
    icon = '✓';
    label = 'Result';
    statusClass = 'success';
  }
  if (step.type === 'error') {
    icon = '✕';
    label = 'Error';
    statusClass = 'error';
  }
  if (step.type === 'retry') {
    icon = '⟳';
    label = 'Retry';
    statusClass = 'retry';
  }

  return (
    <div className={`chat-tool-step ${statusClass}`}>
      <div className="chat-tool-step-header" onClick={() => setExpanded(!expanded)}>
        <span className={`chat-tool-icon ${statusClass}`}>{icon}</span>
        <span className="chat-tool-label">{label}</span>
        {step.toolName && step.type === 'tool_call' && (
          <span className="chat-tool-name">{step.toolName}</span>
        )}
        <span className="chat-tool-expand">{expanded ? '▾' : '▸'}</span>
      </div>
      {expanded && (
        <div className="chat-tool-step-body">
          {step.thought && (
            <div className="chat-tool-section">
              <div className="chat-tool-section-label">Reasoning</div>
              <div className="chat-tool-section-content thought">
                <MarkdownRenderer content={step.thought} />
              </div>
            </div>
          )}
          {step.toolName && step.toolArgs && (
            <div className="chat-tool-section">
              <div className="chat-tool-section-label">Tool Call</div>
              <div className="chat-tool-section-content tool-call">
                <span className="chat-tool-fn-name">{step.toolName}</span>
                <span className="chat-tool-fn-args">
                  ({JSON.stringify(step.toolArgs, null, 2)})
                </span>
              </div>
            </div>
          )}
          {step.toolResult && (
            <div className="chat-tool-section">
              <div className="chat-tool-section-label">Result</div>
              <div className={`chat-tool-section-content result ${step.toolResult.startsWith('ERROR') ? 'error-text' : ''}`}>
                <MarkdownRenderer content={step.toolResult} maxLength={800} />
              </div>
            </div>
          )}
          {step.error && (
            <div className="chat-tool-section">
              <div className="chat-tool-section-content error-msg">
                {step.error}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';
  const hasToolSteps = message.toolSteps && message.toolSteps.length > 0;
  const [showTools, setShowTools] = useState(true);

  return (
    <div className={`cm-wrapper ${isUser ? 'cm-user' : 'cm-assistant'}`}>
      <div className="cm-row">
        {!isUser && (
          <div className="cm-avatar cm-avatar-assistant">
            <span>⚡</span>
          </div>
        )}
        <div className={`cm-content ${isUser ? 'cm-content-user' : 'cm-content-assistant'}`}>
          {/* Role label */}
          <div className="cm-role-label">
            {isUser ? 'You' : 'AgentForge'}
          </div>

          {/* Tool execution (for assistant messages) */}
          {!isUser && hasToolSteps && (
            <div className="cm-tool-execution">
              <div
                className="cm-tool-execution-header"
                onClick={() => setShowTools(!showTools)}
              >
                <span className="cm-tool-execution-icon">
                  {message.isStreaming ? (
                    <span className="cm-thinking-spinner" />
                  ) : (
                    '🔧'
                  )}
                </span>
                <span className="cm-tool-execution-label">
                  {message.isStreaming
                    ? `Working... (${message.toolSteps!.length} steps)`
                    : `Used ${message.toolSteps!.filter(s => s.toolName).length} tools in ${message.toolSteps!.length} steps`}
                </span>
                <span className="cm-tool-toggle">{showTools ? '▾' : '▸'}</span>
              </div>
              {showTools && (
                <div className="cm-tool-steps">
                  {message.toolSteps!.map((step, i) => (
                    <ToolStepCard key={i} step={step} />
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Message content */}
          {message.content && (
            <div className={`cm-bubble ${isUser ? 'cm-bubble-user' : 'cm-bubble-assistant'}`}>
              {isUser ? (
                <span>{message.content}</span>
              ) : (
                <ThinkingBlock content={message.content} />
              )}
            </div>
          )}

          {/* Streaming indicator */}
          {message.isStreaming && !message.content && (
            <div className="cm-bubble cm-bubble-assistant cm-streaming">
              <span className="cm-thinking-spinner" />
              <span style={{ marginLeft: 8, color: 'var(--text-muted)' }}>Thinking...</span>
            </div>
          )}
        </div>
        {isUser && (
          <div className="cm-avatar cm-avatar-user">
            <span>You</span>
          </div>
        )}
      </div>
    </div>
  );
}
