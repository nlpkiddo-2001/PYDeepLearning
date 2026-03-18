import React, { useState, useMemo } from 'react';
import MarkdownRenderer from './MarkdownRenderer';

interface ThinkingBlockProps {
  content: string;
  /** Optional max length for the markdown renderer */
  maxLength?: number;
}

/**
 * Parses content for <think>...</think> tags.
 * Returns { thinking, answer } where:
 *  - thinking: text inside the <think> block (or everything before </think> if no opening tag)
 *  - answer: text after </think>
 *
 * Handles cases:
 *  - <think>...</think>rest     → thinking = "...", answer = "rest"
 *  - .....</think>rest          → thinking = ".....", answer = "rest" (missing opening tag)
 *  - no think tags at all       → thinking = null, answer = full content
 */
export function parseThinkContent(raw: string): { thinking: string | null; answer: string } {
  if (!raw) return { thinking: null, answer: '' };

  // Check for </think> tag (case-insensitive)
  const closeIdx = raw.search(/<\/think>/i);
  if (closeIdx === -1) {
    // No </think> tag — check if there's an unclosed <think> tag (still streaming)
    const openMatch = raw.match(/<think>/i);
    if (openMatch) {
      const openIdx = openMatch.index!;
      const thinkContent = raw.slice(openIdx + openMatch[0].length).trim();
      // Still inside thinking — no answer yet
      return { thinking: thinkContent || null, answer: '' };
    }
    return { thinking: null, answer: raw };
  }

  // Found </think> — extract thinking and answer
  const openMatch = raw.match(/<think>/i);
  let thinkStart = 0;
  if (openMatch && openMatch.index !== undefined) {
    thinkStart = openMatch.index + openMatch[0].length;
  }

  const thinkContent = raw.slice(thinkStart, closeIdx).trim();
  const closeTag = raw.match(/<\/think>/i)!;
  const answer = raw.slice(closeIdx + closeTag[0].length).trim();

  return {
    thinking: thinkContent || null,
    answer,
  };
}

/**
 * Renders content that may contain <think>...</think> tags.
 * The thinking portion is shown in a collapsible box.
 * The answer portion is rendered as markdown.
 */
export default function ThinkingBlock({ content, maxLength }: ThinkingBlockProps) {
  const { thinking, answer } = useMemo(() => parseThinkContent(content), [content]);
  const [expanded, setExpanded] = useState(false);

  if (!thinking) {
    // No thinking content — render normally
    return <MarkdownRenderer content={answer} maxLength={maxLength} />;
  }

  return (
    <div className="thinking-block-wrapper">
      {/* Collapsible thinking box */}
      <div className={`thinking-block ${expanded ? 'expanded' : ''}`}>
        <div
          className="thinking-block-header"
          onClick={() => setExpanded(!expanded)}
        >
          <span className="thinking-block-icon">💭</span>
          <span className="thinking-block-label">Thinking</span>
          <span className="thinking-block-toggle">{expanded ? '▾' : '▸'}</span>
        </div>
        {expanded && (
          <div className="thinking-block-body">
            <MarkdownRenderer content={thinking} />
          </div>
        )}
      </div>

      {/* Answer — only show if there's actual answer text */}
      {answer && (
        <div className="thinking-block-answer">
          <MarkdownRenderer content={answer} maxLength={maxLength} />
        </div>
      )}
    </div>
  );
}
