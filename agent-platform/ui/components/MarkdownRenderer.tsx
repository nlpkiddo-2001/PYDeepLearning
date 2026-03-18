import React, { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

/**
 * Reusable markdown renderer for chat bubbles, run results, and tool output.
 *
 * Supports:
 *  - **bold**, *italic*, ~~strikethrough~~
 *  - Headings, blockquotes, horizontal rules
 *  - Ordered / unordered / task lists
 *  - Fenced code blocks with language labels
 *  - Inline `code`
 *  - Tables (GFM)
 *  - Links and images
 *
 * Wraps content in a `.markdown-body` class so styles can be scoped.
 */

interface MarkdownRendererProps {
  content: string;
  /** Optional CSS class appended to the wrapper */
  className?: string;
  /** Truncate to this many characters (0 = no limit) */
  maxLength?: number;
}

export default function MarkdownRenderer({
  content,
  className = '',
  maxLength = 0,
}: MarkdownRendererProps) {
  const text = useMemo(() => {
    if (maxLength > 0 && content.length > maxLength) {
      return content.slice(0, maxLength) + '…';
    }
    return content;
  }, [content, maxLength]);

  return (
    <div className={`markdown-body ${className}`.trim()}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // Open links in a new tab
          a: ({ node, children, ...props }) => (
            <a {...props} target="_blank" rel="noopener noreferrer">
              {children}
            </a>
          ),
          // Fenced code blocks — lightweight syntax label (no heavy highlight lib)
          code: ({ node, className: codeClass, children, ...props }) => {
            const match = /language-(\w+)/.exec(codeClass || '');
            const isInline = !match && !String(children).includes('\n');

            if (isInline) {
              return (
                <code className="md-inline-code" {...props}>
                  {children}
                </code>
              );
            }

            return (
              <div className="md-code-block">
                {match && <span className="md-code-lang">{match[1]}</span>}
                <pre>
                  <code className={codeClass} {...props}>
                    {children}
                  </code>
                </pre>
              </div>
            );
          },
          // Tables get a scroll wrapper
          table: ({ node, children, ...props }) => (
            <div className="md-table-wrap">
              <table {...props}>{children}</table>
            </div>
          ),
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
