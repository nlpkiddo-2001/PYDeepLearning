import React from 'react';
import { useRouter } from 'next/router';

interface NavHeaderProps {
  agentCount?: number;
}

export default function NavHeader({ agentCount = 0 }: NavHeaderProps) {
  const router = useRouter();
  const current = router.pathname;

  const links = [
    { href: '/', label: '🤖 Agents', key: 'agents' },
    { href: '/chat', label: '💬 Chat', key: 'chat' },
    { href: '/workflows', label: '🔧 Workflows', key: 'workflows' },
  ];

  return (
    <header className="nav-header">
      <div className="nav-header-left">
        <h1 className="nav-brand" onClick={() => router.push('/')}>
          ⚡ AgentForge
        </h1>
        <span className="nav-subtitle">Agentic AI Platform</span>
        <nav className="nav-links">
          {links.map((link) => (
            <button
              key={link.key}
              className={`nav-link ${current === link.href ? 'active' : ''}`}
              onClick={() => router.push(link.href)}
            >
              {link.label}
            </button>
          ))}
        </nav>
      </div>
      <div className="nav-header-right">
        <span className="nav-agent-count">
          {agentCount} agent{agentCount !== 1 ? 's' : ''} registered
        </span>
      </div>
    </header>
  );
}
