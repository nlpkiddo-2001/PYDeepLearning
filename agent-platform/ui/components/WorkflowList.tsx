import React, { useState, useEffect } from 'react';
import { useWorkflow, Workflow, WorkflowTemplate } from '../hooks/useWorkflow';

interface WorkflowListProps {
  selectedId: string | null;
  onSelect: (id: string) => void;
  onCreateNew: () => void;
}

export default function WorkflowList({ selectedId, onSelect, onCreateNew }: WorkflowListProps) {
  const { workflows, fetchWorkflows, createWorkflow, deleteWorkflow, fetchTemplates } = useWorkflow();
  const [templates, setTemplates] = useState<WorkflowTemplate[]>([]);
  const [showTemplates, setShowTemplates] = useState(false);

  useEffect(() => {
    fetchWorkflows();
    const interval = setInterval(fetchWorkflows, 10000);
    return () => clearInterval(interval);
  }, [fetchWorkflows]);

  const handleLoadTemplates = async () => {
    const t = await fetchTemplates();
    setTemplates(t);
    setShowTemplates(true);
  };

  const handleCreateFromTemplate = async (template: WorkflowTemplate) => {
    const wf = await createWorkflow({
      name: template.name,
      description: template.description,
      steps: template.steps as any,
      edges: template.edges as any,
    });
    if (wf) {
      onSelect(wf.id);
      setShowTemplates(false);
    }
  };

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (confirm('Delete this workflow?')) {
      await deleteWorkflow(id);
    }
  };

  return (
    <div className="workflow-list">
      <div className="sidebar-title">Workflows</div>

      {/* Action buttons */}
      <div className="workflow-list-actions">
        <button className="wf-list-btn" onClick={onCreateNew}>+ New Workflow</button>
        <button className="wf-list-btn template" onClick={handleLoadTemplates}>📋 Templates</button>
      </div>

      {/* Workflow items */}
      {workflows.length === 0 && !showTemplates && (
        <div style={{ padding: '16px', color: 'var(--text-muted)', fontSize: '12px' }}>
          No workflows yet. Create one or load a template.
        </div>
      )}

      {workflows.map((wf) => (
        <div
          key={wf.id}
          className={`workflow-card ${selectedId === wf.id ? 'active' : ''}`}
          onClick={() => onSelect(wf.id)}
        >
          <div className="workflow-card-header">
            <span className="workflow-card-name">{wf.name}</span>
            <button
              className="workflow-card-delete"
              onClick={(e) => handleDelete(e, wf.id)}
              title="Delete"
            >✕</button>
          </div>
          <div className="workflow-card-meta">
            {wf.steps.length} steps · {wf.edges.length} connections
          </div>
          {wf.description && (
            <div className="workflow-card-desc">{wf.description}</div>
          )}
        </div>
      ))}

      {/* Template picker */}
      {showTemplates && (
        <div className="template-section">
          <div className="sidebar-title" style={{ marginTop: '12px' }}>
            Templates
            <button
              className="template-close"
              onClick={() => setShowTemplates(false)}
            >✕</button>
          </div>
          {templates.map((t) => (
            <div
              key={t.id}
              className="template-card"
              onClick={() => handleCreateFromTemplate(t)}
            >
              <div className="template-name">{t.name}</div>
              <div className="template-desc">{t.description}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
