import React, { useState, useEffect, useCallback } from 'react';
import WorkflowList from '../components/WorkflowList';
import WorkflowBuilder from '../components/WorkflowBuilder';
import { useWorkflow, Workflow } from '../hooks/useWorkflow';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function WorkflowsPage() {
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string | null>(null);
  const [selectedWorkflow, setSelectedWorkflow] = useState<Workflow | null>(null);
  const { createWorkflow, updateWorkflow, fetchWorkflows, workflows } = useWorkflow();
  const [error, setError] = useState<string | null>(null);

  // Fetch workflows on mount
  useEffect(() => {
    fetchWorkflows();
  }, [fetchWorkflows]);

  // Fetch selected workflow details
  useEffect(() => {
    if (!selectedWorkflowId) {
      setSelectedWorkflow(null);
      return;
    }

    (async () => {
      try {
        const resp = await fetch(`${API_BASE}/workflows/${selectedWorkflowId}`);
        if (resp.ok) {
          const data = await resp.json();
          setSelectedWorkflow(data);
        }
      } catch (err) {
        setError('Failed to load workflow');
      }
    })();
  }, [selectedWorkflowId, workflows]);

  const handleCreateNew = useCallback(async () => {
    const wf = await createWorkflow({
      name: 'Untitled Workflow',
      description: '',
      steps: [],
      edges: [],
    });
    if (wf) {
      setSelectedWorkflowId(wf.id);
    }
  }, [createWorkflow]);

  const handleSaveWorkflow = useCallback(async (data: Partial<Workflow>) => {
    if (selectedWorkflowId) {
      await updateWorkflow(selectedWorkflowId, data);
      // Refresh
      try {
        const resp = await fetch(`${API_BASE}/workflows/${selectedWorkflowId}`);
        if (resp.ok) {
          setSelectedWorkflow(await resp.json());
        }
      } catch {}
    }
  }, [selectedWorkflowId, updateWorkflow]);

  return (
    <div className="workflows-page">
      {/* Left panel: workflow list */}
      <div className="workflows-sidebar">
        <WorkflowList
          selectedId={selectedWorkflowId}
          onSelect={setSelectedWorkflowId}
          onCreateNew={handleCreateNew}
        />
      </div>

      {/* Main area: workflow builder */}
      <div className="workflows-main">
        {selectedWorkflow ? (
          <WorkflowBuilder
            workflow={selectedWorkflow}
            onSave={handleSaveWorkflow}
          />
        ) : (
          <div className="empty-state">
            <div className="empty-icon">🔧</div>
            <p>
              <strong>Workflow Builder</strong>
              <br /><br />
              Create a new workflow or select one from the sidebar.
              <br />
              Drag &amp; drop steps, connect them, and run with live debugging.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
