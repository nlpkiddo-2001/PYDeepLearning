import { useState, useCallback } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ── Types ─────────────────────────────────────────────

export interface WorkflowStep {
  id: string;
  name: string;
  tool: string;
  inputs: Record<string, any>;
  input_mapping: Record<string, string>;
  condition?: string;
  timeout: number;
  max_retries: number;
  depends_on: string[];
  position: { x: number; y: number };
  status?: string;
  output?: string;
  error?: string;
  duration_ms?: number;
}

export interface WorkflowEdge {
  id: string;
  source: string;
  target: string;
  label: string;
}

export interface Workflow {
  id: string;
  name: string;
  description: string;
  steps: WorkflowStep[];
  edges: WorkflowEdge[];
  created_at: number;
  updated_at: number;
}

export interface WorkflowRun {
  run_id: string;
  workflow_id: string;
  workflow_name: string;
  status: string;
  steps: WorkflowStep[];
  step_outputs: Record<string, string>;
  current_step_id?: string;
  started_at: number;
  ended_at?: number;
  error?: string;
  duration_seconds: number;
}

export interface WorkflowTemplate {
  id: string;
  name: string;
  description: string;
  steps: Partial<WorkflowStep>[];
  edges: Partial<WorkflowEdge>[];
}

// ── Hook ──────────────────────────────────────────────

export function useWorkflow() {
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchWorkflows = useCallback(async () => {
    try {
      const resp = await fetch(`${API_BASE}/workflows`);
      if (resp.ok) {
        const data = await resp.json();
        setWorkflows(data.workflows || []);
        setError(null);
      }
    } catch (err) {
      setError('Failed to fetch workflows');
    }
  }, []);

  const createWorkflow = useCallback(async (data: Partial<Workflow>): Promise<Workflow | null> => {
    setLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/workflows`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      if (resp.ok) {
        const result = await resp.json();
        await fetchWorkflows();
        return result.workflow;
      } else {
        const err = await resp.json();
        setError(err.detail || 'Failed to create workflow');
      }
    } catch (err) {
      setError(`Network error: ${err}`);
    } finally {
      setLoading(false);
    }
    return null;
  }, [fetchWorkflows]);

  const updateWorkflow = useCallback(async (id: string, data: Partial<Workflow>) => {
    setLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/workflows/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      if (resp.ok) {
        await fetchWorkflows();
        return true;
      }
    } catch (err) {
      setError(`Network error: ${err}`);
    } finally {
      setLoading(false);
    }
    return false;
  }, [fetchWorkflows]);

  const deleteWorkflow = useCallback(async (id: string) => {
    try {
      const resp = await fetch(`${API_BASE}/workflows/${id}`, { method: 'DELETE' });
      if (resp.ok) {
        await fetchWorkflows();
        return true;
      }
    } catch (err) {
      setError(`Network error: ${err}`);
    }
    return false;
  }, [fetchWorkflows]);

  const runWorkflow = useCallback(async (
    id: string,
    breakpoints: string[] = [],
    initialInputs: Record<string, any> = {},
  ) => {
    setLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/workflows/${id}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          agent_id: `workflow_${id}`,
          breakpoints,
          initial_inputs: initialInputs,
        }),
      });
      if (resp.ok) {
        return await resp.json();
      } else {
        const err = await resp.json();
        setError(err.detail || 'Failed to run workflow');
      }
    } catch (err) {
      setError(`Network error: ${err}`);
    } finally {
      setLoading(false);
    }
    return null;
  }, []);

  const exportYAML = useCallback(async (id: string): Promise<string | null> => {
    try {
      const resp = await fetch(`${API_BASE}/workflows/${id}/export`, { method: 'POST' });
      if (resp.ok) {
        return await resp.text();
      }
    } catch (err) {
      setError(`Network error: ${err}`);
    }
    return null;
  }, []);

  const importYAML = useCallback(async (yamlContent: string): Promise<Workflow | null> => {
    setLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/workflows/import`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ yaml_content: yamlContent }),
      });
      if (resp.ok) {
        const result = await resp.json();
        await fetchWorkflows();
        return result.workflow;
      } else {
        const err = await resp.json();
        setError(err.detail || 'Failed to import YAML');
      }
    } catch (err) {
      setError(`Network error: ${err}`);
    } finally {
      setLoading(false);
    }
    return null;
  }, [fetchWorkflows]);

  const fetchTemplates = useCallback(async (): Promise<WorkflowTemplate[]> => {
    try {
      const resp = await fetch(`${API_BASE}/workflows/templates/list`);
      if (resp.ok) {
        const data = await resp.json();
        return data.templates || [];
      }
    } catch (err) {
      setError(`Network error: ${err}`);
    }
    return [];
  }, []);

  const fetchRuns = useCallback(async (workflowId: string): Promise<WorkflowRun[]> => {
    try {
      const resp = await fetch(`${API_BASE}/workflows/${workflowId}/runs`);
      if (resp.ok) {
        const data = await resp.json();
        return data.runs || [];
      }
    } catch (err) {
      setError(`Network error: ${err}`);
    }
    return [];
  }, []);

  return {
    workflows,
    loading,
    error,
    fetchWorkflows,
    createWorkflow,
    updateWorkflow,
    deleteWorkflow,
    runWorkflow,
    exportYAML,
    importYAML,
    fetchTemplates,
    fetchRuns,
  };
}
