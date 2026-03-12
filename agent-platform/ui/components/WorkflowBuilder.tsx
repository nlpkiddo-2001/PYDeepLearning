import React, { useState, useCallback, useEffect, useRef } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  MiniMap,
  Connection,
  addEdge,
  useNodesState,
  useEdgesState,
  NodeTypes,
  Handle,
  Position,
  ReactFlowProvider,
  BackgroundVariant,
} from 'reactflow';
import 'reactflow/dist/style.css';
import {
  useWorkflow,
  Workflow,
  WorkflowStep,
  WorkflowEdge as WFEdge,
  WorkflowTemplate,
} from '../hooks/useWorkflow';
import { useAgentStream, StreamEvent } from '../hooks/useAgentStream';
import WorkflowDebugPanel from './WorkflowDebugPanelV2';
import WorkflowYAMLPanel from './WorkflowYAMLPanel';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ─── Custom Node ─────────────────────────────────────────────────

interface ToolNodeData {
  label: string;
  tool: string;
  status?: string;
  stepId: string;
  isBreakpoint?: boolean;
  onEdit: (stepId: string) => void;
  onToggleBreakpoint: (stepId: string) => void;
  onDelete: (stepId: string) => void;
}

function ToolNode({ data }: { data: ToolNodeData }) {
  const statusColors: Record<string, string> = {
    pending: 'var(--text-muted)',
    running: 'var(--accent-orange)',
    completed: 'var(--accent-green)',
    failed: 'var(--accent-red)',
    skipped: 'var(--text-muted)',
    paused: 'var(--accent-purple)',
  };
  const borderColor = statusColors[data.status || 'pending'] || 'var(--border)';

  return (
    <div className="workflow-node" style={{ borderColor }}>
      <Handle type="target" position={Position.Top} className="workflow-handle" />
      <div className="workflow-node-header">
        <span className="workflow-node-name">{data.label}</span>
        <div className="workflow-node-actions">
          <button
            className={`bp-btn ${data.isBreakpoint ? 'active' : ''}`}
            onClick={(e) => { e.stopPropagation(); data.onToggleBreakpoint(data.stepId); }}
            title="Toggle Breakpoint"
          >●</button>
          <button
            className="edit-btn"
            onClick={(e) => { e.stopPropagation(); data.onEdit(data.stepId); }}
            title="Edit Step"
          >✎</button>
          <button
            className="del-btn"
            onClick={(e) => { e.stopPropagation(); data.onDelete(data.stepId); }}
            title="Delete Step"
          >✕</button>
        </div>
      </div>
      <div className="workflow-node-tool">{data.tool}</div>
      {data.status && data.status !== 'pending' && (
        <div className="workflow-node-status" style={{ color: borderColor }}>
          {data.status.toUpperCase()}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} className="workflow-handle" />
    </div>
  );
}

const nodeTypes: NodeTypes = { toolNode: ToolNode };

// ─── Step Editor Modal ───────────────────────────────────────────

interface StepEditorProps {
  step: WorkflowStep | null;
  availableTools: string[];
  allSteps: WorkflowStep[];
  onSave: (step: WorkflowStep) => void;
  onClose: () => void;
}

function StepEditor({ step, availableTools, allSteps, onSave, onClose }: StepEditorProps) {
  const [editStep, setEditStep] = useState<WorkflowStep>(
    step || {
      id: `step_${Date.now()}`,
      name: 'New Step',
      tool: availableTools[0] || '',
      inputs: {},
      input_mapping: {},
      timeout: 60,
      max_retries: 2,
      depends_on: [],
      position: { x: 250, y: 200 },
    }
  );
  const [inputKey, setInputKey] = useState('');
  const [inputVal, setInputVal] = useState('');
  const [mappingKey, setMappingKey] = useState('');
  const [mappingVal, setMappingVal] = useState('');

  const addInput = () => {
    if (!inputKey.trim()) return;
    setEditStep({ ...editStep, inputs: { ...editStep.inputs, [inputKey]: inputVal } });
    setInputKey(''); setInputVal('');
  };

  const addMapping = () => {
    if (!mappingKey.trim()) return;
    setEditStep({ ...editStep, input_mapping: { ...editStep.input_mapping, [mappingKey]: mappingVal } });
    setMappingKey(''); setMappingVal('');
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content step-editor-modal" onClick={(e) => e.stopPropagation()}>
        <h3>{step ? 'Edit Step' : 'Add Step'}</h3>

        <div className="config-field">
          <label>Step Name</label>
          <input type="text" value={editStep.name}
            onChange={(e) => setEditStep({ ...editStep, name: e.target.value })} />
        </div>

        <div className="config-field">
          <label>Tool</label>
          <select value={editStep.tool}
            onChange={(e) => setEditStep({ ...editStep, tool: e.target.value })}>
            {availableTools.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>

        <div className="config-field">
          <label>Static Inputs</label>
          <div className="kv-list">
            {Object.entries(editStep.inputs).map(([k, v]) => (
              <div key={k} className="kv-row">
                <span className="kv-key">{k}</span>
                <span className="kv-sep">→</span>
                <span className="kv-val">{String(v)}</span>
                <button className="kv-del" onClick={() => {
                  const copy = { ...editStep.inputs };
                  delete copy[k];
                  setEditStep({ ...editStep, inputs: copy });
                }}>✕</button>
              </div>
            ))}
          </div>
          <div className="kv-add-row">
            <input type="text" placeholder="key" value={inputKey} onChange={(e) => setInputKey(e.target.value)} />
            <input type="text" placeholder="value" value={inputVal} onChange={(e) => setInputVal(e.target.value)} />
            <button onClick={addInput}>+</button>
          </div>
        </div>

        <div className="config-field">
          <label>Input Mappings (from previous steps)</label>
          <div className="kv-list">
            {Object.entries(editStep.input_mapping).map(([k, v]) => (
              <div key={k} className="kv-row">
                <span className="kv-key">{k}</span>
                <span className="kv-sep">←</span>
                <span className="kv-val">{v}</span>
                <button className="kv-del" onClick={() => {
                  const copy = { ...editStep.input_mapping };
                  delete copy[k];
                  setEditStep({ ...editStep, input_mapping: copy });
                }}>✕</button>
              </div>
            ))}
          </div>
          <div className="kv-add-row">
            <input type="text" placeholder="param name" value={mappingKey}
              onChange={(e) => setMappingKey(e.target.value)} />
            <input type="text" placeholder="$steps.stepId.output" value={mappingVal}
              onChange={(e) => setMappingVal(e.target.value)} />
            <button onClick={addMapping}>+</button>
          </div>
        </div>

        <div className="config-field">
          <label>Condition (optional Python expression)</label>
          <input type="text" value={editStep.condition || ''}
            placeholder='e.g., len(steps["step1"]) > 0'
            onChange={(e) => setEditStep({ ...editStep, condition: e.target.value || undefined })} />
        </div>

        <div className="config-field">
          <label>Timeout (seconds): {editStep.timeout}</label>
          <input type="range" min="5" max="300" step="5" value={editStep.timeout}
            onChange={(e) => setEditStep({ ...editStep, timeout: parseInt(e.target.value) })} />
        </div>

        <div className="config-field">
          <label>Max Retries: {editStep.max_retries}</label>
          <input type="range" min="0" max="5" value={editStep.max_retries}
            onChange={(e) => setEditStep({ ...editStep, max_retries: parseInt(e.target.value) })} />
        </div>

        <div className="modal-actions">
          <button className="save-config-btn" onClick={() => onSave(editStep)}>
            {step ? 'Update Step' : 'Add Step'}
          </button>
          <button className="cancel-btn" onClick={onClose}>Cancel</button>
        </div>
      </div>
    </div>
  );
}

// ─── Main Workflow Builder ───────────────────────────────────────

interface WorkflowBuilderProps {
  workflow: Workflow | null;
  onSave: (wf: Partial<Workflow>) => void;
}

export default function WorkflowBuilder({ workflow, onSave }: WorkflowBuilderProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [editingStep, setEditingStep] = useState<WorkflowStep | null>(null);
  const [showStepEditor, setShowStepEditor] = useState(false);
  const [showDebugPanel, setShowDebugPanel] = useState(false);
  const [showYAMLPanel, setShowYAMLPanel] = useState(false);
  const [breakpoints, setBreakpoints] = useState<Set<string>>(new Set());
  const [runEvents, setRunEvents] = useState<StreamEvent[]>([]);
  const [availableTools, setAvailableTools] = useState<string[]>([]);
  const [workflowName, setWorkflowName] = useState(workflow?.name || 'Untitled Workflow');
  const [workflowDesc, setWorkflowDesc] = useState(workflow?.description || '');
  const [isRunning, setIsRunning] = useState(false);
  const { runWorkflow, updateWorkflow, exportYAML } = useWorkflow();

  // Fetch available tools
  useEffect(() => {
    (async () => {
      try {
        const resp = await fetch(`${API_BASE}/tools`);
        if (resp.ok) {
          const data = await resp.json();
          setAvailableTools(data.tools || []);
        }
      } catch {}
    })();
  }, []);

  // Connect to WebSocket for live step updates
  const agentId = workflow ? `workflow_${workflow.id}` : '';
  const { events, connected, connect } = useAgentStream({
    agentId,
    autoConnect: !!workflow,
    onEvent: (event) => {
      setRunEvents((prev) => [...prev, event]);
      // Update node statuses based on workflow events
      const data = event.data;
      if (data.workflow_event === 'workflow_step_start') {
        updateNodeStatus(data.step_id, 'running');
      } else if (data.workflow_event === 'workflow_step_complete') {
        updateNodeStatus(data.step_id, 'completed');
      } else if (data.workflow_event === 'workflow_step_error' && !data.will_retry) {
        updateNodeStatus(data.step_id, 'failed');
      } else if (data.workflow_event === 'workflow_step_skip') {
        updateNodeStatus(data.step_id, 'skipped');
      } else if (data.workflow_event === 'workflow_step_paused') {
        updateNodeStatus(data.step_id, 'paused');
      } else if (data.workflow_event === 'workflow_complete' || data.workflow_event === 'workflow_error') {
        setIsRunning(false);
      }
    },
  });

  const updateNodeStatus = (stepId: string, status: string) => {
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id === stepId) {
          return { ...n, data: { ...n.data, status } };
        }
        return n;
      })
    );
  };

  // Convert workflow steps/edges to ReactFlow format
  useEffect(() => {
    if (!workflow) return;
    setWorkflowName(workflow.name);
    setWorkflowDesc(workflow.description);

    const rfNodes: Node[] = workflow.steps.map((step) => ({
      id: step.id,
      type: 'toolNode',
      position: step.position || { x: 0, y: 0 },
      data: {
        label: step.name,
        tool: step.tool,
        status: step.status || 'pending',
        stepId: step.id,
        isBreakpoint: breakpoints.has(step.id),
        onEdit: handleEditStep,
        onToggleBreakpoint: handleToggleBreakpoint,
        onDelete: handleDeleteStep,
      },
    }));

    const rfEdges: Edge[] = workflow.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      label: edge.label || '',
      animated: true,
      style: { stroke: 'var(--accent-blue)' },
      labelStyle: { fill: 'var(--text-secondary)', fontSize: 11 },
    }));

    setNodes(rfNodes);
    setEdges(rfEdges);
  }, [workflow]); // eslint-disable-line

  // ── Handlers ──────────────────────────────────────────────

  const onConnect = useCallback(
    (params: Connection) => {
      setEdges((eds) =>
        addEdge({
          ...params,
          animated: true,
          style: { stroke: 'var(--accent-blue)' },
        }, eds)
      );
    },
    [setEdges]
  );

  const handleEditStep = useCallback((stepId: string) => {
    const step = workflow?.steps.find((s) => s.id === stepId);
    if (step) {
      setEditingStep(step);
      setShowStepEditor(true);
    }
  }, [workflow]);

  const handleDeleteStep = useCallback((stepId: string) => {
    setNodes((nds) => nds.filter((n) => n.id !== stepId));
    setEdges((eds) => eds.filter((e) => e.source !== stepId && e.target !== stepId));
  }, [setNodes, setEdges]);

  const handleToggleBreakpoint = useCallback((stepId: string) => {
    setBreakpoints((prev) => {
      const next = new Set(prev);
      if (next.has(stepId)) next.delete(stepId);
      else next.add(stepId);
      return next;
    });
    // Update node visual
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id === stepId) {
          return {
            ...n,
            data: { ...n.data, isBreakpoint: !breakpoints.has(stepId) },
          };
        }
        return n;
      })
    );
  }, [breakpoints, setNodes]);

  const handleAddStep = () => {
    setEditingStep(null);
    setShowStepEditor(true);
  };

  const handleStepSave = (step: WorkflowStep) => {
    if (editingStep) {
      // Update existing node
      setNodes((nds) =>
        nds.map((n) => {
          if (n.id === step.id) {
            return {
              ...n,
              data: {
                ...n.data,
                label: step.name,
                tool: step.tool,
              },
            };
          }
          return n;
        })
      );
    } else {
      // Add new node
      const newNode: Node = {
        id: step.id,
        type: 'toolNode',
        position: step.position,
        data: {
          label: step.name,
          tool: step.tool,
          status: 'pending',
          stepId: step.id,
          isBreakpoint: false,
          onEdit: handleEditStep,
          onToggleBreakpoint: handleToggleBreakpoint,
          onDelete: handleDeleteStep,
        },
      };
      setNodes((nds) => [...nds, newNode]);
    }
    setShowStepEditor(false);
  };

  const handleSave = async () => {
    // Convert ReactFlow state back to workflow format
    const steps: WorkflowStep[] = nodes.map((n) => {
      const existing = workflow?.steps.find((s) => s.id === n.id);
      return {
        id: n.id,
        name: n.data.label,
        tool: n.data.tool,
        inputs: existing?.inputs || {},
        input_mapping: existing?.input_mapping || {},
        condition: existing?.condition,
        timeout: existing?.timeout || 60,
        max_retries: existing?.max_retries || 2,
        depends_on: existing?.depends_on || [],
        position: n.position,
      };
    });
    const wfEdges: WFEdge[] = edges.map((e) => ({
      id: e.id || `edge_${e.source}_${e.target}`,
      source: e.source,
      target: e.target,
      label: (e.label as string) || '',
    }));

    onSave({
      name: workflowName,
      description: workflowDesc,
      steps,
      edges: wfEdges,
    });
  };

  const handleRun = async () => {
    if (!workflow) return;
    setIsRunning(true);
    setRunEvents([]);

    // Reset all node statuses
    setNodes((nds) =>
      nds.map((n) => ({ ...n, data: { ...n.data, status: 'pending' } }))
    );

    // Ensure WebSocket is connected
    if (!connected) connect();

    // Save current state first
    await handleSave();
    await runWorkflow(workflow.id, Array.from(breakpoints));
    setShowDebugPanel(true);
  };

  const handleExportYAML = async () => {
    if (!workflow) return;
    setShowYAMLPanel(true);
  };

  return (
    <div className="workflow-builder">
      {/* Toolbar */}
      <div className="workflow-toolbar">
        <div className="workflow-toolbar-left">
          <input
            className="workflow-name-input"
            value={workflowName}
            onChange={(e) => setWorkflowName(e.target.value)}
            placeholder="Workflow Name"
          />
          <input
            className="workflow-desc-input"
            value={workflowDesc}
            onChange={(e) => setWorkflowDesc(e.target.value)}
            placeholder="Description"
          />
        </div>
        <div className="workflow-toolbar-right">
          <button className="wf-btn add-step-btn" onClick={handleAddStep}>
            + Add Step
          </button>
          <button className="wf-btn save-btn" onClick={handleSave}>
            💾 Save
          </button>
          <button className="wf-btn yaml-btn" onClick={handleExportYAML}>
            📄 YAML
          </button>
          <button
            className="wf-btn run-btn"
            onClick={handleRun}
            disabled={isRunning || !workflow}
          >
            {isRunning ? '⟳ Running...' : '▸ Run'}
          </button>
          <button
            className={`wf-btn debug-btn ${showDebugPanel ? 'active' : ''}`}
            onClick={() => setShowDebugPanel(!showDebugPanel)}
          >
            🐛 Debug
          </button>
        </div>
      </div>

      <div className="workflow-main">
        {/* React Flow Canvas */}
        <div className="workflow-canvas">
          <ReactFlowProvider>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              nodeTypes={nodeTypes}
              fitView
              snapToGrid
              snapGrid={[20, 20]}
              defaultEdgeOptions={{
                animated: true,
                style: { stroke: 'var(--accent-blue)', strokeWidth: 2 },
              }}
            >
              <Controls
                style={{ background: 'var(--bg-tertiary)', borderRadius: '8px' }}
              />
              <MiniMap
                nodeColor={(n) => {
                  const st = n.data?.status;
                  if (st === 'completed') return '#3fb950';
                  if (st === 'running') return '#d29922';
                  if (st === 'failed') return '#f85149';
                  return '#484f58';
                }}
                style={{ background: 'var(--bg-secondary)' }}
              />
              <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--border)" />
            </ReactFlow>
          </ReactFlowProvider>
        </div>

        {/* Debug Panel (right side) */}
        {showDebugPanel && (
          <WorkflowDebugPanel
            events={runEvents}
            onClose={() => setShowDebugPanel(false)}
          />
        )}
      </div>

      {/* Step Editor Modal */}
      {showStepEditor && (
        <StepEditor
          step={editingStep}
          availableTools={availableTools}
          allSteps={workflow?.steps || []}
          onSave={handleStepSave}
          onClose={() => setShowStepEditor(false)}
        />
      )}

      {/* YAML Panel Modal */}
      {showYAMLPanel && workflow && (
        <WorkflowYAMLPanel
          workflowId={workflow.id}
          onClose={() => setShowYAMLPanel(false)}
        />
      )}
    </div>
  );
}
