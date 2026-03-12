import React, { useState, useEffect, useRef } from 'react';
import { StreamEvent } from '../hooks/useAgentStream';

interface WorkflowDebugPanelProps {
  events: StreamEvent[];
  onClose: () => void;
}

/** A single workflow step's aggregated state. */
interface WFStepState {
  stepId: string;
  stepName: string;
  tool: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped' | 'paused';
  durationMs?: number;
  output?: string;
  error?: string;
  attempt?: number;
  maxRetries?: number;
  inputs?: Record<string, any>;
  events: StreamEvent[];
}

function aggregateWorkflowSteps(events: StreamEvent[]): {
  steps: WFStepState[];
  workflowName?: string;
  totalSteps?: number;
  isComplete: boolean;
  completedSteps?: number;
  durationSec?: number;
  finalError?: string;
} {
  const stepsMap = new Map<string, WFStepState>();
  let workflowName: string | undefined;
  let totalSteps: number | undefined;
  let isComplete = false;
  let completedSteps: number | undefined;
  let durationSec: number | undefined;
  let finalError: string | undefined;

  for (const ev of events) {
    const d = ev.data;
    const wfEvent = d.workflow_event;

    if (wfEvent === 'workflow_start') {
      workflowName = d.workflow_name;
      totalSteps = d.total_steps;
    } else if (wfEvent === 'workflow_step_start') {
      const existing = stepsMap.get(d.step_id);
      stepsMap.set(d.step_id, {
        ...(existing || { events: [] }),
        stepId: d.step_id,
        stepName: d.step_name,
        tool: d.tool,
        status: 'running',
        attempt: d.attempt,
        inputs: d.inputs,
        events: [...(existing?.events || []), ev],
      });
    } else if (wfEvent === 'workflow_step_complete') {
      const existing = stepsMap.get(d.step_id);
      if (existing) {
        existing.status = 'completed';
        existing.durationMs = d.duration_ms;
        existing.output = d.output;
        existing.events.push(ev);
      }
    } else if (wfEvent === 'workflow_step_error') {
      const existing = stepsMap.get(d.step_id);
      if (existing) {
        if (!d.will_retry) existing.status = 'failed';
        existing.error = d.error;
        existing.attempt = d.attempt;
        existing.maxRetries = d.max_retries;
        existing.events.push(ev);
      }
    } else if (wfEvent === 'workflow_step_skip') {
      const existing = stepsMap.get(d.step_id);
      if (existing) {
        existing.status = 'skipped';
        existing.events.push(ev);
      } else {
        stepsMap.set(d.step_id, {
          stepId: d.step_id,
          stepName: d.step_name,
          tool: '',
          status: 'skipped',
          events: [ev],
        });
      }
    } else if (wfEvent === 'workflow_step_paused') {
      const existing = stepsMap.get(d.step_id);
      if (existing) {
        existing.status = 'paused';
        existing.events.push(ev);
      }
    } else if (wfEvent === 'workflow_complete') {
      isComplete = true;
      completedSteps = d.completed_steps;
      durationSec = d.duration_seconds;
    } else if (wfEvent === 'workflow_error') {
      isComplete = true;
      finalError = d.error;
    }
  }

  return {
    steps: Array.from(stepsMap.values()),
    workflowName,
    totalSteps,
    isComplete,
    completedSteps,
    durationSec,
    finalError,
  };
}

function WFStepCard({ step }: { step: WFStepState }) {
  const [expanded, setExpanded] = useState(step.status === 'running');

  useEffect(() => {
    if (step.status === 'running') setExpanded(true);
  }, [step.status]);

  const statusIcons: Record<string, string> = {
    pending: '○',
    running: '',
    completed: '✓',
    failed: '✕',
    skipped: '⊘',
    paused: '⏸',
  };

  return (
    <div className={`wf-step-card ${step.status}`}>
      <div className="wf-step-card-header" onClick={() => setExpanded(!expanded)}>
        <div className="wf-step-card-left">
          <span className={`wf-step-status-icon ${step.status}`}>
            {step.status === 'running' ? (
              <span className="thinking-spinner" />
            ) : statusIcons[step.status] || '○'}
          </span>
          <span className="wf-step-name">{step.stepName}</span>
          {step.tool && <span className="wf-step-tool-badge">{step.tool}</span>}
          {step.durationMs !== undefined && (
            <span className="wf-step-duration">{step.durationMs}ms</span>
          )}
        </div>
        <span className="thinking-expand-icon">{expanded ? '▾' : '▸'}</span>
      </div>

      {expanded && (
        <div className="wf-step-card-body">
          {/* Inputs */}
          {step.inputs && Object.keys(step.inputs).length > 0 && (
            <div className="wf-step-section">
              <div className="wf-step-section-label">Inputs</div>
              <div className="wf-step-section-content">
                {Object.entries(step.inputs).map(([k, v]) => (
                  <div key={k} className="wf-step-input-row">
                    <span className="wf-step-input-key">{k}</span>
                    <span className="wf-step-input-val">{String(v).slice(0, 100)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Output */}
          {step.output && (
            <div className="wf-step-section">
              <div className="wf-step-section-label">Output</div>
              <div className="wf-step-section-content output">
                {step.output.length > 400 ? step.output.slice(0, 400) + '…' : step.output}
              </div>
            </div>
          )}

          {/* Retries */}
          {step.attempt && step.attempt > 1 && (
            <div className="wf-step-section">
              <div className="wf-step-section-content retry">
                Attempt {step.attempt}{step.maxRetries ? `/${step.maxRetries}` : ''}
              </div>
            </div>
          )}

          {/* Error */}
          {step.error && (
            <div className="wf-step-section">
              <div className="wf-step-section-content error-msg">
                {step.error}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function WorkflowDebugPanel({ events, onClose }: WorkflowDebugPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const { steps, workflowName, totalSteps, isComplete, completedSteps, durationSec, finalError } =
    aggregateWorkflowSteps(events);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [events]);

  const isRunning = !isComplete && events.length > 0;
  const successCount = steps.filter(s => s.status === 'completed').length;
  const failCount = steps.filter(s => s.status === 'failed').length;

  return (
    <div className="workflow-debug-panel">
      {/* Header */}
      <div className="debug-panel-header">
        <span className="debug-panel-title">
          {isRunning ? (
            <><span className="thinking-spinner" /> Running</>
          ) : isComplete && !finalError ? (
            <>✓ Completed</>
          ) : isComplete && finalError ? (
            <>✕ Failed</>
          ) : (
            <>🐛 Step Debugger</>
          )}
        </span>
        {isComplete && durationSec !== undefined && (
          <span className="debug-duration-badge">{durationSec.toFixed(1)}s</span>
        )}
        <span className="debug-event-count">
          {successCount > 0 && <span className="debug-count-success">{successCount} passed</span>}
          {failCount > 0 && <span className="debug-count-fail">{failCount} failed</span>}
        </span>
        <button className="debug-close-btn" onClick={onClose}>✕</button>
      </div>

      {/* Workflow info banner */}
      {workflowName && (
        <div className="wf-debug-banner">
          <span>{workflowName}</span>
          {totalSteps !== undefined && <span className="wf-debug-total">({totalSteps} steps)</span>}
        </div>
      )}

      {/* Step cards */}
      <div className="debug-panel-body">
        {steps.length === 0 ? (
          <div className="debug-empty">
            Run the workflow to see step-by-step execution logs here.
            <br /><br />
            Set breakpoints by clicking the ● button on any step node.
          </div>
        ) : (
          steps.map((step) => <WFStepCard key={step.stepId} step={step} />)
        )}

        {/* Final result / error */}
        {isComplete && !finalError && completedSteps !== undefined && (
          <div className="wf-debug-final success">
            <span className="wf-debug-final-icon">✓</span>
            Workflow completed — {completedSteps}/{totalSteps || '?'} steps
            {durationSec !== undefined && ` in ${durationSec.toFixed(1)}s`}
          </div>
        )}
        {finalError && (
          <div className="wf-debug-final error">
            <span className="wf-debug-final-icon">✕</span>
            {finalError}
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
