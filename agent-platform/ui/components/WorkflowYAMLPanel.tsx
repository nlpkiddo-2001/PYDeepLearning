import React, { useState, useEffect, useCallback } from 'react';
import { useWorkflow } from '../hooks/useWorkflow';

interface WorkflowYAMLPanelProps {
  workflowId: string;
  onClose: () => void;
}

export default function WorkflowYAMLPanel({ workflowId, onClose }: WorkflowYAMLPanelProps) {
  const [yamlContent, setYamlContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);
  const [importMode, setImportMode] = useState(false);
  const [importText, setImportText] = useState('');
  const { exportYAML, importYAML } = useWorkflow();

  useEffect(() => {
    (async () => {
      setLoading(true);
      const yaml = await exportYAML(workflowId);
      if (yaml) setYamlContent(yaml);
      setLoading(false);
    })();
  }, [workflowId, exportYAML]);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(yamlContent);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [yamlContent]);

  const handleDownload = useCallback(() => {
    const blob = new Blob([yamlContent], { type: 'text/yaml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `workflow_${workflowId}.yaml`;
    a.click();
    URL.revokeObjectURL(url);
  }, [yamlContent, workflowId]);

  const handleImport = useCallback(async () => {
    if (!importText.trim()) return;
    const wf = await importYAML(importText);
    if (wf) {
      alert(`Imported workflow: ${wf.name}`);
      onClose();
    }
  }, [importText, importYAML, onClose]);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content yaml-modal" onClick={(e) => e.stopPropagation()}>
        <div className="yaml-modal-header">
          <h3>📄 Workflow YAML</h3>
          <div className="yaml-tabs">
            <button
              className={`yaml-tab ${!importMode ? 'active' : ''}`}
              onClick={() => setImportMode(false)}
            >
              Export
            </button>
            <button
              className={`yaml-tab ${importMode ? 'active' : ''}`}
              onClick={() => setImportMode(true)}
            >
              Import
            </button>
          </div>
          <button className="debug-close-btn" onClick={onClose}>✕</button>
        </div>

        {!importMode ? (
          <>
            <div className="yaml-content">
              {loading ? (
                <div style={{ color: 'var(--text-muted)', padding: '20px' }}>Loading...</div>
              ) : (
                <pre className="yaml-pre">{yamlContent}</pre>
              )}
            </div>
            <div className="yaml-actions">
              <button className="wf-btn" onClick={handleCopy}>
                {copied ? '✓ Copied' : '📋 Copy'}
              </button>
              <button className="wf-btn" onClick={handleDownload}>
                ⬇ Download .yaml
              </button>
            </div>
          </>
        ) : (
          <>
            <div className="yaml-content">
              <textarea
                className="yaml-import-textarea"
                value={importText}
                onChange={(e) => setImportText(e.target.value)}
                placeholder="Paste workflow YAML here..."
                rows={20}
              />
            </div>
            <div className="yaml-actions">
              <button className="wf-btn save-btn" onClick={handleImport} disabled={!importText.trim()}>
                Import Workflow
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
