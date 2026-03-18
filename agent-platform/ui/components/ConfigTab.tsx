import React, { useState, useEffect } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface ConfigTabProps {
  agentId: string;
}

interface ConfigState {
  provider: string;
  model: string;
  base_url: string;
  api_key: string;
  temperature: number;
  max_tokens: number;
}

interface ProviderProfile {
  provider: string;
  model: string;
  label: string;
}

/* Pre-configured provider presets — matching agent.yaml providers section */
const PROVIDER_PRESETS: Record<string, Partial<ConfigState>> = {
  gemini: {
    provider: 'gemini',
    model: 'gemini-3-flash-preview',
    api_key: 'AIzaSyABiD612PXe2QKSnbfVbNHAzKcOmSCJd90',
    base_url: '',
    temperature: 0.3,
    max_tokens: 64000,
  },
  vllm: {
    provider: 'vllm',
    model: 'glm-5',
    base_url: '',
    api_key: '',
    temperature: 0.3,
    max_tokens: 64000,
  },
};

export default function ConfigTab({ agentId }: ConfigTabProps) {
  const [config, setConfig] = useState<ConfigState>({
    provider: 'gemini',
    model: 'gemini-3-flash-preview',
    base_url: '',
    api_key: '',
    temperature: 0.3,
    max_tokens: 64000,
  });
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [profiles, setProfiles] = useState<Record<string, ProviderProfile>>({});

  // Load available profiles from API
  useEffect(() => {
    (async () => {
      try {
        const resp = await fetch(`${API_BASE}/config/providers`);
        if (resp.ok) {
          const data = await resp.json();
          if (data.profiles) setProfiles(data.profiles);
        }
      } catch {}
    })();
  }, []);

  // Load current config
  useEffect(() => {
    (async () => {
      try {
        const resp = await fetch(`${API_BASE}/agents/${agentId}`);
        if (resp.ok) {
          const data = await resp.json();
          if (data.llm) {
            setConfig((prev) => ({
              ...prev,
              provider: data.llm.provider || prev.provider,
              model: data.llm.model || prev.model,
            }));
          }
        }
      } catch {}
    })();
  }, [agentId]);

  /* When provider selection changes, autofill with preset values */
  const handleProviderChange = (provider: string) => {
    const preset = PROVIDER_PRESETS[provider];
    if (preset) {
      setConfig((prev) => ({ ...prev, ...preset }));
    } else {
      setConfig((prev) => ({ ...prev, provider }));
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setSaved(false);
    try {
      const body: Record<string, any> = {};
      if (config.provider) body.provider = config.provider;
      if (config.model) body.model = config.model;
      if (config.base_url) body.base_url = config.base_url;
      if (config.api_key) body.api_key = config.api_key;
      if (config.temperature !== undefined) body.temperature = config.temperature;
      if (config.max_tokens) body.max_tokens = config.max_tokens;

      const resp = await fetch(`${API_BASE}/agents/${agentId}/config`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (resp.ok) {
        setSaved(true);
        setTimeout(() => setSaved(false), 2000);
      } else {
        const err = await resp.json();
        alert(`Error: ${err.detail}`);
      }
    } catch (err) {
      alert(`Network error: ${err}`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="config-tab">
      <div className="config-section">
        <h3>LLM Provider</h3>
        <div className="config-field">
          <label>Provider</label>
          <select
            value={config.provider}
            onChange={(e) => handleProviderChange(e.target.value)}
          >
            <option value="gemini">Google Gemini</option>
            <option value="vllm">vLLM (GLM-5)</option>
            <option value="openai">OpenAI</option>
            <option value="anthropic">Anthropic</option>
            <option value="ollama">Ollama</option>
            <option value="custom">Custom Endpoint</option>
          </select>
        </div>
        <div className="config-field">
          <label>Model</label>
          <input
            type="text"
            value={config.model}
            onChange={(e) => setConfig({ ...config, model: e.target.value })}
            placeholder="e.g., gemini-3-flash-preview, glm-5, gpt-4o"
          />
        </div>
        <div className="config-field">
          <label>API Key</label>
          <input
            type="password"
            value={config.api_key}
            onChange={(e) => setConfig({ ...config, api_key: e.target.value })}
            placeholder="Provider API key"
          />
        </div>
        <div className="config-field">
          <label>Custom Endpoint URL (for Ollama / vLLM)</label>
          <input
            type="text"
            value={config.base_url}
            onChange={(e) => setConfig({ ...config, base_url: e.target.value })}
            placeholder="http://localhost:11434/v1"
          />
        </div>
      </div>

      <div className="config-section">
        <h3>Generation Settings</h3>
        <div className="config-field">
          <label>Temperature: {config.temperature}</label>
          <input
            type="range"
            min="0"
            max="2"
            step="0.1"
            value={config.temperature}
            onChange={(e) => setConfig({ ...config, temperature: parseFloat(e.target.value) })}
          />
        </div>
        <div className="config-field">
          <label>Max Tokens</label>
          <input
            type="number"
            value={config.max_tokens}
            onChange={(e) => setConfig({ ...config, max_tokens: parseInt(e.target.value) || 4096 })}
            min={256}
            max={128000}
          />
        </div>
      </div>

      <button className="save-config-btn" onClick={handleSave} disabled={saving}>
        {saving ? 'Saving...' : saved ? '✓ Saved' : 'Save Configuration'}
      </button>
    </div>
  );
}
