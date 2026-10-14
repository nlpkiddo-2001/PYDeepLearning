"""Mixture configuration: domains -> sources, token weights, epoch caps, filters, stages."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

OVER_LENGTH_POLICIES = ("truncate", "drop")
DECONTAM_MODES = ("required", "if_present", "off")


class ConfigError(ValueError):
    """Invalid or inconsistent mixture configuration."""


@dataclass
class CacheSpec:
    tokenizer: str
    template_path: Optional[str]
    cache_root: Path
    fingerprint: str

    @property
    def root(self) -> Path:
        return self.cache_root / self.fingerprint[:12]

    def source_dir(self, source: str) -> Path:
        return self.root / source



@dataclass
class SourceCfg:
    name: str
    weight: Optional[float] = None          # relative weight inside its domain (all-or-none per domain)
    max_epochs: Optional[float] = None      # None -> MixConfig.max_epochs
    min_trainable_tokens: int = 1
    max_tokens: Optional[int] = None        # drop samples longer than this (independent of seq_len)
    langs: Optional[list[str]] = None       # keep only these meta.lang values
    require_tool_valid: bool = True
    enabled: bool = True


@dataclass
class DomainCfg:
    name: str
    weight: float
    sources: list[SourceCfg] = field(default_factory=list)


@dataclass
class StageCfg:
    name: str
    until_tokens: Optional[float] = None    # global packed-token boundary; None = rest of training
    domain_weights: Optional[dict[str, float]] = None   # override (others keep base weight)
    source_weights: Optional[dict[str, float]] = None   # override of within-domain weights


@dataclass
class TokenConfig:
    instruct_start: str
    assistant_token: str
    instruct_end: str

@dataclass
class MixConfig:
    data_root: str = "out"
    cache_dir: str = "sft_cache"
    tokenizer: str = "Qwen/Qwen2.5-7B-Instruct"
    chat_template: Optional[str] = None     # path to a jinja template; None = the tokenizer's own
    seq_len: int = 8192
    seed: int = 0
    token_config: TokenConfig = None
    alpha: float = 0.5                      # within-domain temperature smoothing: p ~ tokens**alpha
    max_epochs: float = 3.0
    val_fraction: float = 0.01
    val_max_samples_per_source: int = 2000
    split_seed: int = 0
    over_length: str = "truncate"           # truncate at the last complete assistant turn | drop
    min_truncated_trainable: int = 16
    pack_window: int = 512
    micro_batch_size: int = 1
    license_deny: list[str] = field(default_factory=list)   # case-insensitive substrings of the source license
    # benchmark overlap flags from dataset/decontaminate.py (<source>/_contamination.parquet)
    decontam: str = "if_present"            # required: every source must have a current result | if_present | off
    decontam_min_coverage: float = 0.0      # drop flagged samples whose max_item_coverage >= this (0 = any hit)
    domains: list[DomainCfg] = field(default_factory=list)
    stages: list[StageCfg] = field(default_factory=list)

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_yaml(cls, path: str | Path) -> "MixConfig":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MixConfig":
        raw = dict(raw)
        known = set(cls.__dataclass_fields__) - {"domains", "stages"}
        unknown = set(raw) - known - {"domains", "stages"}
        if unknown:
            raise ConfigError(f"unknown config keys: {sorted(unknown)}")
        doms = raw.pop("domains", None) or {}
        if not isinstance(doms, dict):
            raise ConfigError("`domains` must be a mapping: name -> {weight, sources}")
        stages = raw.pop("stages", None) or []
        if "token_config" in raw and raw["token_config"]:
            raw["token_config"] = TokenConfig(**raw["token_config"])
        cfg = cls(**raw)
        for dname, d in doms.items():
            d = d or {}
            unknown = set(d) - {"weight", "sources"}
            if unknown:
                raise ConfigError(f"domain {dname}: unknown keys {sorted(unknown)}")
            srcs = d.get("sources") or {}
            if isinstance(srcs, list):
                srcs = {s: {} for s in srcs}
            sources = []
            for sname, s in srcs.items():
                s = s or {}
                bad = set(s) - set(SourceCfg.__dataclass_fields__)
                if bad:
                    raise ConfigError(f"source {sname}: unknown keys {sorted(bad)}")
                sources.append(SourceCfg(name=sname, **s))
            if "weight" not in d:
                raise ConfigError(f"domain {dname}: `weight` is required")
            cfg.domains.append(DomainCfg(name=dname, weight=float(d["weight"]), sources=sources))
        for i, s in enumerate(stages):
            bad = set(s) - set(StageCfg.__dataclass_fields__)
            if bad:
                raise ConfigError(f"stage {i}: unknown keys {sorted(bad)}")
            cfg.stages.append(StageCfg(**{"name": f"stage{i}", **s}))
        cfg.validate()
        return cfg

    # ------------------------------------------------------------- validation
    def validate(self) -> None:
        if self.seq_len < 16:
            raise ConfigError("seq_len must be >= 16")
        if self.over_length not in OVER_LENGTH_POLICIES:
            raise ConfigError(f"over_length must be one of {OVER_LENGTH_POLICIES}")
        if not 0 <= self.val_fraction < 1:
            raise ConfigError("val_fraction must be in [0, 1)")
        if self.decontam not in DECONTAM_MODES:
            raise ConfigError(f"decontam must be one of {DECONTAM_MODES}")
        if not 0 <= self.decontam_min_coverage <= 1:
            raise ConfigError("decontam_min_coverage must be in [0, 1]")
        if self.max_epochs <= 0 or not 0 <= self.alpha <= 1.5:
            raise ConfigError("max_epochs must be > 0 and alpha in [0, 1.5]")
        if self.pack_window < 1 or self.micro_batch_size < 1:
            raise ConfigError("pack_window and micro_batch_size must be >= 1")
        if not self.domains:
            raise ConfigError("at least one domain is required")
        seen: dict[str, str] = {}
        for d in self.domains:
            if not (d.weight >= 0 and math.isfinite(d.weight)):
                raise ConfigError(f"domain {d.name}: weight must be finite and >= 0")
            if not d.sources:
                raise ConfigError(f"domain {d.name}: no sources")
            for s in d.sources:
                if s.name in seen:
                    raise ConfigError(f"source {s.name!r} listed in domains {seen[s.name]!r} and {d.name!r}")
                seen[s.name] = d.name
                if s.weight is not None and not (s.weight >= 0 and math.isfinite(s.weight)):
                    raise ConfigError(f"source {s.name}: weight must be finite and >= 0")
                if s.max_epochs is not None and s.max_epochs <= 0:
                    raise ConfigError(f"source {s.name}: max_epochs must be > 0")
            given = [s.weight is not None for s in d.sources if s.enabled]
            if given and any(given) and not all(given):
                raise ConfigError(f"domain {d.name}: set `weight` on all of its sources or on none (none = alpha smoothing)")
        if all(d.weight == 0 for d in self.domains):
            raise ConfigError("all domain weights are 0")
        names = {d.name for d in self.domains}
        prev = 0.0
        for i, st in enumerate(self.stages):
            last = i == len(self.stages) - 1
            if st.until_tokens is None and not last:
                raise ConfigError(f"stage {st.name}: only the last stage may omit until_tokens")
            if st.until_tokens is not None:
                if st.until_tokens <= prev:
                    raise ConfigError("stage until_tokens must be strictly increasing")
                prev = st.until_tokens
            for k in (st.domain_weights or {}):
                if k not in names:
                    raise ConfigError(f"stage {st.name}: unknown domain {k!r}")
            for k in (st.source_weights or {}):
                if k not in seen:
                    raise ConfigError(f"stage {st.name}: unknown source {k!r}")

    # --------------------------------------------------------------- helpers
    def source_cfg(self, name: str) -> tuple[DomainCfg, SourceCfg]:
        for d in self.domains:
            for s in d.sources:
                if s.name == name:
                    return d, s
        raise KeyError(name)

    def enabled_sources(self) -> list[tuple[DomainCfg, SourceCfg]]:
        return [(d, s) for d in self.domains for s in d.sources if s.enabled]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def data_hash(self, template_text: Optional[str] = None) -> str:
        """Hash of everything that changes the produced stream (not paths, micro_batch_size)."""
        d = self.to_dict()
        for k in ("data_root", "cache_dir", "micro_batch_size", "chat_template"):
            d.pop(k)
        d["template"] = hashlib.sha256((template_text or "").encode()).hexdigest()
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def stage_weights(self, stage_idx: int, available_tokens: dict[str, float]) -> dict[str, float]:
        """Probability per source for a stage (0 for unavailable sources); sums to 1 over active sources.

        domain weight x within-domain weight. Within a domain: explicit source weights if configured,
        otherwise tokens**alpha (temperature smoothing). Sources with no available tokens are dropped.
        """
        stage = self.stages[stage_idx] if self.stages else None
        dom_w = {d.name: d.weight for d in self.domains}
        if stage and stage.domain_weights:
            dom_w.update(stage.domain_weights)
        src_over = (stage.source_weights if stage else None) or {}
        probs: dict[str, float] = {}
        for d in self.domains:
            act = [s for s in d.sources if s.enabled and available_tokens.get(s.name, 0) > 0]
            if not act or dom_w[d.name] <= 0:
                continue
            explicit = {s.name: src_over.get(s.name, s.weight) for s in act}
            if all(v is not None for v in explicit.values()):
                w = {k: float(v) for k, v in explicit.items()}
            elif all(v is None for v in explicit.values()):
                w = {s.name: float(available_tokens[s.name]) ** self.alpha for s in act}
            else:
                raise ConfigError(f"domain {d.name}: stage overrides leave only some sources with explicit weights")
            tot = sum(w.values())
            if tot <= 0:
                continue
            for k, v in w.items():
                probs[k] = dom_w[d.name] * v / tot
        z = sum(probs.values())
        return {k: v / z for k, v in probs.items()} if z > 0 else {}

    def max_epochs_for(self, source: str) -> float:
        _, s = self.source_cfg(source)
        return s.max_epochs if s.max_epochs is not None else self.max_epochs
