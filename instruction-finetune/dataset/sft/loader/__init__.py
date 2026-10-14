"""Multi-domain SFT data pipeline: token cache, token-budget mixing, packing, resumable loading.

Typical use::

    python -m dataset.sft.loader.cli compile --config dataset/sft/configs/sft_dataloader.yaml
    python -m dataset.sft.loader.cli stats --config dataset/sft/configs/sft_dataloader.yaml --budget-tokens 1e9
    from dataset.sft.loader import MixConfig, PackedStream
    stream = PackedStream.from_config(MixConfig.from_yaml(path), rank=rank, world_size=world)
"""
from .config import ConfigError, MixConfig  # noqa: F401
from .loader import PackedDataset, PackedStream  # noqa: F401

__all__ = ["ConfigError", "MixConfig", "PackedDataset", "PackedStream"]
