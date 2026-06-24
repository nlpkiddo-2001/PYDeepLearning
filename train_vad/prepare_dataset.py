"""
VAD training dataset.

Builds (log-mel spectrogram, frame-level speech/non-speech labels) pairs by
mixing clean LibriSpeech audio with MUSAN noise at random SNRs.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional, Tuple, List, Sequence

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
import soundfile as sf

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

def find_audio_files(root: str | Path) -> List[Path]:
    """Recursively find all audio files in a directory."""
    root = Path(root)
    if not root.exists():
        raise ValueError(f"Directory {root} does not exist")
    files = [p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS]
    if not files:
        raise ValueError(f"No audio files found in {root}")
    return files

def load_audio_mono(path: str | Path, sample_rate: int) -> np.ndarray:
    """
    Load an audio file, resample to the target sample rate, and convert to mono.
    """
    y, sr = sf.read(str(path), dtype="float32", always_2d=False) 
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != sample_rate:
        y_t = torch.from_numpy(y).unsqueeze(0)
        y_t = torchaudio.functional.resample(y_t, orig_freq=sr, new_freq=sample_rate)
        y = y_t.squeeze(0).numpy()
    return y.astype(np.float32)

def fit_length(y: np.ndarray, target_length: int, rng: random.Random) -> np.ndarray:
    """Crop a random window of length target_length from y"""
    if len(y) >= target_length:
        start = rng.randint(0, len(y) - target_length)
        return y[start: start + target_length]

    out = np.zeros(target_length, dtype=np.float32)
    offset = rng.randint(0, target_length - len(y))
    out[offset: offset + len(y)] = y
    return out

def speech_labels_from_energy(
        y_clean: np.ndarray,
        sample_rate : int,
        hop_length: int,
        frame_length: int,
        threshold_db: float = -40.0
) -> np.ndarray:
    """
    Speech labels from clean audio , returns a labels of 0/1 from a clean audio
    """
    pad = frame_length // 2
    y_padded = np.pad(y_clean, pad, mode="reflect")

    n_frames = 1 + (len(y_padded) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        y_padded,
        shape=(n_frames, frame_length),
        strides=(y_padded.strides[0] * hop_length, y_padded.strides[0])
    )
    rms = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    rms_db = 20 * np.log10(rms / (rms.max() + 1e-12) + 1e-12)
    return (rms_db > threshold_db).astype(np.int64)

def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    sp_power = np.mean(speech ** 2) + 1e-12
    no_power = np.mean(noise ** 2) + 1e-12
    target_no_power = sp_power / (10.0 ** (snr_db / 10.0))
    noise_scaled = noise * np.sqrt(target_no_power / no_power)
    mix = speech + noise_scaled

    peak = np.max(np.abs(mix))
    if peak > 0.99:
        mix = mix * (0.99 / peak)
    return mix.astype(np.float32)


class VADDataset(Dataset):

    def __init__(
            self,
            speech_dir: str,
            noise_dir: str,
            music_dir: str,
            clip_seconds: float = 4.0,
            sample_rate: int = 16000,
            n_mels: int = 80,
            n_fft: int = 512,
            hop_ms: float = 10.0,
            frame_ms: float = 25.0,
            snr_range_db: tuple[float, float] = [-5.0, 25.0],
            prob_noise: float = 0.85,
            prob_music: float = 0.25,
            prob_clean: float = 0.15,
            label_threshold_db: float = -40.0,
            epoch_size: int = 50000,
            seed: int = 42
            ):
        
        self.speech_files = find_audio_files(speech_dir)
        self.noise_files = find_audio_files(noise_dir)
        self.music_files = find_audio_files(music_dir) if music_dir else []

        self.clip_seconds = clip_seconds
        self.clip_len = int(self.clip_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_ms = hop_ms
        self.frame_ms = frame_ms
        self.snr_db_range = snr_range_db
        self.prob_noise = prob_noise
        self.prob_music = prob_music
        self.prob_clean = prob_clean
        self.label_threshold_db = label_threshold_db
        self.epoch_size = epoch_size
        self.seed = seed
        self.hop_length = int(hop_ms / 1000.0 * sample_rate)
        self.frame_length = int(frame_ms / 1000.0 * sample_rate)

        #building the mel-spectogram once

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.frame_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
            f_min=0.0,
            f_max=sample_rate / 2.0
        )

        print(
            f"VAD Dataset {len(self.speech_files)} speech files,"
            f"{len(self.noise_files)} noise files",
            f"{len(self.music_files)} music files"
        )

    def __len__(self) -> int:
        return self.epoch_size
    
    def pick_noise(self, rng: random.Random) -> np.ndarray | None:
        if rng.random() >= self.prob_noise:
            return None
        
        use_music = self.music_files and rng.random() < self.prob_music
        pool = self.music_files if use_music else self.noise_files
        noise_path = rng.choice(pool)
        noise = load_audio_mono(noise_path, self.sample_rate)
        return fit_length(noise, self.clip_len, rng)


    def __getitem__(self, index) -> tuple[np.ndarray, np.ndarray]:
        rng = random.Random(hash((self.seed, index)))

        speech_path = rng.choice(self.speech_files)
        speech = load_audio_mono(speech_path, self.sample_rate)
        speech = fit_length(speech, self.clip_len, rng)

        labels_np = speech_labels_from_energy(
            speech,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            frame_length=self.frame_length,
            threshold_db=self.label_threshold_db
        )

        noise = self.pick_noise(rng)
        if noise is not None:
            snr = rng.uniform(*self.snr_db_range)
            mix = mix_at_snr(speech, noise, snr)
        else:
            mix = speech
        
        mix_t = torch.from_numpy(mix).unsqueeze(0)
        mel_t = self.mel(mix_t).squeeze(0)
        log_mel = torch.log(mel_t + 1e-6)

        T = log_mel.shape[-1]
        if len(labels_np) > T:
            labels_np = labels_np[:T]
        elif len(labels_np) < T:
            labels_np = np.pad(labels_np, (0, T - len(labels_np)), mode="edge")
        
        labels = torch.from_numpy(labels_np).long()
        return log_mel, labels
    


        

        
