"""
zaki/core/audio_preprocessor.py

Cleans and normalizes raw audio before passing to ASR.
Whisper is sensitive to audio quality — this module helps.
"""

import numpy as np
from scipy import signal
from loguru import logger
from utils.config_loader import get_config


class AudioPreprocessor:
    """
    Applies a chain of audio processing steps:
      1. Normalize amplitude
      2. High-pass filter (removes mic rumble below 80Hz)
      3. Validate audio quality (not too short, not silent)
    """

    def __init__(self):
        cfg = get_config()["audio"]
        self.sample_rate = cfg["sample_rate"]
        self.silence_threshold = cfg["silence_threshold"] / 32768.0  # normalize to float scale

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline.
        Input:  float32 numpy array in [-1, 1]
        Output: cleaned float32 numpy array, same range
        """
        audio = self._normalize(audio)
        audio = self._highpass_filter(audio)
        audio = np.clip(audio, -1.0, 1.0)
        return audio

    def is_valid(self, audio: np.ndarray) -> bool:
        """
        Returns False if the audio is too short or too quiet to be useful.
        Prevents sending silence to Whisper (saves time and API cost).
        """
        min_duration = 0.3  # seconds
        min_samples = int(min_duration * self.sample_rate)

        if len(audio) < min_samples:
            logger.warning(f"Audio too short: {len(audio)} samples (min {min_samples})")
            return False

        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < self.silence_threshold:
            logger.warning(f"Audio too quiet: RMS={rms:.4f} (threshold={self.silence_threshold:.4f})")
            return False

        return True

    def get_duration(self, audio: np.ndarray) -> float:
        """Return duration in seconds."""
        return len(audio) / self.sample_rate

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize peak amplitude to 0.95 to avoid clipping."""
        peak = np.max(np.abs(audio))
        if peak > 0.001:
            audio = audio / peak * 0.95
        return audio

    def _highpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply a 4th-order Butterworth high-pass filter at 80Hz.
        Removes low-frequency rumble from fans, HVAC, mic handling noise.
        """
        cutoff = 80  # Hz
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff / nyquist

        b, a = signal.butter(4, normalized_cutoff, btype="high", analog=False)
        filtered = signal.filtfilt(b, a, audio)
        return filtered.astype(np.float32)
