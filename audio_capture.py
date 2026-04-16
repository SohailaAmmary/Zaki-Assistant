"""
zaki/core/audio_capture.py

Handles everything related to audio input:
  - Listing available microphones
  - Streaming raw audio chunks (used by wake word detector)
  - Recording a full utterance after wake word fires
  - Silence detection to auto-stop recording
"""

import time
import wave
import tempfile
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pyaudio
from loguru import logger

from utils.config_loader import get_config


class AudioCapture:
    """
    Manages microphone input for Zaki.

    Usage:
        audio = AudioCapture()
        audio.start_stream()

        # In wake word loop:
        for chunk in audio.stream_chunks():
            wake_word_detector.process(chunk)

        # After wake word fires:
        audio_data = audio.record_utterance()
        # → pass audio_data to ASR
    """

    def __init__(self):
        cfg = get_config()["audio"]
        self.sample_rate: int = cfg["sample_rate"]
        self.channels: int = cfg["channels"]
        self.chunk_size: int = cfg["chunk_size"]
        self.record_seconds: int = cfg["record_seconds"]
        self.silence_threshold: int = cfg["silence_threshold"]
        self.silence_timeout: float = cfg["silence_timeout"]
        self.device_index: Optional[int] = cfg.get("device_index")

        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_stream(self):
        """Open the microphone stream. Call once at startup."""
        self._pa = pyaudio.PyAudio()

        device = self._resolve_device()
        logger.info(f"Opening mic: device_index={device}, rate={self.sample_rate}Hz")

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=device,
            frames_per_buffer=self.chunk_size,
        )
        logger.success("Audio stream started ✓")

    def stop_stream(self):
        """Close the microphone stream and release resources."""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa:
            self._pa.terminate()
            self._pa = None
        logger.info("Audio stream stopped")

    def __enter__(self):
        self.start_stream()
        return self

    def __exit__(self, *args):
        self.stop_stream()

    # ------------------------------------------------------------------
    # Streaming (for wake word detector)
    # ------------------------------------------------------------------

    def stream_chunks(self) -> Generator[np.ndarray, None, None]:
        """
        Yields continuous audio chunks as float32 numpy arrays.
        openWakeWord expects float32 in [-1, 1] at 16kHz.
        """
        if not self._stream:
            raise RuntimeError("Stream not started. Call start_stream() first.")

        while True:
            raw = self._stream.read(self.chunk_size, exception_on_overflow=False)
            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            yield chunk

    # ------------------------------------------------------------------
    # Recording (after wake word fires)
    # ------------------------------------------------------------------

    def record_utterance(self) -> np.ndarray:
        """
        Record audio until silence is detected or max duration is reached.
        Returns a float32 numpy array of the full utterance.

        Called immediately after the wake word is detected.
        """
        logger.info("Recording utterance... (speak now)")

        frames = []
        silent_chunks = 0
        max_chunks = int(self.sample_rate / self.chunk_size * self.record_seconds)
        silence_chunks_limit = int(
            self.silence_timeout * self.sample_rate / self.chunk_size
        )

        for _ in range(max_chunks):
            raw = self._stream.read(self.chunk_size, exception_on_overflow=False)
            frames.append(raw)

            # Check silence
            chunk_np = np.frombuffer(raw, dtype=np.int16)
            rms = self._rms(chunk_np)

            if rms < self.silence_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0  # reset on speech

            # If we've had enough speech followed by enough silence, stop
            if len(frames) > 3 and silent_chunks >= silence_chunks_limit:
                logger.debug(f"Silence detected — stopping after {len(frames)} chunks")
                break

        logger.info(f"Utterance recorded: {len(frames) * self.chunk_size / self.sample_rate:.1f}s")

        # Combine all frames into one float32 array
        raw_bytes = b"".join(frames)
        audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return audio_np

    def save_utterance_to_wav(self, audio: np.ndarray) -> str:
        """
        Save a numpy audio array to a temporary WAV file.
        Returns the file path. Faster-whisper can accept a file path directly.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        int16_audio = (audio * 32768).astype(np.int16)

        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(int16_audio.tobytes())

        logger.debug(f"Utterance saved to: {tmp_path}")
        return tmp_path

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _rms(chunk: np.ndarray) -> float:
        """Root mean square — simple measure of audio energy."""
        return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))

    def _resolve_device(self) -> Optional[int]:
        """Return the device index to use, or None for system default."""
        if self.device_index is not None:
            return self.device_index
        return None

    @staticmethod
    def list_devices():
        """Print all available audio input devices. Run this to find your mic index."""
        pa = pyaudio.PyAudio()
        print("\n=== Available Audio Input Devices ===")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                print(f"  [{i}] {info['name']}  ({int(info['defaultSampleRate'])}Hz)")
        print("=====================================\n")
        pa.terminate()
