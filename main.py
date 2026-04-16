"""
zaki/main.py

Zaki Voice Assistant — main entry point.
Run with: python main.py

Phase 1: Audio capture + preprocessing pipeline (no ASR yet)
    Each phase adds a new layer on top of this file.
"""

import sys
import signal
from loguru import logger

from utils.config_loader import get_config
from utils.logger import setup_logger
from core.audio_capture import AudioCapture
from core.audio_preprocessor import AudioPreprocessor


def main():
    setup_logger()
    config = get_config()

    logger.info("=" * 50)
    logger.info("  Zaki Voice Assistant — Starting up")
    logger.info("=" * 50)

    audio_capture = AudioCapture()
    preprocessor = AudioPreprocessor()

    # Graceful shutdown on Ctrl+C
    def shutdown(sig, frame):
        logger.info("Shutting down Zaki...")
        audio_capture.stop_stream()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ----------------------------------------------------------------
    # Phase 1: Test audio pipeline
    # ----------------------------------------------------------------
    logger.info("Phase 1 mode: testing audio capture")
    logger.info("Starting mic stream...")

    with AudioCapture() as audio:
        logger.success("Mic is live. Listening for audio...")
        logger.info("Press Ctrl+C to stop. Say something to test recording.")

        # In Phase 1 we just test recording — no wake word yet
        # In Phase 2, this loop will first pass chunks to the wake word detector
        while True:
            logger.info("Waiting 3 seconds then capturing a test utterance...")
            import time
            time.sleep(3)

            raw_audio = audio.record_utterance()
            processed_audio = preprocessor.process(raw_audio)

            if preprocessor.is_valid(processed_audio):
                duration = preprocessor.get_duration(processed_audio)
                logger.success(f"✓ Valid audio captured: {duration:.1f}s")

                # Save to WAV so you can listen to it
                wav_path = audio.save_utterance_to_wav(processed_audio)
                logger.info(f"Saved to: {wav_path}")
                logger.info("(Phase 2 will pass this to wake word detector)")
            else:
                logger.warning("Audio was too quiet or too short — try again")


if __name__ == "__main__":
    # Optional: list mic devices before starting
    if "--list-devices" in sys.argv:
        AudioCapture.list_devices()
        sys.exit(0)

    main()
