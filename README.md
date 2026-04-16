# Zaki — Arabic/English Voice Assistant

A modular voice-controlled home assistant supporting both Arabic and English.

## Project Structure

```
zaki/
├── main.py                  # Entry point
├── requirements.txt
├── .env.example             # Copy to .env and add your keys
├── config/
│   └── config.yaml          # All settings (mic, ASR, devices, MQTT...)
├── core/
│   ├── audio_capture.py     # Microphone input & recording
│   ├── audio_preprocessor.py # Noise filtering & normalization
│   ├── wake_word.py         # (Phase 2) "Hey Zaki" detection
│   ├── asr.py               # (Phase 3) Speech → text
│   ├── nlu.py               # (Phase 4) Text → intent + entities
│   └── tts.py               # (Phase 5) Voice feedback
├── devices/
│   └── device_controller.py # (Phase 5) MQTT / GPIO control
├── utils/
│   ├── config_loader.py     # Loads config.yaml + .env
│   └── logger.py            # Loguru setup
└── tests/
    └── test_audio_preprocessor.py
```

## Build Phases

| Phase | What gets built |
|-------|----------------|
| ✅ 1 | Audio capture, preprocessing, project structure |
| 🔜 2 | Wake word detection ("Hey Zaki") |
| 🔜 3 | Speech recognition — Arabic + English (Whisper) |
| 🔜 4 | Intent & entity recognition (NLU) |
| 🔜 5 | Device control + voice feedback |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy env file
cp .env.example .env

# 3. List available microphones
python main.py --list-devices

# 4. Set your mic index in config/config.yaml (or leave null for default)

# 5. Run
python main.py
```

## Supported Commands (Phase 4+)

| English | Arabic |
|---------|--------|
| "room 1 on" | "افتح نور الغرفة" |
| "turn off the living room light" | "أطفئ نور المعيشة" |
| "bedroom light off" | "أطفئ نور غرفة النوم" |

## Configuration

All settings live in `config/config.yaml`. Key sections:

- **audio** — sample rate, mic index, silence detection
- **wake_word** — detection threshold
- **asr** — Whisper model size, local vs cloud
- **devices** — room names, aliases in Arabic & English
- **mqtt** — broker settings for ESP32 integration

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```
