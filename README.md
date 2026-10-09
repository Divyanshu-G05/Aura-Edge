# Aura: Offline Wake-Word Voice Activator for Edge Devices

## Summary

Aura is an offline "Hey Aura" wake-word detector that runs entirely on an ESP32-C3 microcontroller. A 61 KB custom keyword-spotting model listens through an I2S microphone and detects the wake word on the device in about 5 ms per inference step. Only after detection does the device stream the following speech over Wi-Fi to a local backend (speech-to-text, local language model, text-to-speech) and play the reply through its speaker. Audio is not sent anywhere until the wake word is heard.


## How the prototype works

1. The ESP32-C3 listens continuously through an INMP441 I2S microphone (16 kHz, mono).
2. A 40-channel spectrogram frontend feeds a streaming TFLite Micro wake-word model (trained with microWakeWord).
3. When "Hey Aura" is detected, the device gives LED and beep feedback and opens a TCP stream (port 8765) to the backend.
4. The backend (laptop on the same network) transcribes the speech with faster-whisper, generates a reply with a local language model through Ollama, and converts it to speech with Piper.
5. The reply audio is streamed back and played through a MAX98357 amplifier and speaker.

In the prototype, the ESP32 and the laptop are both connected to a phone hotspot that acts as the local network, with mobile data off, so everything runs offline. The same backend can later forward text to online AI models.

## Key numbers (measured on the prototype)

| Metric | Value |
| --- | --- |
| Wake-word model size | about 61 KB (62,304 bytes) |
| Tensor arena (RAM) | 70 KB |
| Inference time | about 5.3 ms per 30 ms step |
| Firmware size | about 406 KB (1 MB partition) |

Full measurements and test conditions are in the Technical Documentation.

## Repository layout

The repository is organized as follows:

- `firmware/`: ESP-IDF project for the ESP32-C3
- `backend/`: speech-to-text, LLM and text-to-speech server
- `training/`: wake-word dataset scripts and training config
- `docs/`: documentation

## How to run

**Backend**

1. Install the Python dependencies: `pip install -r requirements.txt`
2. Start Ollama and pull the model: `ollama pull qwen3:8b`
3. Start the server: `python server.py` (listens for the ESP32 on TCP port 8765)

**Firmware**

1. Set the Wi-Fi name, password and the laptop's IP address in the project configuration (`idf.py menuconfig`)
2. Build and flash: `idf.py set-target esp32c3`, then `idf.py build flash monitor`
3. Say "Hey Aura" and speak a command.

## Known limitations

Accuracy is limited by the small amount of real speech data used so far, and inference CPU duty is still above our target. See the Roadmap and Limitations document.

## License

MIT License.
