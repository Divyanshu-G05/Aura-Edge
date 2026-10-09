#!/usr/bin/env python3
"""
Aura backend for the ESP32-C3 satellite.

Pipeline (all overlapped where possible):
  ESP audio chunks -> server-side end-of-speech detection -> STOP signal to ESP
  -> faster-whisper (starts the instant speech ends)
  -> Ollama streaming -> split into sentences -> Piper per sentence
  -> each sentence's audio is sent to the ESP as soon as it is ready.

Wire protocol (big-endian u32 length prefixes, matches main.cpp):
    client -> server : repeated [len][PCM16LE mono 16k], terminated by [len=0]
    server -> client : (during send phase) one [0xFFFFFFFF] = STOP
                       then repeated [len][PCM16LE mono 16k], terminated by [len=0]

Install:
    pip install faster-whisper requests numpy scipy piper-tts
Run:
    python server.py
"""

import asyncio
import json
import math
import re
import struct
import subprocess
import time

import numpy as np
import requests
from faster_whisper import WhisperModel

try:
    from scipy.signal import resample_poly
except ImportError:
    resample_poly = None
try:
    from piper import PiperVoice
except ImportError:
    PiperVoice = None

# ---------------- config ----------------
HOST = "0.0.0.0"
PORT = 8765
TARGET_RATE = 16000                  # must match firmware SAMPLE_RATE

WHISPER_PATH = "./whisper-small"     # CTranslate2 model dir, or a size name like "small.en"
WHISPER_DEVICE = "cpu"               # keep VRAM free for the LLM
WHISPER_COMPUTE = "int8"

# Smaller model = much faster. qwen3:8b (~5.2GB) barely fits a 6GB GPU and spills to CPU.
# Good options: "qwen3:4b" (~2.6GB), "llama3.2:3b", "qwen2.5:3b-instruct", "qwen3:1.7b"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_SYSTEM_PROMPT = ("You are Aura, a voice assistant. Answer in one to three short "
                        "plain sentences. No markdown, no lists, no emojis. /no_think")
OLLAMA_OPTIONS = {"num_predict": 150, "temperature": 0.6, "num_ctx": 2048}

PIPER_MODEL = "en_GB-alba-medium.onnx"
PIPER_BIN = "piper"                  # only used if the Python API is unavailable
PIPER_FALLBACK_RATE = 22050

# ---- end-of-speech detection (tune these here, no reflashing needed) ----
VAD_FRAME = 320                      # 20 ms @16k
THR_MULT = 3.0                       # speech threshold = noise_floor * THR_MULT ...
THR_ADD = 120.0                      # ... or noise_floor + THR_ADD, whichever is larger
MIN_THR = 180.0                      # ... but never below this (RMS, int16 units)
ONSET_FRAMES = 3                     # consecutive loud frames (60 ms) to count as speech
END_SILENCE_MS = 700                 # trailing silence that ends the utterance
MIN_SPEECH_MS = 250
NO_SPEECH_TIMEOUT_MS = 5000          # user never spoke after the beep
MAX_UTTER_MS = 10000
WARMUP_FRAMES = 8                    # first 160 ms used to estimate the noise floor

SEND_CHUNK_BYTES = 3200              # 100 ms per reply frame
MIN_UTTERANCE_BYTES = 3200
RECV_TIMEOUT_S = 15

STOP_BYTES = struct.pack(">I", 0xFFFFFFFF)

whisper_model = WhisperModel(WHISPER_PATH, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
voice = PiperVoice.load(PIPER_MODEL) if PiperVoice else None


# ---------------- framing ----------------
async def read_frame(reader: asyncio.StreamReader):
    (n,) = struct.unpack(">I", await reader.readexactly(4))
    if n == 0:
        return None
    return await reader.readexactly(n)


async def write_frame(writer: asyncio.StreamWriter, data: bytes) -> None:
    writer.write(struct.pack(">I", len(data)))
    if data:
        writer.write(data)
    await writer.drain()


# ---------------- end-of-speech detector ----------------
class EndpointVAD:
    """Adaptive energy VAD. Returns a reason string when the utterance is over."""

    def __init__(self):
        self.buf = np.zeros(0, dtype=np.int16)
        self.floor = None
        self.n = 0
        self.run = 0
        self.in_speech = False
        self.speech_ms = 0
        self.silence_ms = 0
        self.total_ms = 0
        self.thr = MIN_THR

    def feed(self, pcm_bytes: bytes):
        x = np.concatenate([self.buf, np.frombuffer(pcm_bytes, dtype=np.int16)])
        nfr = len(x) // VAD_FRAME
        for i in range(nfr):
            f = x[i * VAD_FRAME:(i + 1) * VAD_FRAME].astype(np.float32)
            e = float(np.sqrt(np.mean(f * f)))
            self.n += 1
            self.total_ms += 20

            if self.n <= WARMUP_FRAMES:                       # calibrate noise floor
                self.floor = e if self.floor is None else min(self.floor, e)
                continue

            self.thr = max(self.floor * THR_MULT, self.floor + THR_ADD, MIN_THR)
            voiced = e > self.thr

            if voiced:
                self.run += 1
                self.silence_ms = 0
                if self.run >= ONSET_FRAMES:
                    self.in_speech = True
                if self.in_speech:
                    self.speech_ms += 20
            else:
                self.run = 0
                if self.in_speech:
                    self.silence_ms += 20
                else:                                          # track noise floor while idle
                    self.floor = self.floor * 0.95 + e * 0.05 if e > self.floor else e

            if self.in_speech and self.speech_ms >= MIN_SPEECH_MS and self.silence_ms >= END_SILENCE_MS:
                self.buf = x[nfr * VAD_FRAME:]
                return "silence"
            if not self.in_speech and self.total_ms >= NO_SPEECH_TIMEOUT_MS:
                self.buf = x[nfr * VAD_FRAME:]
                return "no_speech"
            if self.total_ms >= MAX_UTTER_MS:
                self.buf = x[nfr * VAD_FRAME:]
                return "timeout"
        self.buf = x[nfr * VAD_FRAME:]
        return None


# ---------------- STT ----------------
def transcribe(pcm_bytes: bytes) -> str:
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = whisper_model.transcribe(
        audio, language="en", beam_size=1,
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=300),
        condition_on_previous_text=False, without_timestamps=True,
    )
    return " ".join(s.text for s in segments).strip()


# ---------------- LLM (streamed, yields whole sentences) ----------------
SENT_END = re.compile(r'[.!?]["\')\]]?\s')


def clean_for_tts(s: str) -> str:
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"[*_#`~>|]", "", s)
    s = re.sub(r"[^\x20-\x7E]", "", s)
    return s.strip()


def llm_sentences(text: str):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": OLLAMA_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "stream": True,
        "think": False,
        "keep_alive": "30m",
        "options": OLLAMA_OPTIONS,
    }
    buf, in_think = "", False
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=(5, 60)) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            d = json.loads(line)
            piece = d.get("message", {}).get("content", "")
            # safety net if an older Ollama leaks <think> blocks into content
            if "<think>" in piece:
                in_think = True
                piece = piece.split("<think>")[0]
            if in_think:
                if "</think>" in piece:
                    in_think = False
                    piece = piece.split("</think>", 1)[1]
                else:
                    piece = ""
            buf += piece
            while True:
                m = SENT_END.search(buf)
                if not m:
                    break
                sent, buf = buf[:m.end()], buf[m.end():]
                sent = clean_for_tts(sent)
                if sent:
                    yield sent
            if d.get("done"):
                break
    tail = clean_for_tts(buf)
    if tail:
        yield tail


# ---------------- TTS ----------------
def to_target_rate(pcm: np.ndarray, src: int) -> np.ndarray:
    if src == TARGET_RATE:
        return pcm
    if resample_poly is not None:
        g = math.gcd(src, TARGET_RATE)
        y = resample_poly(pcm.astype(np.float32), TARGET_RATE // g, src // g)
        return np.clip(y, -32768, 32767).astype(np.int16)
    n = int(len(pcm) * TARGET_RATE / src)
    idx = np.linspace(0, len(pcm) - 1, n)
    return np.interp(idx, np.arange(len(pcm)), pcm).astype(np.int16)


def synthesize(text: str) -> bytes:
    """One sentence -> PCM16LE mono @TARGET_RATE."""
    raw, rate = None, None
    if voice is not None:
        try:
            if hasattr(voice, "synthesize_stream_raw"):            # piper-tts 1.2.x
                raw = b"".join(voice.synthesize_stream_raw(text))
            else:                                                  # piper-tts 1.3+
                raw = b"".join(c.audio_int16_bytes for c in voice.synthesize(text))
            rate = voice.config.sample_rate
        except Exception as e:
            print(f"[!] piper python API failed ({e!r}), falling back to CLI")
            raw = None
    if raw is None:
        proc = subprocess.run([PIPER_BIN, "--model", PIPER_MODEL, "--output-raw"],
                              input=text.encode("utf-8"), capture_output=True, check=True)
        raw, rate = proc.stdout, PIPER_FALLBACK_RATE
    pcm = np.frombuffer(raw, dtype=np.int16)
    return to_target_rate(pcm, rate).tobytes()


# ---------------- reply pipeline ----------------
async def stream_reply(text: str, writer: asyncio.StreamWriter) -> None:
    """LLM tokens -> sentences -> TTS -> socket, overlapped."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def worker():
        try:
            for sentence in llm_sentences(text):
                print(f"[LLM] {sentence!r}")
                loop.call_soon_threadsafe(q.put_nowait, synthesize(sentence))
        except Exception as e:
            print(f"[!] reply worker error: {e!r}")
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    t0 = time.monotonic()
    fut = loop.run_in_executor(None, worker)
    first = True
    while True:
        pcm = await q.get()
        if pcm is None:
            break
        if first:
            print(f"[i] first audio ready {time.monotonic() - t0:.2f}s after STT")
            first = False
        for i in range(0, len(pcm), SEND_CHUNK_BYTES):
            await write_frame(writer, pcm[i:i + SEND_CHUNK_BYTES])
    await fut
    await write_frame(writer, b"")
    print("[i] reply sent")


# ---------------- connection handler ----------------
async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    peer = writer.get_extra_info("peername")
    print(f"[+] {peer} connected")
    try:
        vad = EndpointVAD()
        chunks = []
        stt_task = None
        reason = None

        while True:
            frame = await asyncio.wait_for(read_frame(reader), RECV_TIMEOUT_S)
            if frame is None:                       # ESP ended the utterance itself
                break
            chunks.append(frame)
            if reason is None:
                reason = vad.feed(frame)
                if reason:
                    print(f"[VAD] {reason}: {vad.total_ms} ms total, {vad.speech_ms} ms speech, "
                          f"floor={vad.floor:.0f} thr={vad.thr:.0f}")
                    writer.write(STOP_BYTES)        # tell the ESP to stop streaming
                    await writer.drain()
                    if reason != "no_speech":       # start STT now, while the ESP drains its last chunk
                        stt_task = asyncio.create_task(
                            asyncio.to_thread(transcribe, b"".join(chunks)))

        pcm_bytes = b"".join(chunks)
        print(f"[i] received {len(pcm_bytes)} bytes ({len(pcm_bytes) / 2 / TARGET_RATE:.2f}s)")

        if reason == "no_speech" or len(pcm_bytes) < MIN_UTTERANCE_BYTES:
            print("[i] no speech / too short, skipping pipeline")
            await write_frame(writer, b"")
            return

        t0 = time.monotonic()
        text = await (stt_task if stt_task else asyncio.to_thread(transcribe, pcm_bytes))
        print(f"[STT] {text!r} ({time.monotonic() - t0:.2f}s)")
        if not text:
            await write_frame(writer, b"")
            return

        await stream_reply(text, writer)

    except Exception as e:
        print(f"[!] error handling {peer}: {e!r}")
        try:
            await write_frame(writer, b"")
        except Exception:
            pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        print(f"[-] {peer} disconnected")


def warm_up():
    """Load everything once so the first real request isn't slow."""
    try:
        transcribe(np.zeros(TARGET_RATE, dtype=np.int16).tobytes())
        synthesize("Ready.")
        for _ in llm_sentences("Say ok."):
            pass
        print("[i] warm-up done")
    except Exception as e:
        print(f"[!] warm-up failed: {e!r}")


async def main():
    await asyncio.to_thread(warm_up)
    server = await asyncio.start_server(handle_client, HOST, PORT)
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    print(f"Aura backend listening on {addrs}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())