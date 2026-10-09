import subprocess
import os
import random
from pydub import AudioSegment

OUTPUT_ROOT = "data/raw"
MODELS_DIR = "piper-models"  # folder containing the voice .onnx and .onnx.json files

VOICES = [
    "en_US-lessac-medium",
    "en_US-amy-medium",
    "en_US-ryan-medium",
]

POSITIVE_VOICES = [
    "en_US-lessac-medium",
    "en_US-amy-medium",
    "en_US-ryan-medium",
    "en_US-kristin-medium",
    "en_GB-alan-medium",
    "en_GB-jenny_dioco-medium",
]

POSITIVE_REPEATS_PER_VOICE = 15  # raise this to get more positives

VARIATIONS = [
    {"length_scale": 1.0, "noise_scale": 0.667},
    {"length_scale": 1.15, "noise_scale": 0.667},
    {"length_scale": 0.9, "noise_scale": 0.8},
    {"length_scale": 1.0, "noise_scale": 0.9},
]

# single words are only slowed down, never sped up
SINGLE_WORD_VARIATIONS = [
    {"length_scale": 1.1, "noise_scale": 0.667},
    {"length_scale": 1.3, "noise_scale": 0.667},
    {"length_scale": 1.2, "noise_scale": 0.8},
    {"length_scale": 1.4, "noise_scale": 0.75},
]

SINGLE_WORD_FOLDERS = ["hard_negative_hey", "hard_negative_aura", "hard_negative_similar"]

CATEGORIES = {
    "positive": ["Hey Aura"],
    "hard_negative_hey": ["Hey"],
    "hard_negative_aura": ["Aura"],
    "hard_negative_similar": ["Laura", "Nora", "Sara", "Cora", "Ora", "Sora"],
    "hard_negative_partial": ["Hey Laura", "Hey Nora", "Hey Sara", "Hey Cora", "Hey Sora", "Aura Hey", "Hey"],
    "hard_negative_other_wakewords": [
        "Hey Google", "Okay Google", "Hey Siri", "Alexa",
        "Hey Assistant", "Computer", "Hey Cortana",
    ],
    "hard_negative_similar_sentences": [
        "I saw a beautiful aura around the sunset",
        "Hey, are you coming later",
        "Hey Laura, can you check this",
        "Nora said she'll be late",
    ],
}


def get_voices(folder):
    return POSITIVE_VOICES if folder == "positive" else VOICES


def get_variations(folder):
    return SINGLE_WORD_VARIATIONS if folder in SINGLE_WORD_FOLDERS else VARIATIONS


def random_variation():
    return {
        "length_scale": round(random.uniform(0.85, 1.3), 2),
        "noise_scale": round(random.uniform(0.6, 0.9), 2),
    }


def synthesize(text, voice, variation, output_file):
    cmd = [
        "python", "-m", "piper",
        "--model", voice,
        "--data-dir", MODELS_DIR,
        "--output_file", output_file,
        "--length-scale", str(variation["length_scale"]),
        "--noise-scale", str(variation["noise_scale"]),
    ]
    subprocess.run(cmd, input=text.encode("utf-8"), check=True)


def pad_short_clip(path, min_duration_ms=600, pad_ms=200):
    audio = AudioSegment.from_wav(path)
    if len(audio) < min_duration_ms:
        silence = AudioSegment.silent(duration=pad_ms)
        audio = silence + audio + silence
        audio.export(path, format="wav")


def generate_positive():
    out_dir = os.path.join(OUTPUT_ROOT, "positive")
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    for voice in POSITIVE_VOICES:
        for i in range(POSITIVE_REPEATS_PER_VOICE):
            out_path = os.path.join(out_dir, f"hey_aura_{voice}_{i}.wav")
            try:
                synthesize("Hey Aura", voice, random_variation(), out_path)
                count += 1
            except subprocess.CalledProcessError as e:
                print(f"Failed: Hey Aura / {voice} / rep {i} -> {e}")
    print(f"positive: generated {count} files")


def generate_negatives():
    for folder, phrases in CATEGORIES.items():
        if folder == "positive":
            continue

        out_dir = os.path.join(OUTPUT_ROOT, folder)
        os.makedirs(out_dir, exist_ok=True)

        voices = get_voices(folder)
        variations = get_variations(folder)

        count = 0
        for phrase in phrases:
            for voice in voices:
                for v_idx, variation in enumerate(variations):
                    safe_phrase = phrase.lower().replace(" ", "_").replace(",", "").replace("'", "")
                    out_path = os.path.join(out_dir, f"{safe_phrase}_{voice}_{v_idx}.wav")
                    try:
                        synthesize(phrase, voice, variation, out_path)
                        if folder in SINGLE_WORD_FOLDERS:
                            pad_short_clip(out_path)
                        count += 1
                    except subprocess.CalledProcessError as e:
                        print(f"Failed: {phrase} / {voice} / variation {v_idx} -> {e}")

        print(f"{folder}: generated {count} files")


if __name__ == "__main__":
    generate_positive()
    generate_negatives()
