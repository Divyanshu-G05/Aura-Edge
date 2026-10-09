import os
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

INPUT_ROOT = "data/raw"
OUTPUT_ROOT = "data/processed"
TARGET_SR = 16000

FOLDERS = [
    "positive",
    "hard_negative_hey",
    "hard_negative_aura",
    "hard_negative_similar",
    "hard_negative_partial",
    "hard_negative_other_wakewords",
    "hard_negative_similar_sentences",
]

AUDIO_EXTS = (".ogg", ".wav", ".mp3", ".m4a", ".opus", ".aac", ".flac")


def trim_silence(audio, silence_thresh=-40, min_silence_len=300, padding_ms=150):
    ranges = detect_nonsilent(
        audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh
    )
    if not ranges:
        return audio  # nothing detected, keep as-is
    start = max(0, ranges[0][0] - padding_ms)
    end = min(len(audio), ranges[-1][1] + padding_ms)
    return audio[start:end]


def convert_file(src_path, dst_path):
    audio = AudioSegment.from_file(src_path)
    audio = audio.set_frame_rate(TARGET_SR)
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)  # 16-bit
    audio = trim_silence(audio)
    audio.export(dst_path, format="wav")


def process_folder(folder):
    src_dir = os.path.join(INPUT_ROOT, folder)
    dst_dir = os.path.join(OUTPUT_ROOT, folder)

    if not os.path.isdir(src_dir):
        print(f"{folder}: source folder missing, skipping")
        return

    os.makedirs(dst_dir, exist_ok=True)

    ok, failed = 0, 0
    for fname in sorted(os.listdir(src_dir)):
        src_path = os.path.join(src_dir, fname)
        if not os.path.isfile(src_path) or not fname.lower().endswith(AUDIO_EXTS):
            continue
        base = os.path.splitext(fname)[0]
        dst_path = os.path.join(dst_dir, f"{base}.wav")
        try:
            convert_file(src_path, dst_path)
            ok += 1
        except Exception as e:
            failed += 1
            print(f"  failed on {fname}: {e}")

    print(f"{folder}: converted {ok}, failed {failed}")


if __name__ == "__main__":
    for folder in FOLDERS:
        process_folder(folder)
