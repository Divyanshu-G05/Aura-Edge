import os
from mmap_ninja.ragged import RaggedMmap
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration

PROCESSED_ROOT = "data/processed"
FEATURES_ROOT = "data/features"

COMMON = dict(
    augmentation_duration_s=3.2,
    impulse_paths=["mit_rirs"],
    background_paths=["fma_16k"],
    background_min_snr_db=-5,
    background_max_snr_db=10,
    min_jitter_s=0.195,
    max_jitter_s=0.205,
)

# notebook defaults, used for synthetic (Piper) folders
standard_augmenter = Augmentation(
    augmentation_probabilities={
        "SevenBandParametricEQ": 0.1,
        "TanhDistortion": 0.1,
        "PitchShift": 0.1,
        "BandStopFilter": 0.1,
        "AddColorNoise": 0.1,
        "AddBackgroundNoise": 0.75,
        "Gain": 1.0,
        "RIR": 0.5,
    },
    **COMMON,
)

# stronger variation for the small set of real recordings
strong_augmenter = Augmentation(
    augmentation_probabilities={
        "SevenBandParametricEQ": 0.4,
        "TanhDistortion": 0.3,
        "PitchShift": 0.5,
        "BandStopFilter": 0.2,
        "AddColorNoise": 0.3,
        "AddBackgroundNoise": 0.75,
        "Gain": 1.0,
        "RIR": 0.6,
    },
    **COMMON,
)

# folder -> (augmenter, training repeats)
CATEGORIES = {
    "positive": (standard_augmenter, 2),
    "positive_real": (strong_augmenter, 6),
    "hard_negative_hey": (standard_augmenter, 2),
    "hard_negative_aura": (standard_augmenter, 2),
    "hard_negative_similar": (standard_augmenter, 2),
    "hard_negative_partial": (standard_augmenter, 2),
    "hard_negative_partial_real": (strong_augmenter, 6),
    "hard_negative_other_wakewords": (standard_augmenter, 2),
    "hard_negative_similar_sentences": (standard_augmenter, 2),
}


def build_features(category, augmenter, train_repeat):
    input_dir = os.path.join(PROCESSED_ROOT, category)
    if not os.path.isdir(input_dir) or not os.listdir(input_dir):
        print(f"{category}: empty or missing, skipping")
        return

    clips = Clips(
        input_directory=input_dir,
        file_pattern="*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,
    )

    for split in ["training", "validation", "testing"]:
        out_dir = os.path.join(FEATURES_ROOT, category, split)
        os.makedirs(out_dir, exist_ok=True)

        if split == "training":
            split_name, repeat, slide = "train", train_repeat, 10
        elif split == "validation":
            split_name, repeat, slide = "validation", 1, 10
        else:
            split_name, repeat, slide = "test", 1, 1

        spectrograms = SpectrogramGeneration(
            clips=clips, augmenter=augmenter, slide_frames=slide, step_ms=10
        )
        RaggedMmap.from_generator(
            out_dir=os.path.join(out_dir, "wakeword_mmap"),
            sample_generator=spectrograms.spectrogram_generator(
                split=split_name, repeat=repeat
            ),
            batch_size=100,
            verbose=True,
        )
    print(f"{category}: done")


if __name__ == "__main__":
    for category, (augmenter, repeat) in CATEGORIES.items():
        build_features(category, augmenter, repeat)

