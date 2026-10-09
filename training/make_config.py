import os
import yaml

FEATURES = "data/features"
NEGATIVES = "data/negative_datasets"


def own_set(name, sampling_weight, penalty_weight, truth, strategy):
    return {
        "features_dir": f"{FEATURES}/{name}",
        "sampling_weight": sampling_weight,
        "penalty_weight": penalty_weight,
        "truth": truth,
        "truncation_strategy": strategy,
        "type": "mmap",
    }


def big_negative(name, sampling_weight, strategy="random"):
    return {
        "features_dir": f"{NEGATIVES}/{name}",
        "sampling_weight": sampling_weight,
        "penalty_weight": 1.0,
        "truth": False,
        "truncation_strategy": strategy,
        "type": "mmap",
    }


features = [
    # positives
    own_set("positive", 2.0, 1.0, True, "truncate_start"),
    own_set("positive_real", 3.0, 1.0, True, "truncate_start"),
    # your hard negatives, penalised harder so near-misses get rejected
    own_set("hard_negative_hey", 2.0, 1.5, False, "random"),
    own_set("hard_negative_aura", 2.0, 1.5, False, "random"),
    own_set("hard_negative_similar", 2.0, 1.5, False, "random"),
    own_set("hard_negative_partial", 2.0, 1.5, False, "random"),
    own_set("hard_negative_partial_real", 2.0, 1.5, False, "random"),
    own_set("hard_negative_other_wakewords", 2.0, 1.5, False, "random"),
    own_set("hard_negative_similar_sentences", 2.0, 1.5, False, "random"),
    # large pre-featurized negatives from Hugging Face
    big_negative("speech", 10.0),
    big_negative("dinner_party", 10.0),
    big_negative("no_speech", 5.0),
    # only used for validation and testing
    {
        "features_dir": f"{NEGATIVES}/dinner_party_eval",
        "sampling_weight": 0.0,
        "penalty_weight": 1.0,
        "truth": False,
        "truncation_strategy": "split",
        "type": "mmap",
    },
]

# warn about missing folders, and skip our own sets that were never built
kept = []
for f in features:
    if os.path.isdir(f["features_dir"]):
        kept.append(f)
    else:
        print(f"MISSING, skipped: {f['features_dir']}")

config = {
    "window_step_ms": 10,
    "train_dir": "trained_models/hey_aura",
    "features": kept,
    "training_steps": [10000],
    "positive_class_weight": [1],
    "negative_class_weight": [20],
    "learning_rates": [0.001],
    "batch_size": 128,
    "time_mask_max_size": [0],
    "time_mask_count": [0],
    "freq_mask_max_size": [0],
    "freq_mask_count": [0],
    "eval_step_interval": 500,
    "clip_duration_ms": 1500,
    "target_minimization": 0.9,
    "minimization_metric": None,
    "maximization_metric": "average_viable_recall",
}

with open("training_parameters.yaml", "w") as f:
    yaml.dump(config, f, sort_keys=False)

print(f"Wrote training_parameters.yaml with {len(kept)} feature sets")
