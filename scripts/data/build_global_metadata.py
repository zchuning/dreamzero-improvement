"""
Build global metadata for post-training on an on-policy dataset.

When fine-tuning DreamZero on a small on-policy dataset, the action/state
normalization must use the *pretraining* dataset's ranges, not the small
dataset's own stats.  This script reads the pretraining dataset's metadata,
re-indexes the stats to match the on-policy dataset's dimensionality layout,
and writes the result into the global metadata directory that the training
pipeline reads when ``use_global_metadata=true``.

Usage:
    python scripts/data/build_global_metadata.py \
        --pretraining-dataset /path/to/droid_101_success_idlefiltered_train \
        --onpolicy-dataset   /path/to/polaris_food_bussing \
        --embodiment-tag     oxe_droid \
        --metadata-version   0221

    # Then launch training with:
    #   ... use_global_metadata=true ...

What it does:
    1. Reads modality.json from both datasets to learn the sub-key → index
       mappings (e.g. "joint_position" is at 14:21 in DROID, 0:7 in on-policy).
    2. Reads stats.json from the pretraining dataset (28D actions / 14D state).
    3. For each sub-key present in the on-policy modality, extracts the matching
       slice from the pretraining stats and re-packs it into the on-policy layout.
    4. Copies relative_stats_dreamzero.json as-is (already keyed by sub-modality
       name, not by raw index).
    5. Writes modality.json, stats.json, and relative_stats_dreamzero.json into
       groot/vla/data/metadata/{embodiment_tag}/{metadata_version}/.
"""

import argparse
import json
import shutil
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"  Wrote {path}")


def reindex_stats(
    pretraining_stats: dict,
    pretraining_modality: dict,
    onpolicy_modality: dict,
    modality_group: str,  # "state" or "action"
    stats_key: str,       # "observation.state" or "action"
) -> dict:
    """Extract pretraining stats for sub-keys present in the on-policy modality
    and re-pack them into the on-policy index layout."""
    pre_mod = pretraining_modality[modality_group]
    op_mod = onpolicy_modality[modality_group]
    pre_stats = pretraining_stats[stats_key]

    # Total dimensionality of on-policy concatenated vector
    total_dim = max(v["end"] for v in op_mod.values())

    stat_names = list(pre_stats.keys())
    new_stats = {s: [0.0] * total_dim for s in stat_names}

    for subkey, op_slice in op_mod.items():
        if subkey not in pre_mod:
            print(f"  WARNING: sub-key '{subkey}' not in pretraining {modality_group} "
                  f"modality — will use zeros for stats")
            continue

        pre_slice = pre_mod[subkey]
        pre_start, pre_end = pre_slice["start"], pre_slice["end"]
        op_start, op_end = op_slice["start"], op_slice["end"]
        dim = pre_end - pre_start

        assert dim == op_end - op_start, (
            f"Dimension mismatch for {modality_group}.{subkey}: "
            f"pretraining has {dim}, on-policy has {op_end - op_start}"
        )

        for stat_name in stat_names:
            src = pre_stats[stat_name][pre_start:pre_end]
            for i in range(dim):
                new_stats[stat_name][op_start + i] = src[i]

    return new_stats


def main():
    parser = argparse.ArgumentParser(
        description="Build global metadata for post-training with pretraining normalization."
    )
    parser.add_argument(
        "--pretraining-dataset", required=True,
        help="Path to the pretraining dataset (e.g. droid_101_success_idlefiltered_train)",
    )
    parser.add_argument(
        "--onpolicy-dataset", required=True,
        help="Path to the on-policy dataset collected for fine-tuning",
    )
    parser.add_argument(
        "--embodiment-tag", default="oxe_droid",
        help="Embodiment tag used in the training config (default: oxe_droid)",
    )
    parser.add_argument(
        "--metadata-version", default="0221",
        help="Metadata version used in the training config (default: 0221)",
    )
    parser.add_argument(
        "--metadata-root", default=None,
        help="Override the output metadata root directory. "
             "Defaults to groot/vla/data/metadata/ relative to the repo root.",
    )
    args = parser.parse_args()

    pre_path = Path(args.pretraining_dataset)
    op_path = Path(args.onpolicy_dataset)

    # Resolve output directory
    if args.metadata_root:
        out_dir = Path(args.metadata_root) / args.embodiment_tag / args.metadata_version
    else:
        # Auto-detect repo root: walk up from this script
        repo_root = Path(__file__).resolve().parent.parent.parent
        out_dir = repo_root / "groot" / "vla" / "data" / "metadata" / args.embodiment_tag / args.metadata_version

    print(f"Pretraining dataset : {pre_path}")
    print(f"On-policy dataset   : {op_path}")
    print(f"Output directory    : {out_dir}")
    print()

    # Load modality files
    pre_modality = load_json(pre_path / "meta" / "modality.json")
    op_modality = load_json(op_path / "meta" / "modality.json")

    # Load pretraining stats
    pre_stats = load_json(pre_path / "meta" / "stats.json")

    # --- 1. Re-index stats.json ---
    print("Re-indexing stats.json ...")

    new_stats = {}
    # State
    new_stats["observation.state"] = reindex_stats(
        pre_stats, pre_modality, op_modality, "state", "observation.state",
    )
    # Action
    new_stats["action"] = reindex_stats(
        pre_stats, pre_modality, op_modality, "action", "action",
    )

    save_json(new_stats, out_dir / "stats.json")

    # --- 2. Write modality.json (use the on-policy layout) ---
    print("Writing modality.json (on-policy layout) ...")
    save_json(op_modality, out_dir / "modality.json")

    # --- 3. Copy relative stats if they exist ---
    # These are keyed by sub-modality name (e.g. "joint_position"), not raw
    # index, so they don't need re-indexing.
    # Prefer relative_stats_dreamzero.json (flat, 1D per dimension) over
    # relative_stats.json (which may be per-horizon shaped).
    src = pre_path / "meta" / "relative_stats_dreamzero.json"
    if src.exists():
        dst = out_dir / "relative_stats_dreamzero.json"
        shutil.copy2(src, dst)
        print(f"  Copied {src.name} -> {dst}")
    else:
        print("  WARNING: No relative stats found in pretraining dataset")

    # --- Summary ---
    print()
    print("Done. To use these stats during training, add:")
    print("    use_global_metadata=true")
    print("to your training command.")

    # Sanity-check dimensions
    op_state_dim = max(v["end"] for v in op_modality["state"].values())
    op_action_dim = max(v["end"] for v in op_modality["action"].values())
    print()
    print(f"  State dims : {op_state_dim}")
    print(f"  Action dims: {op_action_dim}")
    for group, key in [("state", "observation.state"), ("action", "action")]:
        for subkey in op_modality[group]:
            s, e = op_modality[group][subkey]["start"], op_modality[group][subkey]["end"]
            vals = new_stats[key]["q99"][s:e]
            print(f"  {key}.{subkey} q99[{s}:{e}]: {vals}")


if __name__ == "__main__":
    main()
