"""Pack release assets for GitHub Releases distribution.

Creates:
- release_assets/toki_pona_corpus.tar.gz (clean, deduplicated dataset splits)
- release_assets/best_models.tar.gz (best model checkpoints, tokenizers, evaluation metrics)
"""

import os
import tarfile
import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = ROOT_DIR / "release_assets"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RUNS_DIR = ROOT_DIR / "results" / "metrics" / "runs"
TOKENIZERS_DIR = ROOT_DIR / "results" / "metrics" / "tokenizers"
METRICS_DIR = ROOT_DIR / "results" / "metrics"


def pack_dataset():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ASSETS_DIR / "toki_pona_corpus.tar.gz"
    print(f"Packaging dataset to {out_path}...")
    
    files_to_pack = ["train.jsonl", "val.jsonl", "test.jsonl", "metadata.json"]
    with tarfile.open(out_path, "w:gz") as tar:
        for f in files_to_pack:
            p = DATA_PROCESSED_DIR / f
            if p.exists():
                tar.add(p, arcname=f"toki_pona_corpus/{f}")
                print(f"  + Added {f} ({p.stat().st_size:,} bytes)")
            else:
                print(f"  ! Warning: {p} not found")
                
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"-> Created {out_path.name}: {size_mb:.2f} MB\n")


def pack_best_models():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ASSETS_DIR / "best_models.tar.gz"
    print(f"Packaging best models to {out_path}...")

    # Best run per tokenizer strategy (mini 75% data)
    best_runs = [
        "exp_010_d75_mini_char_s42",
        "exp_011_d75_mini_bpe_s42",
        "exp_012_d75_mini_word_s42"
    ]

    with tarfile.open(out_path, "w:gz") as tar:
        # Add best model checkpoints
        for r in best_runs:
            pt = RUNS_DIR / r / "best_model.pt"
            if pt.exists():
                tar.add(pt, arcname=f"best_models/{r}/best_model.pt")
                print(f"  + Added checkpoint {r}/best_model.pt ({pt.stat().st_size:,} bytes)")

        # Add tokenizer configs
        if TOKENIZERS_DIR.exists():
            for f in TOKENIZERS_DIR.iterdir():
                if f.is_file():
                    tar.add(f, arcname=f"best_models/tokenizers/{f.name}")
                    print(f"  + Added tokenizer {f.name}")

        # Add scaling laws and results table
        for m in ["scaling_laws.json", "results.csv", "anova_table.csv"]:
            mp = METRICS_DIR / m
            if mp.exists():
                tar.add(mp, arcname=f"best_models/metrics/{m}")
                print(f"  + Added metric {m}")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"-> Created {out_path.name}: {size_mb:.2f} MB\n")


def main():
    pack_dataset()
    pack_best_models()
    print("All release assets packed successfully!")


if __name__ == "__main__":
    main()
