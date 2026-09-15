"""
Interactive Inference & Grammatical Evaluation CLI for trained Toki Pona SLMs.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.models.configs import TransformerConfig
from src.models.transformer import TokiPonaTransformer
from src.tokenization.char_tokenizer import CharTokenizer
from src.tokenization.word_tokenizer import WordTokenizer
from src.tokenization.bpe_tokenizer import BPETokenizer
from src.evaluation.metrics import score_sentence
from src.evaluation.grammar_suite import get_default_test_suite, evaluate_grammar_suite


def load_model_and_tokenizer(run_name: str, project_root: Path):
    metrics_file = project_root / "results" / "metrics" / "results.csv"
    models_dir = project_root / "results" / "models"

    import pandas as pd
    df = pd.read_csv(metrics_file)
    matches = df[df["exp_id"] == run_name]
    if matches.empty:
        raise ValueError(f"Run {run_name} not found in {metrics_file}")
    row = matches.iloc[0]

    tok_type = row["tokenizer"]
    model_size = row["model_size"]

    from src.tokenization import load_tokenizer

    # 1. Load tokenizer: prefer pre-saved tokenizer, fallback to training on sample texts
    saved_tok_path = project_root / "results" / "metrics" / "tokenizers" / f"{tok_type}_tok.json"
    if saved_tok_path.exists():
        tokenizer = load_tokenizer(saved_tok_path)
    else:
        data_dir = project_root / "data" / "processed"
        train_file = data_dir / "train.jsonl"
        sample_texts = []
        if train_file.exists():
            with open(train_file, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if idx >= 5000: break
                    sample_texts.append(json.loads(line)["text"])

        if tok_type == "char":
            tokenizer = CharTokenizer(sample_texts)
        elif tok_type == "word":
            tokenizer = WordTokenizer(sample_texts)
        else:
            tokenizer = BPETokenizer(sample_texts, target_vocab_size=80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TransformerConfig.get_preset(model_size, vocab_size=tokenizer.vocab_size)
    model = TokiPonaTransformer(cfg).to(device)

    # 2. Find model checkpoint across standard locations
    candidate_paths = [
        project_root / "results" / "metrics" / "runs" / run_name / "best_model.pt",
        project_root / "results" / "models" / run_name / "best_model.pt",
        project_root / "results" / "models" / "best_model.pt",
    ]
    weights_path = next((p for p in candidate_paths if p.exists()), None)

    if weights_path is not None:
        try:
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[Loaded] Checkpoint {weights_path} onto {device}")
    else:
        print(f"[Notice] Model weights for {run_name} not found locally. Running with architecture initialization.")

    model.eval()
    return model, tokenizer, device, row


@torch.no_grad()
def generate_text(
    model: TokiPonaTransformer,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 30,
    temperature: float = 0.7,
    top_k: int = 20,
) -> str:
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    if tokens and tokens[-1] == tokenizer.eos_token_id:
        tokens = tokens[:-1]

    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        x_cond = input_ids if input_ids.size(1) <= model.config.max_seq_len else input_ids[:, -model.config.max_seq_len:]
        logits, _ = model(x_cond)
        next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

        if top_k > 0:
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

        input_ids = torch.cat((input_ids, next_token), dim=1)

    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toki Pona SLM Inference")
    parser.add_argument("--run", type=str, default="exp_012_d75_mini_word_s42")
    parser.add_argument("--prompt", type=str, default="jan pona mi li")
    args = parser.parse_args()

    model, tokenizer, device, meta = load_model_and_tokenizer(args.run, root)
    print(f"\nID: {meta['exp_id']} | Tokenizer: {meta['tokenizer'].upper()} | Val BPC: {meta['val_bpc']}")
    print(f"Prompt: '{args.prompt}'")
    for i in range(3):
        out = generate_text(model, tokenizer, args.prompt, device, max_new_tokens=20)
        print(f"  Sample {i+1}: {out}")
