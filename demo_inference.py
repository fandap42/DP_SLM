"""
Interactive Inference & Grammatical Evaluation CLI for trained Toki Pona SLMs.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F

from slm_tokipona.models.configs import TransformerConfig
from slm_tokipona.models.transformer import TokiPonaTransformer
from slm_tokipona.tokenizers.char_tokenizer import CharTokenizer
from slm_tokipona.tokenizers.word_tokenizer import WordTokenizer
from slm_tokipona.tokenizers.bpe_tokenizer import BPETokenizer
from slm_tokipona.evaluation.metrics import score_sentence
from slm_tokipona.evaluation.grammar_suite import get_default_test_suite, evaluate_grammar_suite


def load_model_and_tokenizer(run_dir: Path | str):
    run_dir = Path(run_dir)
    results_csv = run_dir.parent.parent / "results.csv"

    # Find experiment metadata from results.csv
    import pandas as pd
    df = pd.read_csv(results_csv)
    exp_name = run_dir.name
    row = df[df["exp_id"] == exp_name].iloc[0]

    tok_type = row["tokenizer"]
    model_size = row["model_size"]

    # Load tokenizer
    data_dir = Path("data/processed")
    sample_texts = []
    with open(data_dir / "train.jsonl", "r", encoding="utf-8") as f:
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

    weights_path = run_dir / "best_model.pt"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"[Loaded] Checkpoint {weights_path} onto {device}")
    else:
        print(f"[Warning] No weights found at {weights_path}, using uninitialized model.")

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
        # Truncate to context window if needed
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


def main():
    parser = argparse.ArgumentParser(description="Toki Pona SLM Inference & Grammar Evaluation")
    parser.add_argument("--run", type=str, default="exp_012_d75_mini_word_s42", help="Run folder name under experiments/outputs/runs/")
    parser.add_argument("--prompt", type=str, default="jan pona mi li", help="Prompt to continue")
    parser.add_argument("--eval-grammar", action="store_true", default=True, help="Run grammar test suite")
    args = parser.parse_args()

    run_path = Path("experiments/outputs/runs") / args.run
    if not run_path.exists():
        runs = list(Path("experiments/outputs/runs").glob("exp_*"))
        if runs:
            run_path = runs[-1]
            print(f"[Notice] Selected latest available run: {run_path.name}")
        else:
            print(f"[Error] No experiment runs found in experiments/outputs/runs/")
            return

    model, tokenizer, device, meta = load_model_and_tokenizer(run_path)
    print(f"\n--- Experiment Metadata ---")
    print(f"ID: {meta['exp_id']} | Tokenizer: {meta['tokenizer'].upper()} | Model: {meta['model_size']} | Params: {meta['total_params']:,}")
    print(f"Val BPC: {meta['val_bpc']} | Grammar Acc: {meta['grammar_acc']*100:.1f}%\n")

    # 1. Text Generation
    print(f"--- Autoregressive Generation ---")
    print(f"Prompt: '{args.prompt}'")
    for i in range(3):
        completion = generate_text(model, tokenizer, args.prompt, device, max_new_tokens=25, temperature=0.7)
        print(f"  Sample {i+1}: {completion}")

    # 2. Grammar Suite Evaluation
    if args.eval_grammar:
        print(f"\n--- Grammar Suite Detailed Breakdown ---")
        res = evaluate_grammar_suite(model, tokenizer, device)
        print(f"Overall Accuracy: {res['overall_accuracy']*100:.1f}% ({res['total_correct']}/{res['total_pairs']})")
        print("Category breakdown:")
        for cat, acc in res["category_accuracies"].items():
            print(f"  - {cat:<24}: {acc*100:5.1f}%")


if __name__ == "__main__":
    main()
