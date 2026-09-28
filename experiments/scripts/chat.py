"""
Interactive Chatbot & Text Completion CLI for trained Toki Pona SLMs.
Supports both dialogue mode (turn-based chat) and continuation/completion mode.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

root = Path(__file__).resolve().parent.parent.parent
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stdin and hasattr(sys.stdin, "reconfigure"):
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from experiments.scripts.demo_inference import load_model_and_tokenizer


TOKIPONA_CHEATSHEET = """
================================================================================
[INFO] TAHAK ZAKLADNICH FRAZI TOKI PONA (Model rozumi pouze Toki Pona):
--------------------------------------------------------------------------------
  toki!                             -> Ahoj!
  sina pona ala pona?               -> Jak se mas? (Jsi v poradku?)
  sina seme?                        -> Kdo jsi?
  sina sona ala sona e toki pona?   -> Umis toki pona?
  sina wile moku e seme?            -> Co chces jist?
  mi wile sona e ni:                -> Chci vedet: ...
  tenpo ni la mi ...                -> Ted zrovna ja ...
  jan pona o, toki!                 -> Ahoj, priteli!
================================================================================
"""

HELP_COMMANDS = """
Příkazy v chatu:
  /help            - Zobrazí tuto nápovědu a tahák Toki Pona
  /mode            - Přepne režim: 'dialogue' (chat) <-> 'complete' (dokončování)
  /temp <hodnota>  - Nastaví teplotu generování (výchozí: 0.7, rozsah 0.1 až 1.5)
  /topk <hodnota>  - Nastaví top_k vzorkování (výchozí: 20, 0 = vypnuto)
  /tokens <počet>  - Nastaví maximální počet tokenů k vygenerování (výchozí: 40)
  /model <exp_id>  - Přepne načtený model (např. exp_010_d75_mini_char_s42)
  /clear           - Vymaže historii konverzace
  /exit nebo /quit - Ukončí chat
"""


def generate_chat_reply(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 40,
    temperature: float = 0.7,
    top_k: int = 20,
    stop_on_newline: bool = True,
) -> str:
    """Generate reply tokens until stop condition (EOS, newline, or max tokens)."""
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    if tokens and tokens[-1] == tokenizer.eos_token_id:
        tokens = tokens[:-1]

    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    generated_ids = []

    for _ in range(max_new_tokens):
        x_cond = (
            input_ids
            if input_ids.size(1) <= model.config.max_seq_len
            else input_ids[:, -model.config.max_seq_len :]
        )
        with torch.no_grad():
            logits, _ = model(x_cond)

        next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

        if top_k > 0:
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(next_token_logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        tid = next_tok.item()

        if tid == tokenizer.eos_token_id:
            break

        piece = tokenizer.decode([tid], skip_special_tokens=True)
        if stop_on_newline and "\n" in piece:
            break
        if "jan wan:" in piece or "jan:" in piece:
            break

        generated_ids.append(tid)
        input_ids = torch.cat((input_ids, next_tok), dim=1)

    reply = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return reply


def start_interactive_chat(
    run_name: str = "exp_012_d75_mini_word_s42",
    temperature: float = 0.7,
    top_k: int = 20,
    max_tokens: int = 40,
    initial_mode: str = "dialogue",
):
    print("=" * 80)
    print("=== TOKI PONA SLM - INTERAKTIVNI CHATBOT & GENERATOR ===")
    print("=" * 80)
    print(f"Nacitam model: {run_name} ...")

    model, tokenizer, device, meta = load_model_and_tokenizer(run_name, root)

    print(f"[OK] Model uspesne nacten na {device}!")
    print(f"  - Tokenizer: {meta['tokenizer'].upper()} (vocab size: {tokenizer.vocab_size})")
    print(f"  - Pocet parametru: {int(meta.get('total_params', 0)):,}")
    print(f"  - Val BPC: {meta.get('val_bpc', 'N/A')} | BLiMP presnost: {float(meta.get('grammar_acc', 0.0))*100:.1f} %")
    print(TOKIPONA_CHEATSHEET)
    print(HELP_COMMANDS)
    print("-" * 80)

    mode = initial_mode
    history: list[tuple[str, str]] = []

    print(f"Aktuální režim: [{mode.upper()}] (Zadejte /mode pro přepnutí, /help pro nápovědu)")
    print("Začněte psát zprávu v Toki Pona (např. 'toki! sina pona ala pona?') a stiskněte Enter.\n")

    while True:
        try:
            user_input = input("Vy (jan wan) > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nUkončuji chat. Mi tawa!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd in ("/exit", "/quit", "/q"):
                print("Mi tawa! (Měj se!)")
                break
            elif cmd == "/help":
                print(TOKIPONA_CHEATSHEET)
                print(HELP_COMMANDS)
            elif cmd == "/mode":
                mode = "complete" if mode == "dialogue" else "dialogue"
                print(f"Režim přepnut na: [{mode.upper()}]")
            elif cmd == "/clear":
                history.clear()
                print("Historie dialogu byla vymazána.")
            elif cmd == "/temp":
                if len(parts) > 1:
                    try:
                        temperature = float(parts[1])
                        print(f"Teplota nastavena na: {temperature}")
                    except ValueError:
                        print("Chyba: Zadejte platné číslo (např. /temp 0.8)")
                else:
                    print(f"Aktuální teplota: {temperature}")
            elif cmd == "/topk":
                if len(parts) > 1:
                    try:
                        top_k = int(parts[1])
                        print(f"Top-K nastaveno na: {top_k}")
                    except ValueError:
                        print("Chyba: Zadejte platné celé číslo (např. /topk 20)")
                else:
                    print(f"Aktuální Top-K: {top_k}")
            elif cmd == "/tokens":
                if len(parts) > 1:
                    try:
                        max_tokens = int(parts[1])
                        print(f"Max nových tokenů nastaveno na: {max_tokens}")
                    except ValueError:
                        print("Chyba: Zadejte platné číslo (např. /tokens 50)")
                else:
                    print(f"Aktuální max_tokens: {max_tokens}")
            elif cmd == "/model":
                if len(parts) > 1:
                    new_run = parts[1]
                    try:
                        print(f"Načítám {new_run} ...")
                        m, t, d, meta_new = load_model_and_tokenizer(new_run, root)
                        model, tokenizer, device, meta = m, t, d, meta_new
                        run_name = new_run
                        history.clear()
                        print(f"[OK] Uspesne prepnuto na model: {new_run}")
                    except Exception as e:
                        print(f"Chyba pri nacitani modelu {new_run}: {e}")
                else:
                    print(f"Aktualni model: {run_name}")
            else:
                print(f"Neznamy prikaz: {cmd}. Napiste /help pro napovedu.")
            continue

        # Generation based on mode
        if mode == "dialogue":
            # Build dialogue prompt with rolling history (keep last 2 turns to fit context)
            dialogue_lines = []
            for h_u, h_b in history[-2:]:
                dialogue_lines.append(f"jan wan: {h_u}")
                dialogue_lines.append(f"jan tu: {h_b}")
            dialogue_lines.append(f"jan wan: {user_input}")
            dialogue_lines.append("jan tu: ")

            prompt = "\n".join(dialogue_lines)
            reply = generate_chat_reply(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                stop_on_newline=True,
            )
            if not reply:
                reply = "..."

            history.append((user_input, reply))
            print(f"[BOT] jan tu > {reply}\n")

        else:  # Completion mode
            reply = generate_chat_reply(
                model=model,
                tokenizer=tokenizer,
                prompt=user_input + " ",
                device=device,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                stop_on_newline=False,
            )
            full_text = (user_input + " " + reply).strip()
            print(f"[COMPLETION] > {full_text}\n")


def main():
    parser = argparse.ArgumentParser(description="Toki Pona SLM Interactive Chatbot")
    parser.add_argument(
        "--run",
        type=str,
        default="exp_012_d75_mini_word_s42",
        help="Experiment ID / model checkpoint (default: exp_012_d75_mini_word_s42)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dialogue", "complete"],
        default="dialogue",
        help="Režim spuštění: 'dialogue' (chat) nebo 'complete' (doplňování textu)",
    )
    parser.add_argument("--temp", type=float, default=0.7, help="Teplota vzorkování (default: 0.7)")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K filtr (default: 20)")
    parser.add_argument("--tokens", type=int, default=40, help="Max nových tokenů (default: 40)")
    args = parser.parse_args()

    start_interactive_chat(
        run_name=args.run,
        temperature=args.temp,
        top_k=args.top_k,
        max_tokens=args.tokens,
        initial_mode=args.mode,
    )


if __name__ == "__main__":
    main()
