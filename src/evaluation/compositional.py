"""
Compositional Generalization Test Suite for Toki Pona SLMs.
Evaluates model capacity to process novel multiword compounds (held-out combinations)
versus familiar control compounds, and measures the Compositional Generalization Gap:
  Gap_diff  = BPC_heldout - BPC_seen
  Gap_ratio = BPC_heldout / BPC_seen
"""

from __future__ import annotations
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch

from ..tokenization.base import BaseTokenizer
from .metrics import score_sentence


@dataclass
class CompositionalSentence:
    item_id: str
    condition: str          # "heldout" or "seen"
    compound: str           # e.g. "telo nasa"
    text: str               # sentence text
    frame_type: str         # syntactic frame identifier

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class CompositionalMinimalPair:
    item_id: str
    compound: str
    condition: str          # "heldout" or "seen"
    description: str
    grammatical: str        # e.g. "ona li moku e telo nasa."
    ungrammatical: str      # e.g. "ona li moku e nasa telo."

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


# Canonical Toki Pona Compounds for Compositional Generalization
HELDOUT_COMPOUNDS = [
    ("telo nasa", "alcohol / crazy water"),
    ("tomo telo", "restroom / water building"),
    ("jan utala", "soldier / warrior"),
    ("ilo moku", "cutlery / eating utensil"),
    ("tomo tawa", "vehicle / moving room"),
    ("jan sona", "scientist / scholar"),
    ("tomo sona", "school / university"),
    ("ilo suno", "flashlight / lamp"),
    ("ma tomo", "city / town"),
    ("jan lawa", "leader / director"),
    ("supa lape", "bed / sleeping surface"),
    ("kiwen suwi", "sugar / sweet rock"),
    ("telo kili", "juice / fruit water"),
    ("jan alasa", "hunter"),
    ("kasi suli", "tree / big plant"),
    ("tomo moku", "restaurant / dining hall"),
    ("lipu sona", "textbook / study guide"),
    ("ma kasi", "forest / jungle"),
    ("telo pimeja", "coffee / dark brew"),
    ("supa pali", "desk / workbench"),
]

SEEN_COMPOUNDS = [
    ("jan pona", "friend / good person"),
    ("toki pona", "Toki Pona / good speech"),
    ("tomo suli", "large building / palace"),
    ("telo seli", "hot water / tea"),
    ("tenpo suno", "day / daylight"),
    ("jan lili", "child / young person"),
    ("tenpo ni", "present / now"),
    ("kili pona", "good fruit"),
    ("soweli wawa", "powerful animal"),
    ("lipu pona", "good book / document"),
    ("ma suli", "large country / land"),
    ("telo pona", "pure water"),
    ("kasi lili", "small plant / flower"),
    ("jan suli", "adult / elder"),
    ("suno suli", "bright sun"),
    ("moku pona", "good meal / healthy food"),
    ("ona mute", "they / group"),
    ("ma lili", "small village / countryside"),
    ("ilo pona", "useful tool"),
    ("supa suli", "large table"),
]

CANONICAL_HELDOUT_COMPOUNDS = [comp for comp, _ in HELDOUT_COMPOUNDS]


def verify_heldout_purity(
    dataset_path: Path | str,
    heldout_compounds: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Checks how many times each held-out compound appears in a given JSONL dataset.
    Returns a dictionary of compound -> occurrence count.
    """
    import re
    if heldout_compounds is None:
        heldout_compounds = CANONICAL_HELDOUT_COMPOUNDS

    patterns = {comp: re.compile(r"\b" + re.escape(comp) + r"\b") for comp in heldout_compounds}
    counts = {comp: 0 for comp in heldout_compounds}

    path = Path(dataset_path)
    if not path.exists():
        return counts

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            text = item.get("text", "")
            for comp, pat in patterns.items():
                if pat.search(text):
                    counts[comp] += 1

    return counts


def generate_compositional_sentences() -> List[CompositionalSentence]:
    """
    Constructs syntactically controlled sentence pairs embedding held-out vs seen compounds.
    """
    sentences: List[CompositionalSentence] = []
    idx = 1

    # Symmetric syntactic frames: each frame takes a compound
    frames = [
        ("trans_obj", "mi lukin e {} lon poka mi."),
        ("trans_eat", "ona li moku e {} lon tomo."),
        ("trans_have", "jan ni li jo e {} wawa."),
        ("trans_give", "ona li pana e {} tawa mi."),
        ("trans_want", "sina wile ala wile e {}?"),
        ("subj_pred", "{} li pona mute tawa mi."),
        ("subj_action", "{} li tawa sewi."),
        ("subj_exist", "{} li lon ma ni."),
        ("prep_loc", "jan lili li awen lon {}."),
        ("prep_inst", "ona li pali kepeken {}."),
    ]

    for f_id, template in frames:
        # Held-out compounds
        for comp, _ in HELDOUT_COMPOUNDS:
            text = template.format(comp)
            sentences.append(CompositionalSentence(
                item_id=f"comp_sent_h_{idx:04d}",
                condition="heldout",
                compound=comp,
                text=text,
                frame_type=f_id,
            ))
            idx += 1

        # Seen control compounds
        for comp, _ in SEEN_COMPOUNDS:
            text = template.format(comp)
            sentences.append(CompositionalSentence(
                item_id=f"comp_sent_s_{idx:04d}",
                condition="seen",
                compound=comp,
                text=text,
                frame_type=f_id,
            ))
            idx += 1

    return sentences


def generate_compositional_minimal_pairs() -> List[CompositionalMinimalPair]:
    """
    Constructs minimal pairs verifying compositional modifier ordering
    (correct head-modifier order vs reversed head-final order) for compounds.
    """
    pairs: List[CompositionalMinimalPair] = []
    idx = 1

    compound_groups = [
        (HELDOUT_COMPOUNDS, "heldout"),
        (SEEN_COMPOUNDS, "seen"),
    ]

    templates = [
        ("ona li lukin e {}.", "ona li lukin e {}."),
        ("{} li suli.", "{} li suli."),
        ("mi wile e {}.", "mi wile e {}."),
        ("jan li moku e {}.", "jan li moku e {}."),
    ]

    for compounds, cond in compound_groups:
        for comp, desc in compounds:
            parts = comp.split()
            if len(parts) != 2:
                continue
            head, mod = parts[0], parts[1]
            rev_comp = f"{mod} {head}"

            for t_g, t_u in templates:
                g = t_g.format(comp)
                u = t_u.format(rev_comp)

                pairs.append(CompositionalMinimalPair(
                    item_id=f"comp_pair_{cond}_{idx:04d}",
                    compound=comp,
                    condition=cond,
                    description=f"Compositional head-initial ordering for '{comp}' ({desc})",
                    grammatical=g,
                    ungrammatical=u,
                ))
                idx += 1

    return pairs


def save_compositional_suite(output_dir: Path | str) -> Tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sents = generate_compositional_sentences()
    pairs = generate_compositional_minimal_pairs()

    # Save sentences
    sents_json = output_dir / "compositional_sentences.json"
    sents_csv = output_dir / "compositional_sentences.csv"
    with open(sents_json, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in sents], f, indent=2, ensure_ascii=False)
    with open(sents_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "condition", "compound", "text", "frame_type"])
        writer.writeheader()
        writer.writerows([s.to_dict() for s in sents])

    # Save pairs
    pairs_json = output_dir / "compositional_pairs.json"
    pairs_csv = output_dir / "compositional_pairs.csv"
    with open(pairs_json, "w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in pairs], f, indent=2, ensure_ascii=False)
    with open(pairs_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "compound", "condition", "description", "grammatical", "ungrammatical"])
        writer.writeheader()
        writer.writerows([p.to_dict() for p in pairs])

    print(f"[Compositional Suite] Generated {len(sents)} test sentences and {len(pairs)} minimal pairs.")
    print(f"  Sentences -> {sents_json}")
    print(f"  Pairs     -> {pairs_json}")

    return sents_json, pairs_json


def load_compositional_suite(data_dir: Path | str) -> Tuple[List[CompositionalSentence], List[CompositionalMinimalPair]]:
    data_dir = Path(data_dir)
    sents_file = data_dir / "compositional_sentences.json"
    pairs_file = data_dir / "compositional_pairs.json"

    if not sents_file.exists() or not pairs_file.exists():
        save_compositional_suite(data_dir)

    with open(sents_file, "r", encoding="utf-8") as f:
        sents_data = json.load(f)
        sents = [CompositionalSentence(**r) for r in sents_data]

    with open(pairs_file, "r", encoding="utf-8") as f:
        pairs_data = json.load(f)
        pairs = [CompositionalMinimalPair(**r) for r in pairs_data]

    return sents, pairs


@torch.no_grad()
def evaluate_compositional_suite(
    model: torch.nn.Module,
    tokenizer: BaseTokenizer,
    device: torch.device,
    data_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Evaluates model performance on held-out vs seen compound sentences,
    computes Bits-Per-Character (BPC), perplexity, and the Compositional Generalization Gap.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "evaluation"

    sents, pairs = load_compositional_suite(data_dir)
    model.eval()

    # 1. Evaluate Sentences (BPC and Perplexity on heldout vs seen)
    stats = {
        "heldout": {"nats": 0.0, "tokens": 0, "chars": 0},
        "seen": {"nats": 0.0, "tokens": 0, "chars": 0},
    }

    ln2 = math.log(2.0)

    for item in sents:
        cond = item.condition
        logp = score_sentence(model, tokenizer, item.text, device)
        tokens = tokenizer.encode(item.text, add_special_tokens=True)
        # target tokens are tokens[1:]
        num_target_tokens = max(1, len(tokens) - 1)
        char_len = max(1, len(item.text))

        nats = -logp if logp > -9000.0 else (50.0 * num_target_tokens)

        stats[cond]["nats"] += nats
        stats[cond]["tokens"] += num_target_tokens
        stats[cond]["chars"] += char_len

    h_bpc = (stats["heldout"]["nats"] / ln2) / max(stats["heldout"]["chars"], 1)
    s_bpc = (stats["seen"]["nats"] / ln2) / max(stats["seen"]["chars"], 1)

    h_loss_tok = stats["heldout"]["nats"] / max(stats["heldout"]["tokens"], 1)
    s_loss_tok = stats["seen"]["nats"] / max(stats["seen"]["tokens"], 1)

    h_ppl = math.exp(min(h_loss_tok, 50.0))
    s_ppl = math.exp(min(s_loss_tok, 50.0))

    # Generalization gap: difference and ratio
    gap_diff = h_bpc - s_bpc
    gap_ratio = h_bpc / max(s_bpc, 1e-6)

    # 2. Evaluate Minimal Pairs (Compositional ordering accuracy)
    pair_correct = 0
    pair_correct_h = 0
    pair_total_h = 0
    pair_correct_s = 0
    pair_total_s = 0

    pair_details = []

    for p in pairs:
        logp_g = score_sentence(model, tokenizer, p.grammatical, device)
        logp_u = score_sentence(model, tokenizer, p.ungrammatical, device)
        is_correct = int(logp_g > logp_u)

        pair_correct += is_correct
        if p.condition == "heldout":
            pair_correct_h += is_correct
            pair_total_h += 1
        else:
            pair_correct_s += is_correct
            pair_total_s += 1

        pair_details.append({
            "item_id": p.item_id,
            "condition": p.condition,
            "compound": p.compound,
            "logp_grammatical": round(logp_g, 3),
            "logp_ungrammatical": round(logp_u, 3),
            "margin": round(logp_g - logp_u, 3),
            "correct": is_correct,
        })

    pair_acc = pair_correct / max(len(pairs), 1)
    pair_acc_h = pair_correct_h / max(pair_total_h, 1)
    pair_acc_s = pair_correct_s / max(pair_total_s, 1)

    return {
        "comp_bpc_heldout": round(h_bpc, 4),
        "comp_bpc_seen": round(s_bpc, 4),
        "comp_ppl_heldout": round(h_ppl, 2),
        "comp_ppl_seen": round(s_ppl, 2),
        "comp_generalization_gap": round(gap_diff, 4),
        "comp_gap_ratio": round(gap_ratio, 4),
        "comp_pair_accuracy": round(pair_acc, 4),
        "comp_pair_acc_heldout": round(pair_acc_h, 4),
        "comp_pair_acc_seen": round(pair_acc_s, 4),
        "total_test_sentences": len(sents),
        "total_minimal_pairs": len(pairs),
        "pair_details": pair_details,
    }
