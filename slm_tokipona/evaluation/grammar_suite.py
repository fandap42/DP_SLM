"""
Grammatical Acceptability & Compositional Generalization Test Suite for Toki Pona.
Inspired by BLiMP (Benchmark of Linguistic Minimal Pairs).
Evaluates whether model assigns higher log-likelihood to grammatical vs ungrammatical sentences.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
from ..tokenizers.base import BaseTokenizer
from .metrics import score_sentence


@dataclass
class MinimalPair:
    category: str
    description: str
    grammatical: str
    ungrammatical: str


def get_default_test_suite() -> List[MinimalPair]:
    """
    Generates a rich suite of minimal pairs testing core Toki Pona syntax rules:
    - li insertion (3rd person requires li)
    - mi/sina subject exclusion (mi/sina alone rejects li)
    - direct object marker e
    - modifier grouping pi (requires at least 2 modifiers)
    - context marker la
    - compositional generalization (semantic modifier compositions)
    - lexical validity (standard word vs pseudo-word)
    """
    pairs: List[MinimalPair] = []

    # 1. Subject 'li' particle for 3rd-person / full noun subjects
    li_examples = [
        ("soweli li moku.", "soweli moku."),
        ("jan li lape.", "jan lape."),
        ("waso li tawa.", "waso tawa."),
        ("kasi li suli.", "kasi suli."),
        ("kili li pona.", "kili pona."),
        ("ona li lukin.", "ona lukin."),
        ("kulupu li toki.", "kulupu toki."),
        ("ilo li pakala.", "ilo pakala."),
        ("soweli li pona e tomo.", "soweli e pona e tomo."),
        ("jan li pali e lipu.", "jan e pali e lipu."),
    ]
    for g, u in li_examples:
        pairs.append(MinimalPair("particle_li", "3rd person subject requires li particle", g, u))

    # 2. Pronouns 'mi' and 'sina' alone DO NOT take 'li'
    mi_sina_examples = [
        ("mi moku e kili.", "mi li moku e kili."),
        ("mi lape lon tomo.", "mi li lape lon tomo."),
        ("mi lukin e suno.", "mi li lukin e suno."),
        ("sina pona tawa mi.", "sina li pona tawa mi."),
        ("sina moku e telo.", "sina li moku e telo."),
        ("sina toki e ni.", "sina li toki e ni."),
    ]
    for g, u in mi_sina_examples:
        pairs.append(MinimalPair("pronoun_no_li", "Standalone mi/sina rejects li", g, u))

    # 3. Direct Object Marker 'e'
    e_examples = [
        ("jan li moku e telo.", "jan li moku telo."),
        ("soweli li lukin e waso.", "soweli li lukin waso."),
        ("mi pana e mani tawa sina.", "mi pana mani tawa sina."),
        ("ona li toki e toki pona.", "ona li toki toki pona."),
        ("kulupu li pali e tomo.", "kulupu li pali tomo."),
        ("jan lili li kute e mama.", "jan lili li kute mama."),
        ("mi wile e supa sin.", "mi wile supa sin."),
    ]
    for g, u in e_examples:
        pairs.append(MinimalPair("direct_object_e", "Transitive verbs require particle e", g, u))

    # 4. Grouping particle 'pi' (must precede a modifier phrase of 2+ words)
    pi_examples = [
        ("tomo pi telo nasa li suli.", "tomo pi telo li suli."),
        ("jan pi sona mute li toki.", "jan pi sona li toki."),
        ("kili pi suwi mute li pona.", "kili pi suwi li pona."),
        ("nasin pi toki pona li lili.", "nasin pi toki li lili."),
        ("ilo pi sitelen sin li awen.", "ilo pi sitelen li awen."),
    ]
    for g, u in pi_examples:
        pairs.append(MinimalPair("modifier_pi", "Particle pi requires compound modifier", g, u))

    # 5. Context / Conditional particle 'la'
    la_examples = [
        ("tenpo suno ni la mi pali.", "tenpo suno ni e mi pali."),
        ("sina wile la mi tawa.", "sina wile li mi tawa."),
        ("telo li lon la kasi li suli.", "telo li lon e kasi li suli."),
        ("ken la ona li kama.", "ken e ona li kama."),
    ]
    for g, u in la_examples:
        pairs.append(MinimalPair("context_la", "Conditional context phrase requires la", g, u))

    # 6. Compositional Generalization: Modifier order & semantic bindings
    comp_examples = [
        ("kili suwi lili li lon.", "kili li suwi lili lon."),
        ("jan pona mi li moku e pan.", "jan mi pona li moku pan e."),
        ("soweli wawa pimeja li tawa.", "soweli li wawa pimeja tawa."),
        ("ilo pali sin li pakala ala.", "ilo li pali sin pakala ala."),
        ("telo seli pimeja li pona tawa mi.", "telo li seli pimeja pona mi tawa."),
    ]
    for g, u in comp_examples:
        pairs.append(MinimalPair("compositional_syntax", "Compositional modifier structure", g, u))

    # 7. Lexicon validity (Toki Pona word vs phonologically valid non-word)
    lex_examples = [
        ("jan li moku e kili.", "jan li moku e blorg."),
        ("ona li lape lon tomo.", "ona li lape lon grok."),
        ("soweli li tawa noka.", "soweli li tawa qwerty."),
        ("waso lili li kalama.", "waso splat li kalama."),
    ]
    for g, u in lex_examples:
        pairs.append(MinimalPair("lexical_validity", "Standard Toki Pona word vs pseudo-word", g, u))

    return pairs


def evaluate_grammar_suite(
    model: torch.nn.Module,
    tokenizer: BaseTokenizer,
    device: torch.device,
    test_suite: List[MinimalPair] | None = None,
) -> Dict[str, any]:
    """
    Evaluates the model across the test suite of minimal pairs.
    Returns:
      - overall_accuracy: % of pairs where log P(grammatical) > log P(ungrammatical)
      - category_accuracies: breakdown per linguistic rule category
      - pair_details: individual evaluations
    """
    if test_suite is None:
        test_suite = get_default_test_suite()

    model.eval()
    cat_correct: Dict[str, int] = {}
    cat_total: Dict[str, int] = {}
    details: List[Dict] = []

    total_correct = 0

    for pair in test_suite:
        logp_g = score_sentence(model, tokenizer, pair.grammatical, device)
        logp_u = score_sentence(model, tokenizer, pair.ungrammatical, device)

        is_correct = bool(logp_g > logp_u)
        if is_correct:
            total_correct += 1

        cat_correct[pair.category] = cat_correct.get(pair.category, 0) + int(is_correct)
        cat_total[pair.category] = cat_total.get(pair.category, 0) + 1

        details.append({
            "category": pair.category,
            "grammatical": pair.grammatical,
            "ungrammatical": pair.ungrammatical,
            "logp_grammatical": round(logp_g, 3),
            "logp_ungrammatical": round(logp_u, 3),
            "correct": is_correct,
            "margin": round(logp_g - logp_u, 3),
        })

    cat_acc = {
        cat: round(cat_correct[cat] / cat_total[cat], 4)
        for cat in cat_total
    }
    overall_acc = round(total_correct / len(test_suite), 4)

    return {
        "overall_accuracy": overall_acc,
        "category_accuracies": cat_acc,
        "total_pairs": len(test_suite),
        "total_correct": total_correct,
        "details": details,
    }
