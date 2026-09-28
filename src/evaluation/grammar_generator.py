"""
Automated Synthetic Minimal Pairs Generator for Toki Pona Grammatical Acceptability.
Implements BLiMP-style minimal pair tests based on strict Toki Pona syntax rules:
1. Particle 'li' rule (mi/sina exemption vs mandatory 3rd-person 'li')
2. Direct object particle 'e' rule (transitive verb objects require 'e')
3. Modifier order rule (head-initial noun + modifier vs head-final English order)
4. Modifier grouping particle 'pi' rule (strictly requires 2+ modifier words)
5. Context/conditional particle 'la' rule
6. Lexical validity (canonical Toki Pona vocabulary vs pseudo-words)
"""

from __future__ import annotations
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class GrammarMinimalPair:
    item_id: str
    rule_id: str
    sub_rule: str
    category: str
    description: str
    grammatical: str
    ungrammatical: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


# Toki Pona Lexicon for Combinatorial Generation
PRONOUNS_NO_LI = ["mi", "sina"]
NOUNS_3RD = [
    "jan", "soweli", "waso", "kasi", "kili", "ona", "kulupu", "ilo",
    "akesi", "kala", "pipi", "tomo", "mun", "suno", "ma", "telo",
]

INTRANSITIVE_VERBS = [
    "moku", "lape", "tawa", "lukin", "kute", "pali", "toki", "awen",
    "kalama", "musi", "kama", "ante", "pilin", "pona",
]

TRANSITIVE_VERBS = [
    "moku", "lukin", "kute", "pali", "toki", "pana", "alasa", "wile",
    "jo", "pakala", "olin", "open", "pini",
]

OBJECT_NOUNS = [
    "kili", "telo", "pan", "lipu", "tomo", "ilo", "waso", "soweli",
    "mani", "kasi", "suno", "kiwen", "moku", "kalama", "suwi",
]

MODIFIERS = [
    "suli", "lili", "pona", "ike", "sin", "nasa", "laso", "loje",
    "jelo", "pimeja", "walo", "mute", "seli", "lete", "wawa", "suwi",
]

PSEUDO_WORDS = [
    "blorg", "grok", "qwerty", "splat", "zorp", "florp", "snark",
    "glub", "plink", "vroom", "krunk", "blip", "drap", "frob",
]


def generate_li_pairs() -> List[GrammarMinimalPair]:
    pairs: List[GrammarMinimalPair] = []
    idx = 1

    # 1A. mi / sina alone must NOT have 'li'
    for pron in PRONOUNS_NO_LI:
        for v in INTRANSITIVE_VERBS:
            g = f"{pron} {v}."
            u = f"{pron} li {v}."
            pairs.append(GrammarMinimalPair(
                item_id=f"li_mi_sina_{idx:03d}",
                rule_id="particle_li",
                sub_rule="pronoun_no_li",
                category="particle_li",
                description="Standalone mi/sina rejects particle li",
                grammatical=g,
                ungrammatical=u,
            ))
            idx += 1

        # With transitive objects
        for obj in ["kili", "telo", "pan", "lipu", "mani"]:
            for v in ["moku", "lukin", "pali", "pana", "wile"]:
                g = f"{pron} {v} e {obj}."
                u = f"{pron} li {v} e {obj}."
                pairs.append(GrammarMinimalPair(
                    item_id=f"li_mi_sina_{idx:03d}",
                    rule_id="particle_li",
                    sub_rule="pronoun_no_li",
                    category="particle_li",
                    description="Standalone mi/sina rejects particle li in transitive sentence",
                    grammatical=g,
                    ungrammatical=u,
                ))
                idx += 1

    # 1B. 3rd-person subjects REQUIRE 'li' (omission error)
    idx_3rd = 1
    for subj in NOUNS_3RD:
        for v in ["moku", "lape", "tawa", "lukin", "pali", "toki", "kalama", "awen"]:
            g = f"{subj} li {v}."
            u = f"{subj} {v}."
            pairs.append(GrammarMinimalPair(
                item_id=f"li_3rd_omit_{idx_3rd:03d}",
                rule_id="particle_li",
                sub_rule="3rd_person_requires_li",
                category="particle_li",
                description="3rd person and nominal subjects require particle li",
                grammatical=g,
                ungrammatical=u,
            ))
            idx_3rd += 1

    # 1C. Spurious 'e' replacing 'li'
    idx_e_rep = 1
    for subj in ["jan", "soweli", "waso", "ona", "kulupu"]:
        for v in ["moku", "lape", "tawa", "lukin", "pali"]:
            g = f"{subj} li {v}."
            u = f"{subj} e {v}."
            pairs.append(GrammarMinimalPair(
                item_id=f"li_spurious_e_{idx_e_rep:03d}",
                rule_id="particle_li",
                sub_rule="spurious_e_for_li",
                category="particle_li",
                description="Subject marker must be li, not particle e",
                grammatical=g,
                ungrammatical=u,
            ))
            idx_e_rep += 1

    return pairs


def generate_e_pairs() -> List[GrammarMinimalPair]:
    pairs: List[GrammarMinimalPair] = []
    idx = 1

    # 2A. Transitive verbs require direct object marker 'e'
    subjects = ["jan", "soweli", "waso", "ona", "mi", "sina"]
    for s in subjects:
        subj_part = f"{s} li" if s not in PRONOUNS_NO_LI else s
        for v in ["moku", "lukin", "kute", "pali", "pana", "alasa", "wile", "jo"]:
            for obj in ["kili", "telo", "pan", "lipu", "mani", "waso", "soweli", "tomo"]:
                g = f"{subj_part} {v} e {obj}."
                u = f"{subj_part} {v} {obj}."
                pairs.append(GrammarMinimalPair(
                    item_id=f"e_object_{idx:03d}",
                    rule_id="direct_object_e",
                    sub_rule="mandatory_e_marker",
                    category="direct_object_e",
                    description="Direct object requires particle e marker",
                    grammatical=g,
                    ungrammatical=u,
                ))
                idx += 1
                if idx > 80:  # Bound to avoid oversized category
                    break
            if idx > 80:
                break
        if idx > 80:
            break

    # 2B. Spurious direct object marker 'e' on prepositional or bare intransitive
    idx_sp = 1
    for s in ["jan", "soweli", "waso"]:
        for prep in ["lon tomo", "tawa ma", "kepeken ilo", "tan tomo"]:
            g = f"{s} li tawa {prep}."
            u = f"{s} li tawa e {prep}."
            pairs.append(GrammarMinimalPair(
                item_id=f"e_spurious_{idx_sp:03d}",
                rule_id="direct_object_e",
                sub_rule="no_e_before_preposition",
                category="direct_object_e",
                description="Prepositional complement must not take object particle e",
                grammatical=g,
                ungrammatical=u,
            ))
            idx_sp += 1

    return pairs


def generate_modifier_order_pairs() -> List[GrammarMinimalPair]:
    pairs: List[GrammarMinimalPair] = []
    idx = 1

    noun_mod_pairs = [
        ("tomo", "suli"), ("jan", "pona"), ("telo", "nasa"), ("kasi", "laso"),
        ("soweli", "wawa"), ("kili", "suwi"), ("waso", "lili"), ("ilo", "sin"),
        ("ma", "tomo"), ("tenpo", "suno"), ("kiwen", "pimeja"), ("lipu", "lili"),
        ("jan", "lili"), ("telo", "seli"), ("kulupu", "suli"), ("akesi", "wawa"),
    ]

    for n, m in noun_mod_pairs:
        # Subject position
        g1 = f"{n} {m} li lon."
        u1 = f"{m} {n} li lon."
        pairs.append(GrammarMinimalPair(
            item_id=f"order_subj_{idx:03d}",
            rule_id="modifier_order",
            sub_rule="head_initial_subject",
            category="modifier_order",
            description="Toki Pona modifiers must follow the head noun in subject",
            grammatical=g1,
            ungrammatical=u1,
        ))
        idx += 1

        # Object position
        g2 = f"mi lukin e {n} {m}."
        u2 = f"mi lukin e {m} {n}."
        pairs.append(GrammarMinimalPair(
            item_id=f"order_obj_{idx:03d}",
            rule_id="modifier_order",
            sub_rule="head_initial_object",
            category="modifier_order",
            description="Toki Pona modifiers must follow the head noun in object",
            grammatical=g2,
            ungrammatical=u2,
        ))
        idx += 1

        # 3rd-person transitive action
        g3 = f"jan li moku e {n} {m}."
        u3 = f"jan li moku e {m} {n}."
        pairs.append(GrammarMinimalPair(
            item_id=f"order_trans_{idx:03d}",
            rule_id="modifier_order",
            sub_rule="head_initial_transitive",
            category="modifier_order",
            description="Modifiers must follow head noun in transitive clause",
            grammatical=g3,
            ungrammatical=u3,
        ))
        idx += 1

    return pairs


def generate_pi_pairs() -> List[GrammarMinimalPair]:
    pairs: List[GrammarMinimalPair] = []
    idx = 1

    # 4A. Single modifier under 'pi' is strictly illegal
    single_mods = [
        ("tomo", "suli"), ("jan", "pona"), ("kili", "suwi"), ("telo", "seli"),
        ("ilo", "sin"), ("soweli", "wawa"), ("waso", "lili"), ("lipu", "lili"),
        ("kasi", "laso"), ("ma", "tomo"), ("kulupu", "lili"), ("akesi", "suli"),
    ]
    for n, m in single_mods:
        g = f"{n} {m} li pona."
        u = f"{n} pi {m} li pona."
        pairs.append(GrammarMinimalPair(
            item_id=f"pi_single_{idx:03d}",
            rule_id="modifier_pi",
            sub_rule="no_pi_with_single_modifier",
            category="modifier_pi",
            description="Particle pi is forbidden when only a single modifier follows",
            grammatical=g,
            ungrammatical=u,
        ))
        idx += 1

        g_trans = f"mi lukin e {n} {m}."
        u_trans = f"mi lukin e {n} pi {m}."
        pairs.append(GrammarMinimalPair(
            item_id=f"pi_single_trans_{idx:03d}",
            rule_id="modifier_pi",
            sub_rule="no_pi_with_single_modifier",
            category="modifier_pi",
            description="Particle pi is forbidden before single modifier in object",
            grammatical=g_trans,
            ungrammatical=u_trans,
        ))
        idx += 1

    # 4B. Grouping 2+ modifiers: compound modifier requires pi vs misplaced/single pi
    compounds_pi = [
        ("tomo", "telo", "nasa"),
        ("jan", "sona", "mute"),
        ("kili", "suwi", "mute"),
        ("nasin", "toki", "pona"),
        ("ilo", "sitelen", "sin"),
        ("kulupu", "jan", "utala"),
        ("telo", "kasi", "laso"),
        ("lipu", "sona", "suli"),
    ]
    for head, m1, m2 in compounds_pi:
        # Subject position: pi grouping 2 modifiers vs misplaced pi before single modifier (equal length)
        g1 = f"{head} pi {m1} {m2} li suli."
        u1 = f"{head} {m1} pi {m2} li suli."
        pairs.append(GrammarMinimalPair(
            item_id=f"pi_compound_subj_{idx:03d}",
            rule_id="modifier_pi",
            sub_rule="pi_requires_two_modifiers",
            category="modifier_pi",
            description="Particle pi must group 2+ modifiers; cannot precede a single final modifier",
            grammatical=g1,
            ungrammatical=u1,
        ))
        idx += 1

        # Object position (transitive, equal length)
        g2 = f"mi lukin e {head} pi {m1} {m2}."
        u2 = f"mi lukin e {head} {m1} pi {m2}."
        pairs.append(GrammarMinimalPair(
            item_id=f"pi_compound_obj_{idx:03d}",
            rule_id="modifier_pi",
            sub_rule="pi_requires_two_modifiers",
            category="modifier_pi",
            description="Particle pi must group 2+ modifiers in direct object position",
            grammatical=g2,
            ungrammatical=u2,
        ))
        idx += 1

        # Chained illegal pi before single modifiers
        g3 = f"{head} pi {m1} {m2} li pona."
        u3 = f"{head} pi {m1} pi {m2} li pona."
        pairs.append(GrammarMinimalPair(
            item_id=f"pi_compound_chain_{idx:03d}",
            rule_id="modifier_pi",
            sub_rule="pi_requires_two_modifiers",
            category="modifier_pi",
            description="Chained pi before single modifiers is ungrammatical",
            grammatical=g3,
            ungrammatical=u3,
        ))
        idx += 1

    return pairs


def generate_la_pairs() -> List[GrammarMinimalPair]:
    pairs: List[GrammarMinimalPair] = []
    idx = 1

    contexts = [
        ("tenpo suno ni la", "tenpo suno ni e"),
        ("tenpo pimeja la", "tenpo pimeja e"),
        ("sina wile la", "sina wile li"),
        ("mi lape la", "mi lape li"),
        ("telo li lon la", "telo li lon e"),
        ("ken la", "ken e"),
        ("jan li moku la", "jan li moku e"),
        ("sina toki e ni la", "sina toki e ni li"),
    ]
    consequents = [
        "mi pali.", "ona li kama.", "kasi li suli.", "jan li lape.", "soweli li tawa.",
    ]

    for (c_g, c_u) in contexts:
        for cons in consequents:
            g = f"{c_g} {cons}"
            u = f"{c_u} {cons}"
            pairs.append(GrammarMinimalPair(
                item_id=f"la_context_{idx:03d}",
                rule_id="context_la",
                sub_rule="context_marker_la",
                category="context_la",
                description="Context or conditional phrase requires marker la",
                grammatical=g,
                ungrammatical=u,
            ))
            idx += 1
            if idx > 30:
                break
        if idx > 30:
            break

    return pairs


def generate_lexical_validity_pairs() -> List[GrammarMinimalPair]:
    pairs: List[GrammarMinimalPair] = []
    idx = 1

    frames = [
        ("jan li moku e {}.", "kili", "blorg"),
        ("ona li lape lon {}.", "tomo", "grok"),
        ("soweli li tawa {}.", "noka", "qwerty"),
        ("waso lili li {}.", "kalama", "splat"),
        ("mi lukin e {}.", "suno", "zorp"),
        ("kasi li suli lon {}.", "ma", "florp"),
        ("sina pana e {} tawa mi.", "mani", "snark"),
        ("ilo ni li {}.", "pakala", "glub"),
        ("kulupu li toki e {}.", "toki", "plink"),
        ("telo li {} lon lupa.", "kama", "vroom"),
        ("akesi li moku e {}.", "pipi", "krunk"),
        ("jan pona mi li {}.", "sona", "blip"),
        ("ona mute li {}.", "olin", "drap"),
        ("lipu ni li {}.", "pona", "frob"),
    ]

    for frame, valid_w, pseudo_w in frames:
        g = frame.format(valid_w)
        u = frame.format(pseudo_w)
        pairs.append(GrammarMinimalPair(
            item_id=f"lex_valid_{idx:03d}",
            rule_id="lexical_validity",
            sub_rule="valid_tokipona_lexicon",
            category="lexical_validity",
            description="Toki Pona vocabulary word vs pseudo-word",
            grammatical=g,
            ungrammatical=u,
        ))
        idx += 1

    return pairs


def generate_all_grammar_pairs() -> List[GrammarMinimalPair]:
    """
    Synthesizes the complete BLiMP-style Toki Pona grammar test suite (300+ pairs).
    """
    all_pairs: List[GrammarMinimalPair] = []
    all_pairs.extend(generate_li_pairs())
    all_pairs.extend(generate_e_pairs())
    all_pairs.extend(generate_modifier_order_pairs())
    all_pairs.extend(generate_pi_pairs())
    all_pairs.extend(generate_la_pairs())
    all_pairs.extend(generate_lexical_validity_pairs())
    return all_pairs


def save_grammar_suite(pairs: List[GrammarMinimalPair], output_dir: Path | str) -> Tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "grammar_pairs.json"
    csv_path = output_dir / "grammar_pairs.csv"

    dict_data = [p.to_dict() for p in pairs]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dict_data, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "item_id", "rule_id", "sub_rule", "category", "description",
            "grammatical", "ungrammatical"
        ])
        writer.writeheader()
        writer.writerows(dict_data)

    print(f"[Grammar Generator] Generated {len(pairs)} minimal pairs.")
    print(f"  Saved JSON -> {json_path}")
    print(f"  Saved CSV  -> {csv_path}")

    return json_path, csv_path


def load_grammar_suite(filepath: Path | str) -> List[GrammarMinimalPair]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Grammar test suite not found at {path}")

    pairs: List[GrammarMinimalPair] = []
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for row in data:
                pairs.append(GrammarMinimalPair(**row))
    elif path.suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append(GrammarMinimalPair(**row))
    else:
        raise ValueError(f"Unsupported format {path.suffix}, expected .json or .csv")

    return pairs


if __name__ == "__main__":
    test_pairs = generate_all_grammar_pairs()
    root = Path(__file__).resolve().parent.parent.parent
    save_grammar_suite(test_pairs, root / "data" / "evaluation")
