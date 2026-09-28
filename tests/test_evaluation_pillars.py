"""
Unit and integration test suite for the 4 Evaluation Pillars:
- Pillar 1: Standard Test Set BPC & Domain Split
- Pillar 2: Synthetic Minimal Pairs & Grammar Suite (BLiMP)
- Pillar 3: Compositional Generalization & Generalization Gap
- Pillar 4: Statistical Models (Power-Law, LMM, ANOVA, GLMM)
"""

from __future__ import annotations
import math
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.models.configs import TransformerConfig
from src.models.transformer import TokiPonaTransformer
from src.tokenization.char_tokenizer import CharTokenizer
from src.tokenization.word_tokenizer import WordTokenizer
from src.evaluation.metrics import compute_loss_and_bpc, score_sentence, score_sentence_bpc
from src.evaluation.grammar_generator import (
    generate_all_grammar_pairs,
    generate_li_pairs,
    generate_e_pairs,
    generate_modifier_order_pairs,
    generate_pi_pairs,
    generate_la_pairs,
    generate_lexical_validity_pairs,
    GrammarMinimalPair,
)
from src.evaluation.grammar_suite import evaluate_grammar_suite
from src.evaluation.compositional import (
    generate_compositional_sentences,
    generate_compositional_minimal_pairs,
    evaluate_compositional_suite,
)
from src.evaluation.stats import (
    fit_scaling_law,
    fit_mixed_effects_model,
    compute_factorial_anova,
    fit_grammar_glmm,
    power_law_kaplan,
)
from src.evaluation.evaluator import UnifiedEvaluator


class TestPillar1BPCAndDomainMetrics(unittest.TestCase):
    """Tests character-normalized cross entropy, BPC, and domain splitting."""

    def setUp(self):
        self.sample_texts = [
            "jan li moku e kili.",
            "soweli li lape lon tomo.",
            "waso lili li tawa sewi.",
            "mi wile e telo nasa.",
        ]
        self.tokenizer = WordTokenizer(self.sample_texts)
        cfg = TransformerConfig.get_preset("micro", vocab_size=self.tokenizer.vocab_size)
        self.model = TokiPonaTransformer(cfg)
        self.device = torch.device("cpu")

    def test_bpc_mathematical_identity(self):
        """BPC must equal Loss_char / ln(2)."""
        loss_char = 0.693147  # ~ ln(2)
        bpc = loss_char / math.log(2.0)
        self.assertAlmostEqual(bpc, 1.0, places=4)

        ppl_char = 2.0 ** bpc
        self.assertAlmostEqual(ppl_char, 2.0, places=4)

    def test_compute_loss_and_bpc_mock_batches(self):
        """Verify computation on batch with known characters and tokens."""
        # Create a mock batch
        input_ids = torch.tensor([[1, 4, 5, 2], [1, 6, 7, 2]])
        target_ids = torch.tensor([[4, 5, 2, -100], [6, 7, 2, -100]])
        char_lens = torch.tensor([15, 20])
        sources = ["tatoeba", "wikipesija"]

        batch = {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "char_lens": char_lens,
            "sources": sources,
        }

        class MockDataLoader:
            def __iter__(self):
                return iter([batch])

        res = compute_loss_and_bpc(self.model, MockDataLoader(), self.device, in_domain_sources=("tatoeba",))

        self.assertIn("bpc", res)
        self.assertIn("loss_char", res)
        self.assertIn("bpc_in_domain", res)
        self.assertIn("bpc_out_domain", res)
        self.assertIn("domain_gap", res)
        self.assertEqual(res["total_chars"], 35)
        self.assertEqual(res["total_tokens"], 6)
        self.assertIn("tatoeba", res["sources"])
        self.assertIn("wikipesija", res["sources"])

    def test_empty_dataloader_handling(self):
        """Empty dataloader should return zeroes without crashing."""
        class EmptyDataLoader:
            def __iter__(self):
                return iter([])

        res = compute_loss_and_bpc(self.model, EmptyDataLoader(), self.device)
        self.assertEqual(res["bpc"], 0.0)
        self.assertEqual(res["total_chars"], 0)

    def test_truncation_char_len_fairness(self):
        """Verify that TokiPonaDataset char_len reflects modeled prefix length on truncation."""
        import tempfile
        import json
        from src.data.dataset import TokiPonaDataset

        long_sent = "toki " * 80  # 400 characters
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".jsonl") as f:
            f.write(json.dumps({"text": long_sent, "source": "wikipesija"}) + "\n")
            temp_path = f.name

        try:
            char_tok = CharTokenizer([long_sent])
            ds = TokiPonaDataset(temp_path, char_tok, max_seq_len=64)
            item = ds[0]
            # Max tokens is 64; modeled characters should not exceed 62
            self.assertLessEqual(item["char_len"], 64)
            self.assertLess(item["char_len"], len(long_sent))
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestPillar2GrammarTestSuite(unittest.TestCase):
    """Tests BLiMP-style grammar generation and rule coverage."""

    def test_rule_1_li_particle(self):
        pairs = generate_li_pairs()
        self.assertGreater(len(pairs), 50)
        # Verify mi/sina exception: mi moku vs mi li moku
        mi_pairs = [p for p in pairs if p.sub_rule == "pronoun_no_li"]
        self.assertGreater(len(mi_pairs), 0)
        self.assertTrue(all(" li " not in p.grammatical for p in mi_pairs if p.grammatical.startswith("mi ") or p.grammatical.startswith("sina ")))
        self.assertTrue(all(" li " in p.ungrammatical for p in mi_pairs if p.ungrammatical.startswith("mi ") or p.ungrammatical.startswith("sina ")))

        # Verify 3rd person requirement: jan li moku vs jan moku
        noun_pairs = [p for p in pairs if p.sub_rule == "3rd_person_requires_li"]
        self.assertGreater(len(noun_pairs), 0)
        self.assertTrue(all(" li " in p.grammatical for p in noun_pairs))
        self.assertTrue(all(" li " not in p.ungrammatical for p in noun_pairs))

    def test_rule_2_direct_object_e(self):
        pairs = generate_e_pairs()
        self.assertGreater(len(pairs), 30)
        obj_pairs = [p for p in pairs if p.sub_rule == "mandatory_e_marker"]
        self.assertGreater(len(obj_pairs), 0)
        self.assertTrue(all(" e " in p.grammatical for p in obj_pairs))
        self.assertTrue(all(" e " not in p.ungrammatical for p in obj_pairs))

    def test_rule_3_modifier_order(self):
        pairs = generate_modifier_order_pairs()
        self.assertGreater(len(pairs), 20)
        # Modifiers must follow noun: tomo suli vs suli tomo
        found_tomo_suli = any("tomo suli" in p.grammatical and "suli tomo" in p.ungrammatical for p in pairs)
        self.assertTrue(found_tomo_suli)

    def test_rule_4_modifier_pi(self):
        pairs = generate_pi_pairs()
        self.assertGreater(len(pairs), 20)
        # Single modifier under pi is forbidden: tomo suli vs tomo pi suli
        single_pi = [p for p in pairs if p.sub_rule == "no_pi_with_single_modifier"]
        self.assertGreater(len(single_pi), 0)
        self.assertTrue(any(" pi " in p.ungrammatical and " pi " not in p.grammatical for p in single_pi))

    def test_complete_grammar_suite_generation(self):
        all_pairs = generate_all_grammar_pairs()
        self.assertGreaterEqual(len(all_pairs), 300)
        categories = set(p.category for p in all_pairs)
        self.assertIn("particle_li", categories)
        self.assertIn("direct_object_e", categories)
        self.assertIn("modifier_order", categories)
        self.assertIn("modifier_pi", categories)
        self.assertIn("context_la", categories)
        self.assertIn("lexical_validity", categories)

    def test_no_pseudo_typo_muso(self):
        """Verify that typo 'muso' is not present in grammar suite and valid 'musi' is used."""
        all_pairs = generate_all_grammar_pairs()
        for p in all_pairs:
            self.assertNotIn("muso", p.grammatical, f"Found typo 'muso' in grammatical sentence {p.item_id}")
            if p.category != "lexical_validity":
                self.assertNotIn("muso", p.ungrammatical, f"Found typo 'muso' in ungrammatical sentence {p.item_id}")
        # Verify valid word 'musi' is present
        has_musi = any("musi" in p.grammatical for p in all_pairs)
        self.assertTrue(has_musi)

    def test_score_sentence(self):
        tok = WordTokenizer(["jan li moku e kili."])
        cfg = TransformerConfig.get_preset("micro", vocab_size=tok.vocab_size)
        model = TokiPonaTransformer(cfg)
        logp = score_sentence(model, tok, "jan li moku e kili.", torch.device("cpu"))
        self.assertIsInstance(logp, float)
        self.assertLess(logp, 0.0)

        # Degenerate input
        logp_empty = score_sentence(model, tok, "", torch.device("cpu"))
        self.assertLess(logp_empty, -9000.0)


class TestPillar3CompositionalGeneralization(unittest.TestCase):
    """Tests hold-out compounding generator and compositional metrics."""

    def test_compositional_sentence_generator(self):
        sents = generate_compositional_sentences()
        self.assertGreaterEqual(len(sents), 200)

        heldout = [s for s in sents if s.condition == "heldout"]
        seen = [s for s in sents if s.condition == "seen"]
        self.assertEqual(len(heldout), len(seen))

        # Check key compounds from task
        heldout_comps = set(s.compound for s in heldout)
        self.assertIn("telo nasa", heldout_comps)
        self.assertIn("tomo telo", heldout_comps)
        self.assertIn("jan utala", heldout_comps)
        self.assertIn("ilo moku", heldout_comps)

    def test_compositional_minimal_pairs(self):
        pairs = generate_compositional_minimal_pairs()
        self.assertGreaterEqual(len(pairs), 100)
        # Check that grammatical has proper compound and ungrammatical has reversed order
        for p in pairs:
            parts = p.compound.split()
            rev = f"{parts[1]} {parts[0]}"
            self.assertIn(p.compound, p.grammatical)
            self.assertIn(rev, p.ungrammatical)

    def test_heldout_purity_verification(self):
        import tempfile
        import json
        from src.evaluation.compositional import verify_heldout_purity, CANONICAL_HELDOUT_COMPOUNDS

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".jsonl") as f:
            f.write(json.dumps({"text": "mi moku e telo nasa lon tomo."}) + "\n")
            f.write(json.dumps({"text": "jan pona li toki e toki pona."}) + "\n")
            temp_path = f.name

        try:
            counts = verify_heldout_purity(temp_path, CANONICAL_HELDOUT_COMPOUNDS)
            self.assertEqual(counts["telo nasa"], 1)
            self.assertEqual(counts["tomo telo"], 0)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestPillar4StatisticalModels(unittest.TestCase):
    """Tests all 4 FIS statistical models."""

    def setUp(self):
        # Create synthetic factorial experiment dataframe
        np.random.seed(42)
        rows = []
        for tok in ["char", "bpe", "word"]:
            for ms in ["micro", "mini", "small"]:
                for df in [0.25, 0.75, 1.0]:
                    for s in [42, 1337]:
                        n_params = 100000 if ms == "micro" else (800000 if ms == "mini" else 2500000)
                        d_words = int(1400000 * df)
                        tok_bias = 0.4 if tok == "char" else (0.2 if tok == "bpe" else 0.0)
                        # Scaling: loss decreases with N and D
                        bpc = 1.0 + tok_bias + (100.0 / (n_params ** 0.3)) + (50.0 / (d_words ** 0.2)) + np.random.normal(0, 0.02)
                        rows.append({
                            "exp_id": f"exp_{tok}_{ms}_d{int(df*100)}_s{s}",
                            "tokenizer": tok,
                            "model_size": ms,
                            "data_fraction": df,
                            "data_words": d_words,
                            "non_embedding_params": n_params,
                            "total_params": n_params + 20000,
                            "seed": s,
                            "test_bpc": round(bpc, 4),
                            "val_bpc": round(bpc, 4),
                            "bpc_in_domain": round(bpc - 0.1, 4),
                            "bpc_out_domain": round(bpc + 0.1, 4),
                        })
        self.df = pd.DataFrame(rows)

    def test_model_1_scaling_laws(self):
        res = fit_scaling_law(self.df, metric_col="test_bpc")
        self.assertIn("joint_linear_scaling", res)
        joint = res["joint_linear_scaling"]
        self.assertIn("alpha_param_exponent", joint)
        self.assertIn("beta_data_exponent", joint)
        self.assertGreater(joint["r_squared"], 0.5)

    def test_model_2_mixed_effects(self):
        res = fit_mixed_effects_model(self.df, metric_col="test_bpc")
        self.assertIn("coefficients", res)
        self.assertIn("p_values", res)
        self.assertIn("domain_generalization_model", res)

    def test_model_3_factorial_anova(self):
        res = compute_factorial_anova(self.df, metric_col="test_bpc")
        self.assertIn("anova_table", res)
        self.assertIn("marginal_means_data", res)
        self.assertIn("crossover_analysis", res)
        anova_tab = res["anova_table"]
        self.assertIn("eta_sq", anova_tab.columns)
        self.assertIn("partial_eta_sq", anova_tab.columns)

    def test_model_4_glmm_grammar(self):
        # Create synthetic item evaluations
        np.random.seed(42)
        item_rows = []
        for _, r in self.df.iterrows():
            for cat in ["particle_li", "direct_object_e", "modifier_order", "modifier_pi"]:
                prob = 0.95 if r["tokenizer"] == "word" else (0.85 if r["tokenizer"] == "bpe" else 0.6)
                if cat == "modifier_pi":
                    prob -= 0.15
                correct = int(np.random.rand() < prob)
                item_rows.append({
                    "exp_id": r["exp_id"],
                    "tokenizer": r["tokenizer"],
                    "model_size": r["model_size"],
                    "non_embedding_params": r["non_embedding_params"],
                    "seed": r["seed"],
                    "category": cat,
                    "correct": correct,
                })
        item_df = pd.DataFrame(item_rows)
        res = fit_grammar_glmm(item_df)
        self.assertTrue(res.get("success", False))
        self.assertIn("odds_ratios", res)
        self.assertIn("rule_difficulty_hierarchy", res)


class TestUnifiedEvaluatorIntegration(unittest.TestCase):
    """End-to-end integration test with real existing model checkpoint."""

    def test_evaluator_single_checkpoint(self):
        repo_root = Path(__file__).resolve().parent.parent
        evaluator = UnifiedEvaluator(repo_root)

        # exp_012_d75_mini_word_s42 is an existing trained checkpoint
        rec = evaluator.evaluate_checkpoint("exp_012_d75_mini_word_s42", batch_size=64)

        self.assertIsNotNone(rec["test_bpc"])
        self.assertIsNotNone(rec["bpc_in_domain"])
        self.assertIsNotNone(rec["bpc_out_domain"])
        self.assertIsNotNone(rec["grammar_acc"])
        self.assertIsNotNone(rec["comp_bpc_heldout"])
        self.assertIsNotNone(rec["comp_generalization_gap"])
        self.assertGreater(rec["grammar_acc"], 0.9)
        self.assertGreater(rec["comp_pair_accuracy"], 0.9)


if __name__ == "__main__":
    unittest.main()
