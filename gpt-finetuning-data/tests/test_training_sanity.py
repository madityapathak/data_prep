"""
=============================================================================
Training Sanity Test Suite
=============================================================================

Verifies that the full training stack is wired correctly *before* spending
money on a rented GPU.  All tests are CPU-only and run in a few seconds —
no shard files or GPU needed.

What is tested
--------------
1.  GPTConfig / model construction      (TestModelConstruction)
2.  Forward pass + loss shape           (TestForwardPass)
3.  ignore_index=-100 loss masking      (TestLossMasking)
4.  Backward pass / gradient flow       (TestBackwardPass)
5.  configure_optimizers                (TestOptimizer)
6.  get_lr schedule                     (TestLRSchedule)
7.  Mini end-to-end training loop       (TestMiniTrainingLoop)
8.  Weight-tying (wte == lm_head)       (TestWeightTying)
9.  GPTConfig default vocab_size check  (TestVocabSize)

USAGE
-----
  # Run from project root:
  python -m pytest tests/test_training_sanity.py -v

  # Stop on first failure:
  python -m pytest tests/test_training_sanity.py -v -x

  # Run directly (no pytest):
  python tests/test_training_sanity.py

NOTE: No GPU needed. All tests are CPU-only.
=============================================================================
"""

import sys
import os
import math
import unittest

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup — works both from project root and from tests/ subfolder
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import GPT, GPTConfig  # UPDATED: imported from model.py

# ---------------------------------------------------------------------------
# A tiny GPT config — fast on CPU, big enough to be meaningful
# ---------------------------------------------------------------------------
TINY_CONFIG = GPTConfig(
    block_size=64,
    vocab_size=50263,   # UPDATED: must match enc.n_vocab (50257 + 6 custom tokens)
    n_layer=2,
    n_head=2,
    n_embd=64,
)

B = 2   # batch size
T = 32  # sequence length (must be <= block_size)


def _make_tiny_model() -> GPT:
    """Return a freshly initialised tiny GPT on CPU."""
    torch.manual_seed(42)
    model = GPT(TINY_CONFIG)
    model.eval()
    return model


def _synthetic_batch(vocab_size: int = TINY_CONFIG.vocab_size,
                     ignore_frac: float = 0.5):
    """
    Return a (inputs, targets) pair of shape (B, T) with real token IDs.

    ``ignore_frac`` of target positions are set to -100 to simulate
    USER-turn / PAD masking — same as the real dataloader does.
    """
    torch.manual_seed(0)
    inputs  = torch.randint(0, vocab_size, (B, T))
    targets = torch.randint(0, vocab_size, (B, T))

    # Randomly mask ~50 % of positions (simulate USER + PAD tokens)
    mask = torch.rand(B, T) < ignore_frac
    targets[mask] = -100

    return inputs, targets


# =============================================================================
# 1. Model Construction Tests
# =============================================================================

class TestModelConstruction(unittest.TestCase):
    """GPT model must build without errors for a range of configs."""

    def test_tiny_model_constructs(self):
        """The shared TINY_CONFIG must produce a GPT instance."""
        model = _make_tiny_model()
        self.assertIsInstance(model, GPT)

    def test_default_config_constructs(self):
        """GPTConfig() with all defaults (vocab_size=50263) must work."""
        model = GPT(GPTConfig())
        self.assertIsInstance(model, GPT)

    def test_parameter_count_is_positive(self):
        model = _make_tiny_model()
        n_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(n_params, 0, "Model has no parameters.")

    def test_model_is_on_cpu(self):
        model = _make_tiny_model()
        for p in model.parameters():
            self.assertEqual(p.device.type, "cpu",
                             "A parameter is not on CPU.")
            break  # checking one is enough


# =============================================================================
# 2. Forward Pass Tests
# =============================================================================

class TestForwardPass(unittest.TestCase):
    """forward() must return logits of the right shape."""

    @classmethod
    def setUpClass(cls):
        cls.model = _make_tiny_model()

    def test_logits_shape_no_targets(self):
        """Without targets, loss must be None and logits (B, T, vocab_size)."""
        inputs = torch.randint(0, TINY_CONFIG.vocab_size, (B, T))
        logits, loss = self.model(inputs)
        self.assertIsNone(loss)
        self.assertEqual(logits.shape, (B, T, TINY_CONFIG.vocab_size))

    def test_logits_shape_with_targets(self):
        """With targets, both logits and loss must be returned."""
        inputs, targets = _synthetic_batch()
        logits, loss = self.model(inputs, targets)
        self.assertIsNotNone(loss)
        self.assertEqual(logits.shape, (B, T, TINY_CONFIG.vocab_size))

    def test_loss_is_scalar(self):
        inputs, targets = _synthetic_batch()
        _, loss = self.model(inputs, targets)
        self.assertEqual(loss.dim(), 0, "Loss must be a scalar (0-d tensor).")

    def test_loss_is_finite(self):
        inputs, targets = _synthetic_batch()
        _, loss = self.model(inputs, targets)
        self.assertTrue(torch.isfinite(loss).item(),
                        f"Loss is not finite: {loss.item()}")

    def test_loss_is_positive(self):
        inputs, targets = _synthetic_batch()
        _, loss = self.model(inputs, targets)
        self.assertGreater(loss.item(), 0.0, "Loss should be > 0 on random data.")

    def test_sequence_length_assert(self):
        """Forward must raise AssertionError for T > block_size."""
        inputs = torch.randint(0, TINY_CONFIG.vocab_size,
                               (1, TINY_CONFIG.block_size + 1))
        with self.assertRaises(AssertionError):
            self.model(inputs)


# =============================================================================
# 3. Loss Masking Tests  (UPDATED: verifies ignore_index=-100 behaviour)
# =============================================================================

class TestLossMasking(unittest.TestCase):
    """
    Verify that -100 positions do NOT contribute to the loss.

    This is the core behaviour that allows the model to train only on
    ASSISTANT responses while ignoring USER turns and PAD tokens.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = _make_tiny_model()

    def test_fully_masked_targets_raises_or_nan(self):
        """
        If ALL target positions are -100 cross_entropy returns NaN.
        Documents the expected behaviour for fully-masked batches.
        """
        inputs  = torch.randint(0, TINY_CONFIG.vocab_size, (B, T))
        targets = torch.full((B, T), -100, dtype=torch.long)
        _, loss = self.model(inputs, targets)
        # PyTorch returns NaN when there are no valid targets
        self.assertTrue(
            torch.isnan(loss) or not torch.isfinite(loss),
            "Expected NaN loss when all targets are masked."
        )

    def test_partial_mask_loss_differs_from_no_mask(self):
        """
        Loss computed with some -100 positions must differ from loss where
        ALL positions are active (different positions contribute).
        """
        torch.manual_seed(7)
        inputs  = torch.randint(0, TINY_CONFIG.vocab_size, (B, T))
        targets = torch.randint(0, TINY_CONFIG.vocab_size, (B, T))

        _, loss_full = self.model(inputs, targets)

        targets_masked = targets.clone()
        targets_masked[:, :T // 2] = -100   # mask first half
        _, loss_masked = self.model(inputs, targets_masked)

        self.assertNotAlmostEqual(
            loss_full.item(), loss_masked.item(), places=4,
            msg="Masking half the targets should change the loss value."
        )

    def test_single_unmasked_position_produces_finite_loss(self):
        """Even one valid target position must yield a finite loss."""
        inputs  = torch.randint(0, TINY_CONFIG.vocab_size, (B, T))
        targets = torch.full((B, T), -100, dtype=torch.long)
        targets[0, 0] = torch.randint(0, TINY_CONFIG.vocab_size, (1,)).item()
        _, loss = self.model(inputs, targets)
        self.assertTrue(torch.isfinite(loss).item(),
                        "Loss must be finite with one valid target.")


# =============================================================================
# 4. Backward Pass / Gradient Flow Tests
# =============================================================================

class TestBackwardPass(unittest.TestCase):
    """Gradients must flow to every trainable parameter."""

    def test_gradients_flow_to_all_parameters(self):
        """After one backward pass every parameter must have a gradient."""
        model = _make_tiny_model()
        model.train()
        inputs, targets = _synthetic_batch()
        _, loss = model(inputs, targets)
        loss.backward()

        params_without_grad = [
            name for name, p in model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        self.assertEqual(params_without_grad, [],
            f"Parameters with no gradient after backward: {params_without_grad}")

    def test_gradients_are_finite(self):
        """Every gradient must be finite (no NaN / Inf)."""
        model = _make_tiny_model()
        model.train()
        inputs, targets = _synthetic_batch()
        _, loss = model(inputs, targets)
        loss.backward()

        bad_params = [
            name for name, p in model.named_parameters()
            if p.requires_grad and p.grad is not None and
               not torch.isfinite(p.grad).all()
        ]
        self.assertEqual(bad_params, [],
            f"Non-finite gradients in: {bad_params}")

    def test_gradient_clipping_does_not_error(self):
        """torch.nn.utils.clip_grad_norm_ must not raise."""
        model = _make_tiny_model()
        model.train()
        inputs, targets = _synthetic_batch()
        _, loss = model(inputs, targets)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.assertTrue(torch.isfinite(norm).item(),
                        f"Gradient norm is not finite: {norm}")


# =============================================================================
# 5. Optimizer Tests
# =============================================================================

class TestOptimizer(unittest.TestCase):
    """configure_optimizers() must build a valid AdamW optimizer."""

    @classmethod
    def setUpClass(cls):
        cls.model = _make_tiny_model()
        cls.optimizer = cls.model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=3e-4,
            device_type="cpu",
        )

    def test_optimizer_is_adamw(self):
        self.assertIsInstance(self.optimizer, torch.optim.AdamW)

    def test_two_param_groups(self):
        """configure_optimizers must produce exactly 2 groups (decay / nodecay)."""
        self.assertEqual(len(self.optimizer.param_groups), 2)

    def test_decay_group_has_weight_decay(self):
        decay_group = next(
            g for g in self.optimizer.param_groups if g["weight_decay"] > 0
        )
        self.assertAlmostEqual(decay_group["weight_decay"], 0.1)

    def test_nodecay_group_has_zero_weight_decay(self):
        nodecay_group = next(
            g for g in self.optimizer.param_groups if g["weight_decay"] == 0.0
        )
        self.assertEqual(nodecay_group["weight_decay"], 0.0)

    def test_all_params_covered(self):
        """Every trainable parameter must appear in exactly one group."""
        all_model_params = {
            id(p) for p in self.model.parameters() if p.requires_grad
        }
        all_optim_params = {
            id(p)
            for group in self.optimizer.param_groups
            for p in group["params"]
        }
        self.assertEqual(all_model_params, all_optim_params,
            "Some parameters are missing from or duplicated in the optimizer.")

    def test_optimizer_step_changes_parameters(self):
        """An optimizer.step() must actually change at least one parameter."""
        model = _make_tiny_model()
        model.train()
        optimizer = model.configure_optimizers(0.1, 3e-4, "cpu")

        params_before = {
            name: p.data.clone()
            for name, p in model.named_parameters()
        }

        inputs, targets = _synthetic_batch()
        _, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()

        changed = any(
            not torch.equal(params_before[name], p.data)
            for name, p in model.named_parameters()
        )
        self.assertTrue(changed,
            "No parameter changed after optimizer.step() — something is broken.")


# =============================================================================
# 6. Learning Rate Schedule Tests  (mirrors get_lr() in train_model.py)
# =============================================================================

class TestLRSchedule(unittest.TestCase):
    """
    Inline copy of get_lr() to verify the schedule without importing
    train_model.py (which would trigger DDP / data-loader setup at module level).
    """

    MAX_LR       = 6e-5
    MIN_LR       = MAX_LR * 0.1
    WARMUP_STEPS = 100
    MAX_STEPS    = 5000

    @staticmethod
    def get_lr(step: int) -> float:
        max_lr       = TestLRSchedule.MAX_LR
        min_lr       = TestLRSchedule.MIN_LR
        warmup_steps = TestLRSchedule.WARMUP_STEPS
        max_steps    = TestLRSchedule.MAX_STEPS

        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        if step > max_steps:
            return min_lr
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    def test_lr_at_step_zero_is_small(self):
        lr = self.get_lr(0)
        self.assertLess(lr, self.MAX_LR,
            "Step 0 LR should be less than max_lr (warmup).")

    def test_lr_at_warmup_peak(self):
        """At the end of warmup, LR should equal max_lr."""
        lr = self.get_lr(self.WARMUP_STEPS)
        self.assertAlmostEqual(lr, self.MAX_LR, places=10)

    def test_lr_at_max_steps_is_min_lr(self):
        """At max_steps, cosine decay reaches min_lr."""
        lr = self.get_lr(self.MAX_STEPS)
        self.assertAlmostEqual(lr, self.MIN_LR, places=10)

    def test_lr_after_max_steps_is_min_lr(self):
        lr = self.get_lr(self.MAX_STEPS + 100)
        self.assertAlmostEqual(lr, self.MIN_LR, places=10)

    def test_lr_is_monotonically_decaying_after_warmup(self):
        """LR must not increase after the warmup peak."""
        lrs = [self.get_lr(s) for s in range(self.WARMUP_STEPS, self.MAX_STEPS, 100)]
        for i in range(len(lrs) - 1):
            self.assertGreaterEqual(lrs[i], lrs[i + 1],
                f"LR increased at step {self.WARMUP_STEPS + i * 100}: "
                f"{lrs[i]:.2e} -> {lrs[i+1]:.2e}")

    def test_all_lrs_in_valid_range(self):
        """Every LR value must be in [MIN_LR, MAX_LR]."""
        # Warmup region (steps < WARMUP_STEPS) can go below MIN_LR by design
        # Only check the range from end-of-warmup onward
        for step in range(self.WARMUP_STEPS, self.MAX_STEPS + 200, 50):
            lr = self.get_lr(step)
            self.assertGreaterEqual(lr, self.MIN_LR - 1e-12,
                f"LR below MIN_LR at step {step}: {lr:.2e}")
            self.assertLessEqual(lr, self.MAX_LR + 1e-12,
                f"LR above MAX_LR at step {step}: {lr:.2e}")


# =============================================================================
# 7. Mini End-to-End Training Loop  (THE KEY SANITY CHECK before GPU rent)
# =============================================================================

class TestMiniTrainingLoop(unittest.TestCase):
    """
    Run 10 gradient-update steps on synthetic data and assert that:
      1. No exception is raised.
      2. Loss decreases — on fixed random data the model should over-fit.
      3. Gradient norms stay finite throughout.
    """

    NUM_STEPS = 10

    def _run_loop(self):
        """
        Returns list of (step, loss, grad_norm) tuples from the mini loop.
        Uses a high LR so the tiny model can memorise random data in 10 steps.
        """
        torch.manual_seed(999)
        model = GPT(TINY_CONFIG)
        model.train()
        optimizer = model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=1e-2,   # high LR to see loss drop on toy data
            device_type="cpu",
        )

        # Fixed synthetic batch — model should over-fit it quickly
        inputs, targets = _synthetic_batch(ignore_frac=0.3)

        history = []
        for step in range(self.NUM_STEPS):
            optimizer.zero_grad()
            _, loss = model(inputs, targets)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            history.append((step, loss.item(), norm.item()))

        return history

    def test_no_exception_during_loop(self):
        """The training loop must complete 10 steps without raising."""
        try:
            self._run_loop()
        except Exception as exc:
            self.fail(f"Mini training loop raised an exception: {exc}")

    def test_loss_decreases_over_steps(self):
        """
        Loss at the end of 10 steps must be lower than at step 0.
        With LR=1e-2 the tiny model over-fits random data quickly.
        """
        history = self._run_loop()
        loss_start = history[0][1]
        loss_end   = history[-1][1]
        self.assertLess(loss_end, loss_start,
            f"Loss did not decrease: {loss_start:.4f} -> {loss_end:.4f}. "
            f"The model may not be training correctly.")

    def test_loss_remains_finite_throughout(self):
        """Loss must be finite at every step."""
        history = self._run_loop()
        for step, loss, _ in history:
            self.assertTrue(
                math.isfinite(loss),
                f"Loss became non-finite at step {step}: {loss}"
            )

    def test_grad_norms_are_finite_throughout(self):
        """Gradient norms must be finite at every step."""
        history = self._run_loop()
        for step, _, norm in history:
            self.assertTrue(
                math.isfinite(norm),
                f"Gradient norm became non-finite at step {step}: {norm}"
            )


# =============================================================================
# 8. Weight Tying Test
# =============================================================================

class TestWeightTying(unittest.TestCase):
    """
    GPT uses tied weights: the input embedding (wte) and the output
    projection (lm_head) must share the exact same parameter tensor.
    """

    def test_wte_and_lm_head_share_weights(self):
        model = _make_tiny_model()
        self.assertIs(
            model.transformer.wte.weight,
            model.lm_head.weight,
            "wte.weight and lm_head.weight are not the same tensor — "
            "weight tying is broken."
        )

    def test_weight_shape_matches_vocab_and_embedding(self):
        model = _make_tiny_model()
        expected = (TINY_CONFIG.vocab_size, TINY_CONFIG.n_embd)
        self.assertEqual(model.lm_head.weight.shape, expected,
            f"lm_head weight shape {model.lm_head.weight.shape} != {expected}")


# =============================================================================
# 9. Vocab Size Tests  (UPDATED: ensures the fix in model.py is correct)
# =============================================================================

class TestVocabSize(unittest.TestCase):
    """
    GPTConfig default vocab_size must be 50263 (50257 base + 6 custom tokens).
    This is the key fix made in this session — if this breaks, the model will
    have mis-aligned embeddings with the tokenizer.
    """

    # UPDATED: 50257 base GPT-2 + 6 custom special tokens
    EXPECTED_VOCAB = 50257 + 6   # = 50263

    def test_default_vocab_size(self):
        """GPTConfig() with no args must default to 50263."""
        config = GPTConfig()
        self.assertEqual(config.vocab_size, self.EXPECTED_VOCAB,
            f"GPTConfig.vocab_size={config.vocab_size}, expected {self.EXPECTED_VOCAB}. "
            f"Did you forget to update the TODO in model.py?")

    def test_tiny_config_uses_correct_vocab(self):
        """TINY_CONFIG must also use the correct vocab size."""
        self.assertEqual(TINY_CONFIG.vocab_size, self.EXPECTED_VOCAB)

    def test_embedding_table_has_correct_size(self):
        """wte embedding table rows must equal vocab_size."""
        model = GPT(GPTConfig(
            vocab_size=self.EXPECTED_VOCAB,
            n_layer=1, n_head=1, n_embd=64, block_size=16
        ))
        self.assertEqual(model.transformer.wte.weight.shape[0],
                         self.EXPECTED_VOCAB)

    def test_lm_head_output_size(self):
        """lm_head must project to vocab_size logits."""
        model = GPT(GPTConfig(
            vocab_size=self.EXPECTED_VOCAB,
            n_layer=1, n_head=1, n_embd=64, block_size=16
        ))
        inputs = torch.randint(0, self.EXPECTED_VOCAB, (1, 8))
        logits, _ = model(inputs)
        self.assertEqual(logits.shape[-1], self.EXPECTED_VOCAB)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Training Sanity Test Suite  (CPU-only, no GPU / shards needed)")
    print("=" * 70)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Tiny config  : {TINY_CONFIG}")
    print(f"  Batch        : B={B}, T={T}")
    print("=" * 70)
    unittest.main(verbosity=2)
