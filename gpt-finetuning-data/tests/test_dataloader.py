"""
=============================================================================
Comprehensive Test Suite for ConversationDataLoader
=============================================================================

USAGE
-----
Run from the PROJECT ROOT (gpt-finetuning-data/) so that relative imports
(data_sanitizer, shards/) resolve correctly.

  # Basic run (all tests, verbose):
  python -m pytest tests/test_dataloader.py -v

  # Run only a specific test class:
  python -m pytest tests/test_dataloader.py::TestBatchProperties -v

  # Run only a specific test:
  python -m pytest tests/test_dataloader.py::TestNoDataDuplication::test_single_process_no_duplicate_tokens -v

  # Stop at first failure:
  python -m pytest tests/test_dataloader.py -v -x

  # Show print() output live (useful for debugging):
  python -m pytest tests/test_dataloader.py -v -s

  # Run without pytest (plain Python):
  python tests/test_dataloader.py

NOTE: No GPU is needed. All tests are CPU-only.
=============================================================================
"""

import sys
import os
import math
import unittest
from collections import Counter

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Make sure the project root is on the path so that imports work whether we
# run via `pytest` from the project root or via `python tests/test_dataloader.py`
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataloader import ConversationDataLoader, load_shard_tokens
from data_sanitizer.utils import enc, special_tokens

# ---------------------------------------------------------------------------
# Constants – edit if your actual shard layout differs
# ---------------------------------------------------------------------------
SHARDS_ROOT   = os.path.join(PROJECT_ROOT, "shards")
BATCH_SIZE    = 2
SEQ_LEN       = 1024
VOCAB_SIZE    = enc.n_vocab   # 50257 + 6 custom = 50263

# Derived
BT = BATCH_SIZE * SEQ_LEN   # tokens consumed per batch step per process


# =============================================================================
# Helper utilities
# =============================================================================

def _count_shard_tokens(split: str) -> int:
    """Return total number of tokens across all shards for a split."""
    shard_dir = os.path.join(SHARDS_ROOT, split)
    total = 0
    for fname in sorted(os.listdir(shard_dir)):
        if fname.startswith(split) and fname.endswith(".npy"):
            arr = np.load(os.path.join(shard_dir, fname))
            total += len(arr)
    return total


def _batches_in_single_pass(split: str, num_processes: int = 1) -> int:
    """
    Conservative lower bound on how many batches exist before the loader
    must wrap around to shard 0.

    Each shard yields floor((shard_tokens - 1) / (B*T*num_processes)) steps
    before the position check triggers _advance_to_next_shard.
    """
    shard_dir = os.path.join(SHARDS_ROOT, split)
    total_batches = 0
    for fname in sorted(os.listdir(shard_dir)):
        if fname.startswith(split) and fname.endswith(".npy"):
            arr = np.load(os.path.join(shard_dir, fname))
            n = len(arr)
            step = BT * num_processes
            usable = n - BT * (num_processes - 1)
            if usable > 0:
                total_batches += max(0, (usable - 1) // step)
    return total_batches


def _make_loader(split="train", process_rank=0, num_processes=1,
                 batch_size=BATCH_SIZE, seq_len=SEQ_LEN):
    return ConversationDataLoader(
        batch_size=batch_size,
        sequence_length=seq_len,
        process_rank=process_rank,
        num_processes=num_processes,
        split=split,
        data_root=SHARDS_ROOT,
    )


# =============================================================================
# 1.  Smoke / Sanity Tests
# =============================================================================

class TestSmoke(unittest.TestCase):
    """Fast sanity checks – run these first."""

    def test_loader_constructs_train(self):
        """DataLoader must construct without error for split='train'."""
        loader = _make_loader(split="train")
        self.assertIsInstance(loader, ConversationDataLoader)

    def test_loader_constructs_val(self):
        """DataLoader must construct without error for split='val'."""
        loader = _make_loader(split="val")
        self.assertIsInstance(loader, ConversationDataLoader)

    def test_shard_files_found(self):
        """At least one shard must be discovered."""
        loader = _make_loader()
        self.assertGreater(len(loader.shard_files), 0,
                           "No shard files found – run the data prep pipeline first.")

    def test_invalid_split_raises(self):
        """Constructing with an unsupported split should raise AssertionError."""
        with self.assertRaises(AssertionError):
            ConversationDataLoader(
                batch_size=BATCH_SIZE,
                sequence_length=SEQ_LEN,
                split="test",
                data_root=SHARDS_ROOT,
            )

    def test_missing_shard_dir_raises(self):
        """Pointing to a non-existent data_root should raise ValueError."""
        with self.assertRaises(ValueError):
            ConversationDataLoader(
                batch_size=BATCH_SIZE,
                sequence_length=SEQ_LEN,
                split="train",
                data_root="/tmp/__nonexistent_shards__",
            )

    def test_get_stats_keys(self):
        """get_stats() must return the expected dictionary keys."""
        loader = _make_loader()
        stats = loader.get_stats()
        required_keys = {"split", "num_shards", "total_tokens",
                         "shard_info", "avg_tokens_per_shard"}
        self.assertTrue(required_keys.issubset(stats.keys()))

    def test_get_stats_positive_token_count(self):
        """Total token count reported by get_stats() must be positive."""
        loader = _make_loader()
        stats = loader.get_stats()
        self.assertGreater(stats["total_tokens"], 0)


# =============================================================================
# 2.  Batch Shape & Type Tests
# =============================================================================

class TestBatchProperties(unittest.TestCase):
    """Verify shape, dtype, device, and value range of each batch."""

    @classmethod
    def setUpClass(cls):
        cls.loader = _make_loader()
        cls.inputs, cls.targets = cls.loader.next_batch()

    def test_input_shape(self):
        """Input tensor must be [batch_size, seq_len]."""
        self.assertEqual(self.inputs.shape, (BATCH_SIZE, SEQ_LEN))

    def test_target_shape(self):
        """Target tensor must be [batch_size, seq_len]."""
        self.assertEqual(self.targets.shape, (BATCH_SIZE, SEQ_LEN))

    def test_input_dtype_is_long(self):
        """Input tokens must be torch.long (int64)."""
        self.assertEqual(self.inputs.dtype, torch.long)

    def test_target_dtype_is_long(self):
        """Target tokens must be torch.long (int64)."""
        self.assertEqual(self.targets.dtype, torch.long)

    def test_input_on_cpu(self):
        """Without explicit GPU placement tensors must reside on CPU."""
        self.assertEqual(self.inputs.device.type, "cpu")

    def test_target_on_cpu(self):
        self.assertEqual(self.targets.device.type, "cpu")

    def test_token_ids_in_valid_range(self):
        """Every token ID must be in [0, VOCAB_SIZE)."""
        self.assertTrue((self.inputs >= 0).all().item(),
                        "Negative token IDs found in inputs.")
        self.assertTrue((self.inputs < VOCAB_SIZE).all().item(),
                        f"Token IDs >= vocab_size ({VOCAB_SIZE}) found in inputs.")
        self.assertTrue((self.targets >= 0).all().item(),
                        "Negative token IDs found in targets.")
        self.assertTrue((self.targets < VOCAB_SIZE).all().item(),
                        f"Token IDs >= vocab_size ({VOCAB_SIZE}) found in targets.")

    def test_input_target_causal_shift(self):
        """
        Targets must be inputs shifted by one position.
        Within each row: inputs[b, 1:] == targets[b, :-1]
        """
        for b in range(BATCH_SIZE):
            self.assertTrue(
                torch.equal(self.inputs[b, 1:], self.targets[b, :-1]),
                f"Row {b}: input[1:] != target[:-1] – causal shift is broken."
            )

    def test_consecutive_batches_are_not_identical(self):
        """Two back-to-back batches must not be byte-for-byte equal."""
        loader = _make_loader()
        inp1, tgt1 = loader.next_batch()
        inp2, tgt2 = loader.next_batch()
        self.assertFalse(torch.equal(inp1, inp2),
                         "Consecutive input batches are identical – position not advancing.")
        self.assertFalse(torch.equal(tgt1, tgt2),
                         "Consecutive target batches are identical – position not advancing.")

    def test_batch_position_advances_correctly(self):
        """
        After one next_batch() call the internal position must have advanced
        by exactly B*T*num_processes tokens.
        """
        loader = _make_loader(num_processes=1)
        pos_before = loader.current_position
        loader.next_batch()
        pos_after = loader.current_position
        self.assertEqual(pos_after - pos_before, BATCH_SIZE * SEQ_LEN)


# =============================================================================
# 3.  No-Duplication Tests  <- THE CORE REQUIREMENT
# =============================================================================

class TestNoDataDuplication(unittest.TestCase):
    """
    Verify that within one full pass over all shards no token position is
    delivered twice, AND that after reset() the loader faithfully loops
    back to the very beginning.
    """

    @staticmethod
    def _collect_positions(loader, max_steps: int):
        """
        Drain the loader for `max_steps` batches and return a list of
        (shard_idx, position) tuples recorded *before* each call.
        """
        positions = []
        for _ in range(max_steps):
            shard_before = loader.current_shard_idx
            pos_before   = loader.current_position
            loader.next_batch()
            positions.append((shard_before, pos_before))
        return positions

    def test_single_process_no_duplicate_positions(self):
        """
        Within one epoch (before shard index wraps back) no
        (shard_idx, start_pos) pair should appear twice.
        """
        loader = _make_loader(num_processes=1)
        n_batches = _batches_in_single_pass("train", num_processes=1)
        if n_batches < 2:
            self.skipTest("Not enough data for duplication check.")

        positions = self._collect_positions(loader, n_batches)
        counts = Counter(positions)
        dupes = {k: v for k, v in counts.items() if v > 1}
        self.assertEqual(len(dupes), 0,
            f"Duplicate (shard, pos) pairs found within one pass: {dupes}")

    def test_multi_process_no_overlap_between_ranks(self):
        """
        With num_processes=2, rank-0 and rank-1 must never read overlapping
        token windows within the same shard.
        """
        num_proc = 2
        n_batches = _batches_in_single_pass("train", num_processes=num_proc)
        if n_batches < 2:
            self.skipTest("Not enough data for multi-process overlap check.")

        steps_per_rank = max(1, n_batches // num_proc)
        windows = {0: [], 1: []}

        for rank in range(num_proc):
            loader = _make_loader(process_rank=rank, num_processes=num_proc, split="train")
            positions = self._collect_positions(loader, steps_per_rank)
            for shard_idx, pos in positions:
                windows[rank].append((shard_idx, pos, pos + BT))  # each rank reads exactly BT tokens

        overlaps = []
        for s0, start0, end0 in windows[0]:
            for s1, start1, end1 in windows[1]:
                if s0 == s1:
                    if start0 < end1 and start1 < end0:
                        overlaps.append(((s0, start0, end0), (s1, start1, end1)))

        self.assertEqual(len(overlaps), 0,
            f"Rank 0 and rank 1 read overlapping token windows:\n{overlaps[:5]}")

    def test_multi_process_union_covers_all_tokens(self):
        """
        The union of what rank-0 and rank-1 read must form a contiguous
        B*T-spaced sequence with no gaps within every shard.

        We compute the expected positions analytically per shard (using
        the same offset/stride logic the loader uses) to avoid the epoch-
        boundary confusion that arises when driving a live loader for a
        global step total.
        """
        num_proc = 2
        loader   = _make_loader(process_rank=0, num_processes=num_proc)
        if len(loader.shard_files) < 2:
            self.skipTest("Need at least 2 shards.")

        step = BT * num_proc  # stride between consecutive reads by the same rank

        for shard_path in loader.shard_files:
            n       = len(np.load(shard_path))
            shard   = os.path.basename(shard_path)

            # Positions each rank would visit in this shard
            combined = set()
            for rank in range(num_proc):
                start = rank * BT
                pos   = start
                while pos + BT + 1 <= n:   # same condition as the loader
                    combined.add(pos)
                    pos += step

            if not combined:
                continue  # shard too small for even one batch per rank

            sorted_pos = sorted(combined)
            gaps = [
                (i, i * BT, pos)
                for i, pos in enumerate(sorted_pos)
                if pos != i * BT
            ]
            self.assertEqual(
                len(gaps), 0,
                f"{shard}: combined rank positions are not contiguous "
                f"(step={BT}). First mismatches: {gaps[:3]}"
            )


    def test_reset_loops_back_to_start(self):
        """
        After exhausting shard 0 and calling reset(), the loader must
        reproduce the very same first batch.
        """
        loader = _make_loader()
        inp_first, tgt_first = loader.next_batch()

        max_steps = _batches_in_single_pass("train") + 5
        for _ in range(max_steps):
            loader.next_batch()

        loader.reset()
        inp_reset, tgt_reset = loader.next_batch()

        self.assertTrue(torch.equal(inp_first, inp_reset),
            "After reset(), first input batch does not match original first batch.")
        self.assertTrue(torch.equal(tgt_first, tgt_reset),
            "After reset(), first target batch does not match original first batch.")

    def test_pool_exhaustion_advances_shard(self):
        """
        When the loader runs out of tokens in shard N it must advance
        to shard N+1 automatically (not skip or crash).
        """
        loader = _make_loader()
        num_shards = len(loader.shard_files)
        if num_shards < 2:
            self.skipTest("Need at least 2 shards to test shard advancement.")

        visited_shards = set()
        n_steps = _batches_in_single_pass("train") + num_shards + 10
        for _ in range(n_steps):
            visited_shards.add(loader.current_shard_idx)
            loader.next_batch()

        self.assertEqual(visited_shards, set(range(num_shards)),
            f"Not all shards were visited. Visited: {sorted(visited_shards)}, "
            f"expected: {list(range(num_shards))}")

    def test_shard_cycling_wraps_to_zero(self):
        """
        After the last shard the index must wrap back to 0 and training
        must continue without raising any exception.
        """
        loader = _make_loader()
        n_shards = len(loader.shard_files)
        n_steps  = _batches_in_single_pass("train") + n_shards * 2 + 20
        try:
            for _ in range(n_steps):
                loader.next_batch()
        except Exception as exc:
            self.fail(f"next_batch() raised an exception during shard cycling: {exc}")

        self.assertIn(loader.current_shard_idx, range(n_shards))


# =============================================================================
# 4.  Token Content Correctness Tests
# =============================================================================

class TestTokenContent(unittest.TestCase):
    """Verify that the tokens stored in shards are sane."""

    @classmethod
    def setUpClass(cls):
        cls.loader = _make_loader()

    def test_no_negative_tokens_in_shards(self):
        """Every shard must contain non-negative token IDs."""
        for path in self.loader.shard_files:
            arr = np.load(path).astype(np.int32)
            self.assertTrue(
                (arr >= 0).all(),
                f"Negative token IDs found in shard: {os.path.basename(path)}"
            )

    def test_no_out_of_range_tokens_in_shards(self):
        """Every token ID must be < VOCAB_SIZE."""
        for path in self.loader.shard_files:
            arr = np.load(path).astype(np.int32)
            self.assertTrue(
                (arr < VOCAB_SIZE).all(),
                f"Token ID >= vocab_size ({VOCAB_SIZE}) in: {os.path.basename(path)}"
            )

    def test_shard_dtype_is_uint16(self):
        """Raw shard files must be stored as uint16."""
        for path in self.loader.shard_files:
            arr = np.load(path)
            self.assertEqual(
                arr.dtype, np.uint16,
                f"Shard {os.path.basename(path)} has dtype {arr.dtype}, expected uint16."
            )

    def test_shards_non_empty(self):
        """Every shard must contain at least B*T+1 tokens to produce one batch."""
        min_tokens = BATCH_SIZE * SEQ_LEN + 1
        for path in self.loader.shard_files:
            arr = np.load(path)
            self.assertGreater(
                len(arr), min_tokens,
                f"Shard {os.path.basename(path)} has only {len(arr)} tokens; "
                f"need at least {min_tokens} for one batch."
            )

    def test_special_tokens_present_in_data(self):
        """
        At least one BOS and one EOT token must exist across training shards,
        confirming the custom tokenizer was used correctly.
        """
        bos_id = special_tokens["<|BOS|>"]
        eot_id = special_tokens["<|EOT|>"]
        found_bos = False
        found_eot = False

        for path in self.loader.shard_files:
            arr = np.load(path).astype(np.int32)
            if bos_id in arr:
                found_bos = True
            if eot_id in arr:
                found_eot = True
            if found_bos and found_eot:
                break

        self.assertTrue(found_bos,
            "<|BOS|> token never found in any shard – data may be mis-tokenised.")
        self.assertTrue(found_eot,
            "<|EOT|> token never found in any shard – data may be mis-tokenised.")

    def test_load_shard_tokens_returns_long_tensor(self):
        """load_shard_tokens() must return a 1-D torch.long tensor."""
        path   = self.loader.shard_files[0]
        tokens = load_shard_tokens(path)
        self.assertIsInstance(tokens, torch.Tensor)
        self.assertEqual(tokens.dtype, torch.long)
        self.assertEqual(tokens.dim(), 1)

    def test_load_shard_preserves_token_count(self):
        """load_shard_tokens() must not drop or add tokens vs. np.load."""
        path   = self.loader.shard_files[0]
        arr    = np.load(path)
        tokens = load_shard_tokens(path)
        self.assertEqual(len(tokens), len(arr))

    def test_contiguous_raw_token_stream(self):
        """
        The flattened input + last-target token from one batch must exactly
        match the corresponding slice in the raw shard array.
        """
        loader = _make_loader(num_processes=1)
        raw    = load_shard_tokens(loader.shard_files[0])
        start  = loader.current_position
        tokens_needed  = BATCH_SIZE * SEQ_LEN + 1
        expected_slice = raw[start: start + tokens_needed]

        inp, tgt = loader.next_batch()
        flat_inp  = inp.reshape(-1)
        last_tgt  = tgt[-1, -1].unsqueeze(0)
        reconstructed = torch.cat([flat_inp, last_tgt])

        self.assertTrue(
            torch.equal(reconstructed, expected_slice),
            "Batch tokens do not match the raw shard slice at the same offset."
        )


# =============================================================================
# 5.  get_stats() Accuracy Tests
# =============================================================================

class TestGetStats(unittest.TestCase):
    """Validate that get_stats() returns accurate metadata."""

    @classmethod
    def setUpClass(cls):
        cls.loader = _make_loader()
        cls.stats  = cls.loader.get_stats()

    def test_num_shards_matches_files(self):
        self.assertEqual(self.stats["num_shards"], len(self.loader.shard_files))

    def test_total_tokens_matches_disk(self):
        """Stats total_tokens must exactly match what is on disk."""
        disk_total = _count_shard_tokens("train")
        self.assertEqual(self.stats["total_tokens"], disk_total,
            f"Stats report {self.stats['total_tokens']} tokens but disk has {disk_total}.")

    def test_shard_info_length(self):
        self.assertEqual(len(self.stats["shard_info"]), self.stats["num_shards"])

    def test_avg_tokens_per_shard(self):
        expected_avg = self.stats["total_tokens"] / self.stats["num_shards"]
        self.assertAlmostEqual(self.stats["avg_tokens_per_shard"], expected_avg, places=1)

    def test_shard_info_has_required_keys(self):
        for info in self.stats["shard_info"]:
            self.assertIn("filename", info)
            self.assertIn("tokens",   info)

    def test_split_field_correct(self):
        self.assertEqual(self.stats["split"], "train")

    def test_val_split_stats(self):
        loader_val = _make_loader(split="val")
        stats_val  = loader_val.get_stats()
        self.assertEqual(stats_val["split"], "val")
        self.assertGreater(stats_val["total_tokens"], 0)


# =============================================================================
# 6.  Multi-Process Partitioning Tests
# =============================================================================

class TestMultiProcessPartitioning(unittest.TestCase):
    """
    Verify that process ranks correctly partition the token stream with no
    two ranks reading the same starting position within the same shard.
    """

    def _starting_positions(self, num_processes: int, split="train"):
        return [
            _make_loader(process_rank=r, num_processes=num_processes,
                         split=split).current_position
            for r in range(num_processes)
        ]

    def test_two_processes_start_at_different_positions(self):
        positions = self._starting_positions(2)
        self.assertNotEqual(positions[0], positions[1],
            "Rank 0 and rank 1 start at the same position – duplicated data!")

    def test_four_processes_all_unique_starting_positions(self):
        positions = self._starting_positions(4)
        self.assertEqual(len(set(positions)), 4,
            f"Not all 4 ranks start at unique positions: {positions}")

    def test_rank_zero_starts_at_zero(self):
        loader = _make_loader(process_rank=0, num_processes=1)
        self.assertEqual(loader.current_position, 0)

    def test_rank_offset_equals_rank_times_BT(self):
        """rank r must start at r * B * T."""
        for rank in range(4):
            loader = _make_loader(process_rank=rank, num_processes=4)
            expected_start = rank * BATCH_SIZE * SEQ_LEN
            self.assertEqual(
                loader.current_position, expected_start,
                f"Rank {rank}: expected start={expected_start}, "
                f"got {loader.current_position}"
            )

    def test_step_size_is_bt_times_num_processes(self):
        """
        After each next_batch() call position must advance by B*T*num_processes
        so all ranks advance in lock-step.
        """
        num_proc = 3
        for rank in range(num_proc):
            loader = _make_loader(process_rank=rank, num_processes=num_proc)
            pos_before = loader.current_position
            loader.next_batch()
            advance = loader.current_position - pos_before
            expected = BATCH_SIZE * SEQ_LEN * num_proc
            self.assertEqual(
                advance, expected,
                f"Rank {rank}: position advanced by {advance} but expected "
                f"{expected} (B*T*P = {BATCH_SIZE}*{SEQ_LEN}*{num_proc})."
            )


# =============================================================================
# 7.  Full-Epoch / Pool-Exhaustion Tests
# =============================================================================

class TestFullEpochExhaustion(unittest.TestCase):
    """
    Simulate running for a full epoch and confirm that:
      - All shards are visited
      - The loader wraps around seamlessly
      - No exception is raised when the token pool is exhausted and refilled
    """

    def test_all_shards_visited_after_one_epoch(self):
        loader   = _make_loader()
        n_shards = len(loader.shard_files)
        n_steps  = _batches_in_single_pass("train") + n_shards * 2

        shard_visit_counts = Counter()
        for _ in range(n_steps):
            shard_visit_counts[loader.current_shard_idx] += 1
            loader.next_batch()

        for shard_idx in range(n_shards):
            self.assertGreater(
                shard_visit_counts[shard_idx], 0,
                f"Shard {shard_idx} was never visited during epoch sweep."
            )

    def test_no_exception_over_full_epoch(self):
        """next_batch() must never raise for one full epoch + buffer steps."""
        loader  = _make_loader()
        n_steps = _batches_in_single_pass("train") + 50
        try:
            for _ in range(n_steps):
                loader.next_batch()
        except Exception as exc:
            self.fail(f"next_batch() raised during full-epoch sweep: {exc}")

    def test_second_epoch_same_first_batch(self):
        """
        After wrapping all shards, a reset() must reproduce the exact same
        first batch (determinism guarantee for reproducible training).
        """
        loader   = _make_loader()
        inp_e1, tgt_e1 = loader.next_batch()

        steps_to_cycle = _batches_in_single_pass("train") + len(loader.shard_files) * 2
        for _ in range(steps_to_cycle):
            loader.next_batch()

        loader.reset()
        inp_e2, tgt_e2 = loader.next_batch()

        self.assertTrue(torch.equal(inp_e1, inp_e2),
            "Second epoch first batch differs from first epoch first batch.")

    def test_two_complete_cycles_no_exception(self):
        """Stress-test: run 2 full epochs without any error."""
        loader  = _make_loader()
        n_steps = (_batches_in_single_pass("train") + 10) * 2
        try:
            for _ in range(n_steps):
                loader.next_batch()
        except Exception as exc:
            self.fail(f"next_batch() raised during 2-cycle stress test: {exc}")


# =============================================================================
# 8.  Edge-Case / Boundary Tests
# =============================================================================

class TestEdgeCases(unittest.TestCase):

    def test_batch_size_1_works(self):
        loader = _make_loader(batch_size=1)
        inp, tgt = loader.next_batch()
        self.assertEqual(inp.shape, (1, SEQ_LEN))
        self.assertEqual(tgt.shape, (1, SEQ_LEN))

    def test_large_batch_size_works(self):
        loader = _make_loader(batch_size=16, seq_len=64)
        inp, tgt = loader.next_batch()
        self.assertEqual(inp.shape, (16, 64))

    def test_large_seq_len_works(self):
        loader = _make_loader(batch_size=1, seq_len=512)
        inp, tgt = loader.next_batch()
        self.assertEqual(inp.shape, (1, 512))

    def test_reset_idempotent(self):
        """Calling reset() twice must produce the same starting state."""
        loader = _make_loader()
        loader.next_batch()
        loader.reset()
        pos1, shard1 = loader.current_position, loader.current_shard_idx
        loader.reset()
        pos2, shard2 = loader.current_position, loader.current_shard_idx
        self.assertEqual(pos1, pos2)
        self.assertEqual(shard1, shard2)

    def test_val_loader_works_independently(self):
        """Train and val loaders must operate without interfering."""
        train_loader = _make_loader(split="train")
        val_loader   = _make_loader(split="val")
        t_inp, _     = train_loader.next_batch()
        v_inp, _     = val_loader.next_batch()
        self.assertEqual(t_inp.shape, (BATCH_SIZE, SEQ_LEN))
        self.assertEqual(v_inp.shape, (BATCH_SIZE, SEQ_LEN))

    def test_shard_order_is_deterministic(self):
        """Two independently constructed loaders must list shards in the same order."""
        loader1 = _make_loader()
        loader2 = _make_loader()
        self.assertEqual(loader1.shard_files, loader2.shard_files)

    def test_tokens_tensor_matches_numpy_load(self):
        """
        loader.tokens after reset() must exactly match np.load() for shard 0.
        """
        loader = _make_loader()
        expected = torch.tensor(
            np.load(loader.shard_files[0]).astype(np.int32),
            dtype=torch.long,
        )
        self.assertTrue(torch.equal(loader.tokens, expected),
            "loader.tokens does not match raw np.load() of shard 0.")


# =============================================================================
# 9.  Tokenizer Sanity Tests
# =============================================================================

class TestTokenizerSanity(unittest.TestCase):
    """
    Quick checks that the custom encoder (enc) behaves as expected before
    we trust the data it produced.
    """

    def test_encode_decode_roundtrip(self):
        text    = "Hello, world!"
        tokens  = enc.encode(text)
        decoded = enc.decode(tokens)
        self.assertEqual(decoded, text)

    def test_special_tokens_have_correct_ids(self):
        """Custom special tokens must have IDs >= base GPT-2 vocab size (50257)."""
        base_vocab = 50257
        for name, tid in special_tokens.items():
            self.assertGreaterEqual(tid, base_vocab,
                f"{name} has ID {tid} which is < base vocab size {base_vocab}.")

    def test_special_tokens_unique(self):
        ids = list(special_tokens.values())
        self.assertEqual(len(ids), len(set(ids)),
            "Two or more custom special tokens share the same ID.")

    def test_vocab_size_consistent(self):
        """enc.n_vocab must equal base vocab + number of specials."""
        expected_vocab = 50257 + len(special_tokens)
        self.assertEqual(enc.n_vocab, expected_vocab,
            f"enc.n_vocab={enc.n_vocab} but expected {expected_vocab}.")

    def test_bos_token_encodes_as_single_id(self):
        """<|BOS|> must encode to exactly one token ID equal to special_tokens value."""
        ids = enc.encode("<|BOS|>", allowed_special="all")
        self.assertEqual(len(ids), 1, "<|BOS|> should encode to a single token.")
        self.assertEqual(ids[0], special_tokens["<|BOS|>"])

    def test_eot_token_encodes_as_single_id(self):
        """<|EOT|> must encode to exactly one token ID."""
        ids = enc.encode("<|EOT|>", allowed_special="all")
        self.assertEqual(len(ids), 1, "<|EOT|> should encode to a single token.")
        self.assertEqual(ids[0], special_tokens["<|EOT|>"])


# =============================================================================
# Entry point  (plain Python fallback – no pytest required)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  ConversationDataLoader – Comprehensive Test Suite")
    print("=" * 70)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Shards root  : {SHARDS_ROOT}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Seq length   : {SEQ_LEN}")
    print(f"  Vocab size   : {VOCAB_SIZE}")
    print("=" * 70)
    unittest.main(verbosity=2)
