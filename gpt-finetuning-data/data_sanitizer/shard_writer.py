"""
ShardWriter
===========
Writes already-tokenised, validated conversation sequences into fixed-size
NumPy shards using the FineWeb-style approach.

Design decisions
----------------
Shard size: 2**25 = 33,554,432 tokens  (~32 M tokens per shard)
- Power-of-two aligned → friendly for PyTorch DataLoader chunk slicing.
- At 1 024-token context length: 32,768 complete training windows per shard.
- ~64 MB on disk per shard (uint16, 2 bytes/token) — easy to memory-map.
- With ~147 M total tokens → ~4-5 full shards, manageable shard count.
- 2**26 (~64 M) would give only 2 shards which is too few for shuffling;
  2**24 (~16 M) gives ~9 shards which is fine but 32 M is the sweet spot.

Conversation-boundary guarantee
--------------------------------
Each conversation is kept intact within a single shard.  If the next
conversation would overflow the current shard the current shard is flushed
first and the conversation starts a new shard.  A single conversation that
exceeds the shard size is written out as its own oversized shard with a
warning rather than silently truncated or split.

Token integrity guarantee
--------------------------
Every token appears in exactly one shard — no gaps, no duplicates.
See verify_shard_integrity() for a post-hoc check.

Storage format
--------------
- dtype : np.uint16  (vocab size of custom gpt2_custom encoder ≤ 65 535)
- naming: {split}_{index:06d}.npy   e.g.  train_000000.npy
- the final shard may be shorter than SHARD_SIZE — no zero-padding is added.

Usage
-----
    from data_sanitizer.shard_writer import ShardWriter

    writer = ShardWriter(output_dir="shards/train", split="train")
    writer.write(validated_packed_sequences)
    writer.verify_shard_integrity(validated_packed_sequences)
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
from tqdm import tqdm

# Re-use the single encoder that the rest of the project uses.
from data_sanitizer.utils import enc, special_tokens


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 2^25 = 33 554 432 tokens ≈ 32 M tokens per shard
SHARD_SIZE: int = 2 ** 25

# GPT-2 base vocab has 50 257 tokens; our custom encoder adds 6 specials
# → maximum token ID = 50 262, well within uint16 range (65 535).
_MAX_UINT16: int = 2 ** 16  # exclusive upper bound for the assertion


# ---------------------------------------------------------------------------
# ShardWriter
# ---------------------------------------------------------------------------
class ShardWriter:
    """
    Encodes validated packed conversation strings and writes NumPy shards.

    Parameters
    ----------
    output_dir : str
        Directory where .npy shard files will be written.  Created if absent.
    split : str
        Logical split name used as the filename prefix, e.g. "train" or "val".
    shard_size : int
        Maximum number of tokens per shard.  Defaults to SHARD_SIZE (2^25).
    """

    def __init__(
        self,
        output_dir: str,
        split: str = "train",
        shard_size: int = SHARD_SIZE,
    ) -> None:
        self.output_dir = output_dir
        self.split      = split
        self.shard_size = shard_size

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def write(self, validated_sequences: List[str]) -> List[str]:
        """
        Encode each sequence and write tokens into shard files.

        Conversation boundaries are respected: a conversation is never split
        across two shards.  If a single conversation exceeds ``shard_size`` it
        is written as its own (oversized) shard with a printed warning.

        Args:
            validated_sequences: Packed, validated conversation strings.
                                 These must already have passed ConversationValidator.

        Returns:
            List of absolute paths to every shard file that was written.
        """
        print(f"\n  [ShardWriter] Encoding and writing shards to '{self.output_dir}'")
        print(f"  [ShardWriter] Shard size : {self.shard_size:,} tokens (2^{self.shard_size.bit_length() - 1})")

        # ---- Pre-encode every conversation to token IDs ------------------
        # We do this upfront so that shard decisions are made on actual lengths.
        token_lists: List[np.ndarray] = []
        total_input_tokens = 0

        for seq in tqdm(validated_sequences, desc="  Encoding sequences", unit="seq"):
            ids = enc.encode(seq, allowed_special="all")
            arr = np.array(ids, dtype=np.int64)  # use int64 for the assertion

            # Verify every token ID fits in uint16
            assert (
                (arr >= 0).all() and (arr < _MAX_UINT16).all()
            ), (
                f"Token ID out of uint16 range [0, {_MAX_UINT16}).  "
                f"Vocabulary too large for this storage format."
            )

            token_lists.append(arr.astype(np.uint16))
            total_input_tokens += len(arr)

        print(f"  [ShardWriter] Total tokens to shard : {total_input_tokens:,}")

        # ---- Write shards ------------------------------------------------
        shard_paths: List[str] = []
        shard_index = 0

        # Preallocate a buffer for the current shard
        buffer = np.empty(self.shard_size, dtype=np.uint16)
        tokens_in_buffer = 0

        for conv_tokens in tqdm(token_lists, desc="  Writing shards    ", unit="conv"):
            conv_len = len(conv_tokens)

            # Case 1: conversation is larger than an entire shard.
            # Flush any buffered tokens first, then write this conversation
            # as its own oversized shard.
            if conv_len > self.shard_size:
                print("[DEBUG] THIS PRINT SHOULD NEVER BE IN TERMINAL AS conv_len > shard_size IS NEVER GOING TO BE TRUE")
                print(
                    f"\n  [ShardWriter] ⚠ Oversized conversation: {conv_len:,} tokens "
                    f"> shard_size {self.shard_size:,}.  Writing as standalone shard."
                )
                # Flush current buffer first (if non-empty)
                if tokens_in_buffer > 0:
                    path = self._flush_shard(buffer, tokens_in_buffer, shard_index)
                    shard_paths.append(path)
                    shard_index += 1
                    tokens_in_buffer = 0

                # Write oversized conversation as its own shard
                oversized_arr = conv_tokens  # already np.uint16
                path = self._write_shard(oversized_arr, shard_index)
                shard_paths.append(path)
                shard_index += 1
                continue

            # Case 2: conversation fits in the current buffer.
            if tokens_in_buffer + conv_len <= self.shard_size:
                buffer[tokens_in_buffer : tokens_in_buffer + conv_len] = conv_tokens
                tokens_in_buffer += conv_len

            # Case 3: conversation does NOT fit in the remaining buffer space.
            # Flush current buffer first, then start a fresh shard.
            else:
                path = self._flush_shard(buffer, tokens_in_buffer, shard_index)
                shard_paths.append(path)
                shard_index += 1

                # Start a new buffer with this conversation
                tokens_in_buffer = 0
                buffer[tokens_in_buffer : tokens_in_buffer + conv_len] = conv_tokens
                tokens_in_buffer += conv_len

        # ---- Write the final (possibly partial) shard --------------------
        if tokens_in_buffer > 0:
            path = self._flush_shard(buffer, tokens_in_buffer, shard_index)
            shard_paths.append(path)

        # ---- Summary -----------------------------------------------------
        total_shards      = len(shard_paths)
        total_shard_tokens = sum(np.load(p).size for p in shard_paths)

        print(f"\n  [ShardWriter] ── Summary ──────────────────────────────────────────")
        print(f"  [ShardWriter]   Shards written       : {total_shards}")
        print(f"  [ShardWriter]   Input tokens         : {total_input_tokens:,}")
        print(f"  [ShardWriter]   Tokens in shards     : {total_shard_tokens:,}")

        if total_input_tokens == total_shard_tokens:
            print(f"  [ShardWriter]   ✓ Token count integrity check PASSED.")
        else:
            diff = total_input_tokens - total_shard_tokens
            print(f"  [ShardWriter]   ✗ Token count mismatch: {diff:+,} tokens!")

        print(f"  [ShardWriter] ────────────────────────────────────────────────────\n")

        return shard_paths

    def verify_shard_integrity(self, validated_sequences: List[str]) -> bool:
        """
        Post-hoc integrity check: re-encode all sequences and compare the
        total token count against the tokens stored in every existing shard.

        This lets you confirm that the sharding stage neither lost nor
        duplicated a single token.

        Args:
            validated_sequences: The same list passed to write().

        Returns:
            True if input_token_count == shard_token_count, False otherwise.
        """
        print(f"\n  [ShardWriter] Running shard integrity verification...")

        # Count tokens in input
        input_token_count = sum(
            len(enc.encode(seq, allowed_special="all"))
            for seq in tqdm(validated_sequences, desc="  Counting input tokens", unit="seq")
        )

        # Count tokens across all shards on disk
        shard_files = sorted(
            f for f in os.listdir(self.output_dir)
            if f.startswith(self.split) and f.endswith(".npy")
        )

        if not shard_files:
            print(f"  [ShardWriter] ✗ No shard files found in '{self.output_dir}'.")
            return False

        shard_token_count = 0
        for fname in shard_files:
            path = os.path.join(self.output_dir, fname)
            arr  = np.load(path)
            shard_token_count += arr.size
            print(f"    {fname}: {arr.size:,} tokens")

        print(f"\n  [ShardWriter]   Input tokens : {input_token_count:,}")
        print(f"  [ShardWriter]   Shard tokens : {shard_token_count:,}")

        if input_token_count == shard_token_count:
            print("  [ShardWriter]   ✓ Integrity check PASSED – no tokens lost or duplicated.")
            return True
        else:
            diff = input_token_count - shard_token_count
            print(f"  [ShardWriter]   ✗ Integrity check FAILED – difference: {diff:+,} tokens.")
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _flush_shard(
        self,
        buffer: np.ndarray,
        token_count: int,
        shard_index: int,
    ) -> str:
        """Write the filled portion of ``buffer`` to disk and return the path."""
        return self._write_shard(buffer[:token_count].copy(), shard_index)

    def _write_shard(self, tokens: np.ndarray, shard_index: int) -> str:
        """Persist a uint16 token array to a .npy file and return the path."""
        filename = f"{self.split}_{shard_index:06d}.npy"
        path = os.path.join(self.output_dir, filename)
        np.save(path, tokens)
        print(f"\r  [ShardWriter] Saved {filename}  ({tokens.size:,} tokens)", end="")
        return path
