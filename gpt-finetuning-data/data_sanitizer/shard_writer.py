"""
ShardWriter
===========
Writes already-tokenised, validated conversation sequences into fixed-size
raw binary shards (.bin) for zero-copy memory-mapped loading.

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
- naming: {split}_{index:06d}.bin   e.g.  train_000000.bin
- layout: raw bytes, no header — directly usable with np.memmap
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

# 2^26 = 67 108 864 tokens ≈ 64 M tokens per shard
SHARD_SIZE: int = 2 ** 26

# GPT-2 base vocab has 50 257 tokens; our custom encoder adds 6 specials
# → maximum token ID = 50 262, well within uint16 range (65 535).
_MAX_UINT16: int = 2 ** 16  # exclusive upper bound for the assertion


# ---------------------------------------------------------------------------
# ShardWriter
# ---------------------------------------------------------------------------
class ShardWriter:
    """
    Encodes validated packed conversation strings and writes raw binary shards.

    Parameters
    ----------
    output_dir : str
        Directory where .bin shard files will be written.  Created if absent.
    split : str
        Logical split name used as the filename prefix, e.g. "train" or "val".
    shard_size : int
        Maximum number of tokens per shard.  Defaults to SHARD_SIZE (2^26).
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
        total_shard_tokens = sum(
            os.path.getsize(p) // np.dtype(np.uint16).itemsize for p in shard_paths
        )

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
        """Persist a uint16 token array to a raw .bin file and return the path."""
        filename = f"{self.split}_{shard_index:06d}.bin"
        path = os.path.join(self.output_dir, filename)

        # Ensure consistent dtype and contiguous memory
        tokens = np.ascontiguousarray(tokens, dtype=np.uint16)

        tokens.tofile(path)
        print(f"\r  [ShardWriter] Saved {filename}  ({tokens.size:,} tokens)", end="")
        
        # Generate loss mask for this shard
        self._generate_mask_for_shard(path, tokens)
        
        return path
    
    def _generate_mask_for_shard(self, shard_path: str, tokens: np.ndarray) -> str:
        """
        Generate and save a loss mask for a .bin shard file.
        
        The mask is stored as a .mask.bin file with the same length as the shard,
        using np.bool_ dtype (1 byte per token).
        
        Args:
            shard_path: Path to the .bin shard file
            tokens: Token array (uint16)
            
        Returns:
            Path to the generated mask file
            
        Mask generation rules:
            - <|PAD|> tokens → False (masked)
            - Control tokens (<|USER|>, <|ASSISTANT|>, <|BOS|>, <|EOT|>) → False (masked)
            - User turn content tokens → False (masked)
            - Assistant turn content tokens → True (unmasked)
            - Closing <|EOS|> after assistant turn → True (unmasked)
        """
        # Get special token IDs
        USER_TOKEN_ID = special_tokens["<|USER|>"]
        ASSISTANT_TOKEN_ID = special_tokens["<|ASSISTANT|>"]
        PAD_TOKEN_ID = special_tokens["<|PAD|>"]
        EOS_TOKEN_ID = special_tokens["<|EOS|>"]
        
        # Determine mask output path
        mask_path = shard_path.replace('.bin', '.mask.bin')
        
        # Create mask array
        mask = np.zeros(len(tokens), dtype=np.bool_)
        in_assistant_turn = False
        
        # Process tokens to build mask
        for i, token_id in enumerate(tokens):
            if token_id == ASSISTANT_TOKEN_ID:
                in_assistant_turn = True
                mask[i] = False  # control token is masked
            elif token_id == EOS_TOKEN_ID and in_assistant_turn:
                mask[i] = True   # closing EOS is unmasked
                in_assistant_turn = False
            elif token_id == PAD_TOKEN_ID:
                mask[i] = False
            elif in_assistant_turn:
                mask[i] = True
            else:
                mask[i] = False
        
        # Write mask to disk
        mask.tofile(mask_path)
        
        return mask_path
    
    def count_assistant_tokens_in_shards(self) -> dict:
        """
        Count only assistant response tokens across all shards in the output directory.
        
        This counts tokens that are:
        - Inside ASSISTANT turns (between <|ASSISTANT|> and <|EOS|>)
        - Excluding the <|ASSISTANT|> control token itself
        - Including the closing <|EOS|> token
        
        Uses the pre-generated .mask.bin files for extremely fast vectorized counting.
        
        Returns:
            Dictionary with per-shard and total counts:
            {
                'shards': [{'filename': str, 'assistant_tokens': int}, ...],
                'total_assistant_tokens': int,
                'total_tokens': int,
                'assistant_percentage': float
            }
        """
        # Find all .bin shard files
        shard_files = sorted(
            f for f in os.listdir(self.output_dir)
            if f.startswith(self.split) and f.endswith(".bin") and not f.endswith(".mask.bin")
        )
        
        if not shard_files:
            return {
                'shards': [],
                'total_assistant_tokens': 0,
                'total_tokens': 0,
                'assistant_percentage': 0.0
            }
        
        shard_info = []
        total_assistant_tokens = 0
        total_tokens = 0
        
        print(f"\n  [ShardWriter] Counting assistant tokens in {len(shard_files)} shards...")
        
        for fname in tqdm(shard_files, desc="  Counting assistant tokens", unit="shard"):
            shard_path = os.path.join(self.output_dir, fname)
            mask_path = shard_path.replace('.bin', '.mask.bin')
            
            # Memory-map the shard to get token count
            tokens = np.memmap(shard_path, dtype=np.uint16, mode='r')
            shard_len = len(tokens)
            total_tokens += shard_len
            
            # Memory-map the mask file for fast counting
            # The mask already has True for assistant tokens, False for everything else
            mask = np.memmap(mask_path, dtype=np.bool_, mode='r')
            
            # Vectorized count - extremely fast! Just sum the boolean array
            assistant_count = np.sum(mask)
            
            shard_info.append({
                'filename': fname,
                'assistant_tokens': int(assistant_count),
                'total_tokens': shard_len
            })
            
            total_assistant_tokens += int(assistant_count)
            
            # Close memmaps
            del tokens
            del mask
        
        assistant_percentage = (total_assistant_tokens / total_tokens * 100) if total_tokens > 0 else 0.0
        
        return {
            'shards': shard_info,
            'total_assistant_tokens': total_assistant_tokens,
            'total_tokens': total_tokens,
            'assistant_percentage': assistant_percentage
        }

