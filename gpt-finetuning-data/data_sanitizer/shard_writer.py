"""
ShardWriter
===========
Writes already-tokenised, validated conversation sequences into fixed-size
raw binary shards (.bin) for zero-copy memory-mapped loading.

Fine-tuning Design
------------------
Each input conversation is already packed and padded to exactly SEQUENCE_LENGTH
tokens (1024 by default). The shard writer respects these sequence boundaries
and stores complete sequences only.

Shard size: Configurable via sequences_per_shard (default: 16)
- Each sequence is exactly 1024 tokens (packed + padded)
- Default shard = 16 sequences × 1024 tokens = 16,384 tokens = 32 KB
- Each shard contains a whole number of sequences (no partial sequences)
- Clean sequence boundaries enable proper fine-tuning semantics

Sequence-boundary guarantee
----------------------------
Each packed conversation sequence (1024 tokens) is kept intact within a single
shard. Sequences are NEVER split across shard boundaries. If adding the next
sequence would exceed the shard size, the current shard is flushed first.

Token integrity guarantee
--------------------------
Every token appears in exactly one shard — no gaps, no duplicates.

Storage format
--------------
- dtype : np.uint16  (vocab size of custom gpt2_custom encoder ≤ 65 535)
- naming: {split}_{index:06d}.bin   e.g.  train_000000.bin
- layout: raw bytes, no header — directly usable with np.memmap
- each shard contains an exact multiple of SEQUENCE_LENGTH tokens

Usage
-----
    from data_sanitizer.shard_writer import ShardWriter

    writer = ShardWriter(
        output_dir="shards/train",
        split="train",
        sequence_length=1024,
        sequences_per_shard=16
    )
    writer.write(validated_packed_sequences)
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

# Default sequence length (each packed conversation is exactly this many tokens)
DEFAULT_SEQUENCE_LENGTH: int = 1024

# Default sequences per shard (16 sequences × 1024 tokens = 16,384 tokens = 32 KB)
DEFAULT_SEQUENCES_PER_SHARD: int = 16

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
    sequence_length : int
        Length of each packed sequence in tokens.  Defaults to 1024.
    sequences_per_shard : int
        Number of complete sequences to store per shard.  Defaults to 16.
        Shard size in tokens = sequence_length × sequences_per_shard.
    """

    def __init__(
        self,
        output_dir: str,
        split: str = "train",
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        sequences_per_shard: int = DEFAULT_SEQUENCES_PER_SHARD,
    ) -> None:
        self.output_dir = output_dir
        self.split      = split
        self.sequence_length = sequence_length
        self.sequences_per_shard = sequences_per_shard
        self.shard_size = sequence_length * sequences_per_shard  # in tokens

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def write(self, validated_sequences: List[str]) -> List[str]:
        """
        Encode each sequence and write tokens into shard files.

        Each input sequence is already packed and padded to exactly
        self.sequence_length tokens (1024 by default). Sequences are
        grouped into shards containing exactly self.sequences_per_shard
        complete sequences each. Sequence boundaries are never split.

        Args:
            validated_sequences: Packed, padded, validated conversation strings.
                                 Each must be exactly self.sequence_length tokens.

        Returns:
            List of absolute paths to every shard file that was written.
        """
        print(f"\n  [ShardWriter] Encoding and writing shards to '{self.output_dir}'")
        print(f"  [ShardWriter] Sequence length        : {self.sequence_length:,} tokens")
        print(f"  [ShardWriter] Sequences per shard    : {self.sequences_per_shard}")
        print(f"  [ShardWriter] Shard size (tokens)    : {self.shard_size:,}")

        # ---- Pre-encode every sequence to token IDs ----------------------
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

            # Verify sequence length matches expected
            seq_len = len(arr)
            if seq_len != self.sequence_length:
                raise ValueError(
                    f"Sequence length mismatch: expected {self.sequence_length}, "
                    f"got {seq_len}. All sequences must be packed to exactly "
                    f"{self.sequence_length} tokens."
                )

            token_lists.append(arr.astype(np.uint16))
            total_input_tokens += len(arr)

        print(f"  [ShardWriter] Total sequences        : {len(token_lists):,}")
        print(f"  [ShardWriter] Total tokens to shard  : {total_input_tokens:,}")

        # ---- Write shards ------------------------------------------------
        shard_paths: List[str] = []
        shard_index = 0
        
        # Buffer to accumulate sequences
        buffer_sequences = []

        for conv_tokens in tqdm(token_lists, desc="  Writing shards    ", unit="seq"):
            buffer_sequences.append(conv_tokens)

            # When buffer reaches target size, flush to disk
            if len(buffer_sequences) >= self.sequences_per_shard:
                # Concatenate all sequences in buffer
                shard_data = np.concatenate(buffer_sequences)
                path = self._write_shard(shard_data, shard_index)
                shard_paths.append(path)
                shard_index += 1
                buffer_sequences = []

        # ---- Write the final (possibly partial) shard --------------------
        if buffer_sequences:
            shard_data = np.concatenate(buffer_sequences)
            path = self._write_shard(shard_data, shard_index)
            shard_paths.append(path)

        # ---- Summary -----------------------------------------------------
        total_shards = len(shard_paths)
        total_shard_tokens = sum(
            os.path.getsize(p) // np.dtype(np.uint16).itemsize for p in shard_paths
        )

        print(f"\n  [ShardWriter] ── Summary ──────────────────────────────────────────")
        print(f"  [ShardWriter]   Shards written       : {total_shards}")
        print(f"  [ShardWriter]   Input sequences      : {len(token_lists):,}")
        print(f"  [ShardWriter]   Input tokens         : {total_input_tokens:,}")
        print(f"  [ShardWriter]   Tokens in shards     : {total_shard_tokens:,}")
        print(f"  [ShardWriter]   Sequences per shard  : {self.sequences_per_shard}")
        print(f"  [ShardWriter]   Tokens per shard     : {self.shard_size:,}")

        if total_input_tokens == total_shard_tokens:
            print(f"  [ShardWriter]   ✓ Token count integrity check PASSED.")
        else:
            diff = total_input_tokens - total_shard_tokens
            print(f"  [ShardWriter]   ✗ Token count mismatch: {diff:+,} tokens!")

        # Verify all shards contain complete sequences
        for path in shard_paths:
            tokens_in_shard = os.path.getsize(path) // np.dtype(np.uint16).itemsize
            if tokens_in_shard % self.sequence_length != 0:
                print(f"  [ShardWriter]   ⚠ Warning: {os.path.basename(path)} "
                      f"contains partial sequence ({tokens_in_shard} tokens)")

        print(f"  [ShardWriter] ────────────────────────────────────────────────────\n")

        return shard_paths

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
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

