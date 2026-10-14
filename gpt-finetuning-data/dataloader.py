"""
DataLoader for GPT Chat Fine-tuning

Loads tokenised conversation shards (created by ShardWriter) and produces
(input, target) batches with a loss mask already applied:

    - <|USER|> turns   → target = -100  (masked, not learned)
    - <|PAD|> tokens   → target = -100  (masked, not learned)
    - <|ASSISTANT|> turns → target = real token  (learned)

cross_entropy with ignore_index=-100 will automatically skip masked positions,
so the model only trains to predict ASSISTANT responses.

Token format inside each shard sequence:
    <|BOS|> <|USER|> ... <|EOS|> <|ASSISTANT|> ... <|EOS|> ... <|EOT|> <|PAD|>...
    
Loss masks are stored as .mask.bin files alongside token shards and are
memory-mapped for efficient loading.
"""

import os
import torch
import numpy as np
from data_sanitizer.utils import enc, special_tokens


# ---------------------------------------------------------------------------
# Special token IDs (resolved once at import time)
# ---------------------------------------------------------------------------
USER_TOKEN_ID      = special_tokens["<|USER|>"]
ASSISTANT_TOKEN_ID = special_tokens["<|ASSISTANT|>"]
PAD_TOKEN_ID       = special_tokens["<|PAD|>"]
EOS_TOKEN_ID       = special_tokens["<|EOS|>"]
EOT_TOKEN_ID       = special_tokens["<|EOT|>"]

IGNORE_INDEX = -100   # value used by cross_entropy to skip a position


# ---------------------------------------------------------------------------
# Main DataLoader
# ---------------------------------------------------------------------------

class ConversationDataLoader:
    """
    Shard-based dataloader for chat fine-tuning.

    Each call to next_batch() returns:
        inputs  : (B, T) long tensor — token IDs fed into the model
        targets : (B, T) long tensor — next-token targets with -100 for
                  positions the model should NOT learn (USER turns + PAD)

    Args:
        batch_size      : Number of sequences per batch (B)
        sequence_length : Context window length (T)
        process_rank    : This process's DDP rank (0 for single-GPU)
        num_processes   : Total number of DDP processes (1 for single-GPU)
        split           : 'train' or 'val'
        data_root       : Root directory that contains train/ and val/ shard folders
    """

    def __init__(
        self,
        batch_size: int,
        sequence_length: int,
        process_rank: int = 0,
        num_processes: int = 1,
        split: str = 'train',
        data_root: str = 'shards'
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.process_rank = process_rank
        self.num_processes = num_processes

        assert split in {'train', 'val'}, \
            f"Split must be 'train' or 'val', got '{split}'"
        self.split = split

        shard_dir = os.path.join(data_root, split)
        if not os.path.exists(shard_dir):
            raise ValueError(f"Shard directory does not exist: {shard_dir}")

        # Look for .bin files (not .mask.bin)
        shard_files = sorted(
            f for f in os.listdir(shard_dir)
            if f.startswith(split) and f.endswith('.bin') and not f.endswith('.mask.bin')
        )
        if not shard_files:
            raise ValueError(f"No .bin shard files found in {shard_dir} for split '{split}'")

        self.shard_files = [os.path.join(shard_dir, f) for f in shard_files]

        # Active memory-map references
        self.tokens = None
        self.loss_mask = None
        
        # Verify that all mask files exist and lengths match shards
        self._verify_mask_files()
        
        self.reset()

    # ------------------------------------------------------------------
    # State management & resource cleanup
    # ------------------------------------------------------------------

    @staticmethod
    def _close_memmap(arr):
        """Explicitly close the underlying mmap buffer to release OS file descriptors."""
        if arr is not None and hasattr(arr, '_mmap') and arr._mmap is not None:
            try:
                arr._mmap.close()
            except Exception:
                pass

    def _close_current_shard(self):
        """Explicitly close the current shard and mask memmap buffers."""
        if hasattr(self, 'tokens') and self.tokens is not None:
            self._close_memmap(self.tokens)
            self.tokens = None
        if hasattr(self, 'loss_mask') and self.loss_mask is not None:
            self._close_memmap(self.loss_mask)
            self.loss_mask = None

    def close(self):
        """Explicitly release all memory-mapped file handles."""
        self._close_current_shard()

    def __del__(self):
        self.close()
    
    def _verify_mask_files(self):
        """Verify that all mask files exist and their token counts match the shard files."""
        for shard_path in self.shard_files:
            mask_path = shard_path.replace('.bin', '.mask.bin')
            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"Mask file not found: {mask_path}\n"
                    f"Run the data preparation pipeline to generate masks."
                )

            shard_bytes = os.path.getsize(shard_path)
            mask_bytes = os.path.getsize(mask_path)

            if shard_bytes % 2 != 0:
                raise ValueError(
                    f"Corrupted shard file (odd byte count {shard_bytes}): {os.path.basename(shard_path)}"
                )

            token_count = shard_bytes // 2
            mask_count = mask_bytes

            if token_count != mask_count:
                raise ValueError(
                    f"Shard and mask length mismatch in {os.path.basename(shard_path)}: "
                    f"{token_count:,} tokens vs {mask_count:,} mask values. "
                    f"Regenerate masks or re-run the pipeline."
                )

    def reset(self):
        """Reset to the beginning of the first shard."""
        self.current_shard_idx = 0
        self._load_shard(self.current_shard_idx)

        tokens_per_rank = self.batch_size * self.sequence_length
        self.current_position = tokens_per_rank * self.process_rank
        self.current_step = 0

    def _load_shard(self, idx: int):
        """Load the shard at index idx using memory mapping."""
        # Release the previous shard's OS file descriptors before opening new ones
        self._close_current_shard()

        shard_path = self.shard_files[idx]
        mask_path = shard_path.replace('.bin', '.mask.bin')
        
        # Memory-map the token shard (do not load into RAM)
        self.tokens = np.memmap(shard_path, dtype=np.uint16, mode='r')
        
        # Memory-map the loss mask
        self.loss_mask = np.memmap(mask_path, dtype=np.bool_, mode='r')
        
        # Defensive check
        if len(self.tokens) != len(self.loss_mask):
            raise ValueError(
                f"Shard and mask length mismatch in {os.path.basename(shard_path)}: "
                f"{len(self.tokens)} tokens vs {len(self.loss_mask)} mask values"
            )

        tokens_per_rank = self.batch_size * self.sequence_length
        self.distributed_step_tokens = tokens_per_rank * self.num_processes

        # How many complete distributed steps fit in this shard
        # (-1 because we need T+1 tokens to produce an (input, target) pair)
        usable_tokens = len(self.tokens) - 1
        self.steps_in_shard = usable_tokens // self.distributed_step_tokens

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the next (inputs, targets) batch.

        targets already has IGNORE_INDEX (-100) at every position the model
        should not learn: USER-turn tokens and PAD tokens.
        """
        B, T = self.batch_size, self.sequence_length

        if self.current_step >= self.steps_in_shard:
            self._advance_to_next_shard()

        start = self.current_position
        end   = start + B * T + 1

        # Extract slices from memory-mapped arrays (minimal memory footprint)
        token_seq = self.tokens[start:end]
        mask_seq  = self.loss_mask[start:end]
        
        # Convert to torch tensors only for the batch
        token_tensor = torch.from_numpy(token_seq.astype(np.int32))

        mask_tensor = torch.from_numpy(mask_seq.copy())

        # input  = first B*T tokens
        # target = next  B*T tokens (shifted right by 1)
        inputs      = token_tensor[:-1].view(B, T)
        targets     = token_tensor[1:].view(B, T).clone()
        target_mask = mask_tensor[1:].view(B, T)   # mask also shifted to align with targets

        # Zero-out positions the model should not learn
        targets[~target_mask] = IGNORE_INDEX

        self.current_position += B * T * self.num_processes
        self.current_step     += 1

        return inputs, targets

    def _advance_to_next_shard(self):
        """Move to the next shard, wrapping around after the last one."""
        self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shard_files)
        self._load_shard(self.current_shard_idx)

        tokens_per_rank = self.batch_size * self.sequence_length
        self.current_position = tokens_per_rank * self.process_rank
        self.current_step = 0

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return basic statistics about the dataset (uses memory mapping, minimal RAM)."""
        total_tokens = 0
        shard_info = []
        for shard_path in self.shard_files:
            # Memory-map instead of loading
            tokens = np.memmap(shard_path, dtype=np.uint16, mode='r')
            shard_tokens = len(tokens)
            total_tokens += shard_tokens
            shard_info.append({
                'filename': os.path.basename(shard_path),
                'tokens': shard_tokens,
            })
            self._close_memmap(tokens)
            del tokens
        return {
            'split': self.split,
            'num_shards': len(self.shard_files),
            'total_tokens': total_tokens,
            'avg_tokens_per_shard': total_tokens / len(self.shard_files),
            'shard_info': shard_info,
        }


# ---------------------------------------------------------------------------
# Quick smoke-test  (python dataloader.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("ConversationDataLoader smoke test")
    print("=" * 50)

    loader = None
    try:
        loader = ConversationDataLoader(
            batch_size=2,
            sequence_length=128,
            process_rank=0,
            num_processes=1,
            split='train',
        )

        stats = loader.get_stats()
        print(f"Split        : {stats['split']}")
        print(f"Shards       : {stats['num_shards']}")
        print(f"Total tokens : {stats['total_tokens']:,}")

        inputs, targets = loader.next_batch()
        print(f"\nInput shape  : {inputs.shape}")
        print(f"Target shape : {targets.shape}")

        n_masked   = (targets == IGNORE_INDEX).sum().item()
        n_unmasked = (targets != IGNORE_INDEX).sum().item()
        print(f"Masked positions  (USER/PAD) : {n_masked}")
        print(f"Unmasked positions (ASST)    : {n_unmasked}")

        sample = inputs[0, :20].tolist()
        print(f"\nFirst 20 input tokens : {sample}")
        print(f"Decoded               : {enc.decode(sample, errors='replace')!r}")
        
        print("\n✓ Memory-mapped .bin shards and .mask.bin files loaded successfully")

    except Exception as e:
        print(f"Test failed: {e}")
        print("Run 'python run_pipeline.py' first to create .bin shards and masks.")
    finally:
        if loader is not None:
            loader.close()
