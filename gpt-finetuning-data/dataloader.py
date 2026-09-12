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
# Shard loading
# ---------------------------------------------------------------------------

def load_shard_tokens(filename: str) -> torch.Tensor:
    """
    Load a .npy shard (uint16) and return a torch.long tensor of token IDs.
    """
    token_array = np.load(filename).astype(np.int32)
    return torch.tensor(token_array, dtype=torch.long)


# ---------------------------------------------------------------------------
# Loss-mask builder
# ---------------------------------------------------------------------------

def build_loss_mask(token_sequence: torch.Tensor) -> torch.Tensor:
    """
    Given a flat 1-D token sequence, return a boolean mask of the same length
    where True means "this position should contribute to the loss".

    Rules:
        - <|PAD|> tokens are always masked (False).
        - Control tokens (<|USER|>, <|ASSISTANT|>, <|BOS|>, <|EOT|>) are masked.
        - Content tokens inside a USER turn are masked (False).
        - Content tokens inside an ASSISTANT turn, including the closing <|EOS|>,
          are unmasked (True).

    The sequence is scanned left-to-right. We start in 'masked' mode and
    flip to 'unmasked' after each <|ASSISTANT|> token, then back to 'masked'
    after each <|EOS|> that closes an ASSISTANT turn.
    """
    mask = torch.zeros(len(token_sequence), dtype=torch.bool)
    in_assistant_turn = False

    for i, token_id in enumerate(token_sequence.tolist()):
        if token_id == ASSISTANT_TOKEN_ID:
            in_assistant_turn = True
            mask[i] = False          # the control token itself is not a target
        elif token_id == EOS_TOKEN_ID and in_assistant_turn:
            mask[i] = True           # include the closing <|EOS|> in the loss
            in_assistant_turn = False
        elif token_id == PAD_TOKEN_ID:
            mask[i] = False
        elif in_assistant_turn:
            mask[i] = True

    return mask


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

        shard_files = sorted(
            f for f in os.listdir(shard_dir)
            if f.startswith(split) and f.endswith('.npy')
        )
        if not shard_files:
            raise ValueError(f"No shard files found in {shard_dir} for split '{split}'")

        self.shard_files = [os.path.join(shard_dir, f) for f in shard_files]
        self.reset()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self):
        """Reset to the beginning of the first shard."""
        self.current_shard_idx = 0
        self._load_shard(self.current_shard_idx)

        tokens_per_rank = self.batch_size * self.sequence_length
        self.current_position = tokens_per_rank * self.process_rank
        self.current_step = 0

    def _load_shard(self, idx: int):
        """Load the shard at index idx and pre-build its loss mask."""
        shard_path = self.shard_files[idx]
        self.tokens = load_shard_tokens(shard_path)

        # Pre-build the loss mask for the whole shard in one pass
        self.loss_mask = build_loss_mask(self.tokens)

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

        token_seq = self.tokens[start:end]
        mask_seq  = self.loss_mask[start:end]

        # input  = first B*T tokens
        # target = next  B*T tokens (shifted right by 1)
        inputs      = token_seq[:-1].view(B, T)
        targets     = token_seq[1:].view(B, T).clone()
        target_mask = mask_seq[1:].view(B, T)   # mask also shifted to align with targets

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
        """Return basic statistics about the dataset (loads all shards)."""
        total_tokens = 0
        shard_info = []
        for shard_path in self.shard_files:
            tokens = load_shard_tokens(shard_path)
            total_tokens += len(tokens)
            shard_info.append({
                'filename': os.path.basename(shard_path),
                'tokens': len(tokens),
            })
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

    except Exception as e:
        print(f"Test failed: {e}")
        print("Run the data preparation pipeline first to create shards.")
