"""
Simple DataLoader for GPT Fine-tuning Training Data

This dataloader loads tokenized conversation data from NumPy shard files
created by the data preparation pipeline. It's designed to be simple,
readable, and easy to understand.

The dataloader works with shards created by ShardWriter, which contain
tokenized conversation data stored as uint16 numpy arrays.
"""

import os
import torch
import numpy as np
from typing import List, Optional
from data_sanitizer.utils import enc


def load_shard_tokens(filename: str) -> torch.Tensor:
    """
    Load tokens from a numpy shard file and convert to PyTorch tensor.
    
    Args:
        filename: Path to the .npy shard file
        
    Returns:
        PyTorch tensor containing token IDs as long integers
    """
    # Load numpy array (uint16) and convert to int32 then to torch.long
    token_array = np.load(filename)
    token_array = token_array.astype(np.int32)
    return torch.tensor(token_array, dtype=torch.long)




class ConversationDataLoader:
    def __init__(
        self,
        batch_size,
        sequence_length,
        process_rank=0,
        num_processes=1,
        split='train',
        data_root='shards'
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

        all_files = os.listdir(shard_dir)
        shard_files = [
            f for f in all_files
            if f.startswith(split) and f.endswith('.npy')
        ]
        shard_files = sorted(shard_files)

        if not shard_files:
            raise ValueError(
                f"No shard files found in {shard_dir} for split '{split}'"
            )

        self.shard_files = [
            os.path.join(shard_dir, f)
            for f in shard_files
        ]

        self.reset()

    def reset(self):
        self.current_shard_idx = 0
        self.tokens = load_shard_tokens(
            self.shard_files[self.current_shard_idx]
        )

        tokens_per_batch = self.batch_size * self.sequence_length

        # CHANGED:
        # Number of tokens consumed by ALL processes for one training step.
        self.distributed_step_tokens = (
            tokens_per_batch * self.num_processes
        )

        # CHANGED:
        # Calculate how many complete distributed steps fit in this shard.
        # -1 is required because each batch needs T+1 tokens.
        usable_tokens = len(self.tokens) - 1

        self.steps_in_shard = (
            usable_tokens // self.distributed_step_tokens
        )

        # CHANGED:
        # Each rank starts at its own section of the first distributed batch.
        self.current_step = 0
        self.current_position = (
            tokens_per_batch * self.process_rank
        )

        print(
            f"Reset to shard 0, "
            f"position {self.current_position}, "
            f"steps_in_shard {self.steps_in_shard}"
        )

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = self.batch_size, self.sequence_length

        # CHANGED:
        # Instead of checking independently whether the current position
        # is beyond the shard, use the same step count for every rank.
        if self.current_step >= self.steps_in_shard:
            self._advance_to_next_shard()

        # T + 1 tokens are required to create input and target.
        tokens_needed = B * T + 1

        start_pos = self.current_position
        end_pos = start_pos + tokens_needed

        token_sequence = self.tokens[start_pos:end_pos]

        input_tokens = token_sequence[:-1]
        target_tokens = token_sequence[1:]

        inputs = input_tokens.view(B, T)
        targets = target_tokens.view(B, T)

        # CHANGED:
        # Move to the next distributed batch.
        # Every rank advances by the total amount consumed by all ranks.
        self.current_position += (
            B * T * self.num_processes
        )

        # CHANGED:
        self.current_step += 1

        return inputs, targets

    def _advance_to_next_shard(self):
        self.current_shard_idx = (
            self.current_shard_idx + 1
        ) % len(self.shard_files)

        shard_path = self.shard_files[self.current_shard_idx]
        self.tokens = load_shard_tokens(shard_path)

        # CHANGED:
        # Recalculate how many complete distributed steps
        # are available in the new shard.
        usable_tokens = len(self.tokens) - 1

        self.steps_in_shard = (
            usable_tokens // self.distributed_step_tokens
        )

        # CHANGED:
        # Reset the step counter for the new shard.
        self.current_step = 0

        # CHANGED:
        # Every rank starts at its own section of the first
        # distributed batch in the new shard.
        tokens_per_batch = self.batch_size * self.sequence_length

        self.current_position = (
            tokens_per_batch * self.process_rank
        )

        print(
            f"Advanced to shard {self.current_shard_idx}, "
            f"position {self.current_position}, "
            f"steps_in_shard {self.steps_in_shard}"
        )



    def get_stats(self) -> dict:
        """
        Get statistics about the loaded data.
        
        Returns:
            Dictionary containing data statistics
        """
        total_tokens = 0
        shard_info = []
        
        for shard_path in self.shard_files:
            tokens = load_shard_tokens(shard_path)
            num_tokens = len(tokens)
            total_tokens += num_tokens
            
            shard_info.append({
                'filename': os.path.basename(shard_path),
                'tokens': num_tokens
            })
        
        return {
            'split': self.split,
            'num_shards': len(self.shard_files),
            'total_tokens': total_tokens,
            'shard_info': shard_info,
            'avg_tokens_per_shard': total_tokens / len(self.shard_files) if self.shard_files else 0
        }



class ConversationDataLoader_old:
    """
    Simple dataloader for conversation training data stored in shards.
    
    This dataloader:
    - Loads tokenized data from .npy shard files
    - Provides batches for training with configurable batch size and sequence length
    - Handles multiple processes for distributed training
    - Cycles through all available shards
    - Returns input and target sequences (shifted by one token)
    
    Args:
        batch_size (B): Number of sequences per batch
        sequence_length (T): Length of each sequence (context window)
        process_rank: Current process rank (for distributed training)
        num_processes: Total number of processes (for distributed training)
        split: Data split to use ('train' or 'val')
        data_root: Root directory containing shard folders (default: 'shards')
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
        
        # Validate split
        assert split in {'train', 'val'}, f"Split must be 'train' or 'val', got '{split}'"
        self.split = split
        
        # Build path to shard directory
        shard_dir = os.path.join(data_root, split)
        
        # Find all shard files for this split
        if not os.path.exists(shard_dir):
            raise ValueError(f"Shard directory does not exist: {shard_dir}")
            
        all_files = os.listdir(shard_dir)
        shard_files = [f for f in all_files if f.startswith(split) and f.endswith('.npy')]
        shard_files = sorted(shard_files)  # Ensure consistent ordering
        
        if not shard_files:
            raise ValueError(f"No shard files found in {shard_dir} for split '{split}'")
            
        # Convert to full paths
        self.shard_files = [os.path.join(shard_dir, f) for f in shard_files]
        
        print(f"DataLoader initialized:")
        print(f"  Split: {split}")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Found {len(self.shard_files)} shard files")
        print(f"  Process {process_rank}/{num_processes}")
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """
        Reset the dataloader to the beginning.
        
        This loads the first shard and sets the position for this process.
        Each process starts at a different position to avoid data overlap.
        """
        # Start with the first shard
        self.current_shard_idx = 0
        self.tokens = load_shard_tokens(self.shard_files[self.current_shard_idx])
        
        # Each process starts at its own position to avoid overlap
        # This ensures each process gets different data
        tokens_per_batch = self.batch_size * self.sequence_length
        self.current_position = tokens_per_batch * self.process_rank
        
        print(f"Reset to shard 0, position {self.current_position}")
    
    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the next batch of training data.
        
        Returns:
            tuple: (input_sequences, target_sequences)
                - input_sequences: [batch_size, sequence_length] tensor
                - target_sequences: [batch_size, sequence_length] tensor (shifted by 1)
        """
        B, T = self.batch_size, self.sequence_length
        
        # We need T+1 consecutive tokens to create input and target sequences
        tokens_needed = B * T + 1
        
        # Check if we have enough tokens in the current shard
        if self.current_position + tokens_needed > len(self.tokens):
            # Need to move to the next shard
            self._advance_to_next_shard()
        
        # Extract the token sequence for this batch
        start_pos = self.current_position
        end_pos = start_pos + tokens_needed
        token_sequence = self.tokens[start_pos:end_pos]
        
        # Split into inputs (first T tokens) and targets (last T tokens, shifted by 1)
        input_tokens = token_sequence[:-1]  # All but last token
        target_tokens = token_sequence[1:]  # All but first token
        
        # Reshape into batch format [batch_size, sequence_length]
        inputs = input_tokens.view(B, T)
        targets = target_tokens.view(B, T)
        
        # Advance position for next batch
        # Skip ahead by num_processes to avoid overlap between processes
        self.current_position += B * T * self.num_processes
        
        return inputs, targets
    
    def _advance_to_next_shard(self):
        """
        Move to the next shard file and reset position.
        
        If we're at the last shard, cycle back to the first shard.
        This allows for infinite training iterations.
        """
        # Move to next shard (with wraparound)
        self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shard_files)
        
        # Load the new shard
        shard_path = self.shard_files[self.current_shard_idx]
        self.tokens = load_shard_tokens(shard_path)
        
        # Reset position for this process
        tokens_per_batch = self.batch_size * self.sequence_length
        self.current_position = tokens_per_batch * self.process_rank
        
        shard_filename = os.path.basename(shard_path)
        print(f"Advanced to shard {self.current_shard_idx}: {shard_filename}")
        print(f"  Shard size: {len(self.tokens):,} tokens")
        print(f"  Reset position to: {self.current_position}")
    
    def get_stats(self) -> dict:
        """
        Get statistics about the loaded data.
        
        Returns:
            Dictionary containing data statistics
        """
        total_tokens = 0
        shard_info = []
        
        for shard_path in self.shard_files:
            tokens = load_shard_tokens(shard_path)
            num_tokens = len(tokens)
            total_tokens += num_tokens
            
            shard_info.append({
                'filename': os.path.basename(shard_path),
                'tokens': num_tokens
            })
        
        return {
            'split': self.split,
            'num_shards': len(self.shard_files),
            'total_tokens': total_tokens,
            'shard_info': shard_info,
            'avg_tokens_per_shard': total_tokens / len(self.shard_files) if self.shard_files else 0
        }


# Example usage and testing
if __name__ == "__main__":
    print("ConversationDataLoader Test")
    print("=" * 50)
    
    try:
        # Try to create a dataloader
        # Note: This will only work if shards have been created
        dataloader = ConversationDataLoader(
            batch_size=2,
            sequence_length=128,
            process_rank=0,
            num_processes=1,
            split='train'
        )
        
        # Print statistics
        stats = dataloader.get_stats()
        print(f"\nDataLoader Statistics:")
        print(f"  Split: {stats['split']}")
        print(f"  Number of shards: {stats['num_shards']}")
        print(f"  Total tokens: {stats['total_tokens']:,}")
        print(f"  Average tokens per shard: {stats['avg_tokens_per_shard']:.0f}")
        
        # Get a sample batch
        print(f"\nTesting batch generation...")
        inputs, targets = dataloader.next_batch()
        print(f"  Input batch shape: {inputs.shape}")
        print(f"  Target batch shape: {targets.shape}")
        print(f"  Sample input tokens (first 10): {inputs[0][:10].tolist()}")
        print(f"  Sample target tokens (first 10): {targets[0][:10].tolist()}")
        
        # Test decoding a few tokens to see if they make sense
        sample_tokens = inputs[0][:20].tolist()
        try:
            decoded_text = enc.decode(sample_tokens)
            print(f"  Decoded sample: '{decoded_text[:100]}...'")
        except Exception as e:
            print(f"  Could not decode sample: {e}")
            
    except Exception as e:
        print(f"Cannot test dataloader: {e}")
        print("To test the dataloader, first run the data preparation pipeline to create shards.")