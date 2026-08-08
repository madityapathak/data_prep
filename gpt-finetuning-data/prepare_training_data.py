"""
Main script to prepare fine-tuning data for LLM conversation training.

Pipeline:
1. Load data from all datasets
2. Sanitize data through 3-step process:
   - DatasetCleaner: Remove HTML, unwanted tags, long sequences
   - ConversationFilter: Filter by context length, separate multi-turn
   - ConversationWindowProcessor: Apply sliding window to long conversations
3. Mix datasets according to target token counts
4. Pack sequences (TokenPacker):
   - Wrap every entry with <|BOS|> and <|EOT|> text tokens
   - Sort ascending by num_tokens, then merge consecutive entries that fit
     within max_context (maximises GPU utilisation)
   - Right-pad every sequence with <|PAD|> to exactly max_context tokens
5. Validate packed sequences (ConversationValidator):
   - Confirm BOS/EOT boundaries, USER-first, ASSISTANT-last, strict alternation
   - Raises ValueError and halts if any invalid sequences are found
6. Write NumPy shards (ShardWriter):
   - Encode validated strings to uint16 token arrays
   - Respect conversation boundaries — never split a conversation across shards
   - Write fixed-size shards (2^25 ≈ 32 M tokens) for efficient DataLoader use
"""

import random
import re
from typing import List, Tuple

# Import dataset processors
from get_formatted_datasets.daily_dialog import DailyDialogProcessor
from get_formatted_datasets.databricks_dolly import DollyProcessor
from get_formatted_datasets.oasst import OASSTProcessor
from get_formatted_datasets.open_orca import OpenOrcaProcessor
from get_formatted_datasets.ultrachat import UltraChatProcessor
from get_formatted_datasets.utils import TARGETS, SEED

# Import sanitizers
from data_sanitizer.dataset_cleaner import DatasetCleaner
from data_sanitizer.conversation_filter import ConversationFilter
from data_sanitizer.conversation_window_processor import ConversationWindowProcessor
from data_sanitizer.token_packer import TokenPacker
from data_sanitizer.conversation_validator import ConversationValidator
from data_sanitizer.shard_writer import ShardWriter, SHARD_SIZE
from data_sanitizer.utils import num_tokens


class DataPreparationPipeline:
    """Main pipeline for preparing fine-tuning data."""
    
    def __init__(self, seed: int = SEED):
        """
        Initialize the data preparation pipeline.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        
        # Initialize sanitizers
        self.dataset_cleaner = DatasetCleaner()
        self.conversation_filter = ConversationFilter()
        self.conversation_window_processor = ConversationWindowProcessor(max_length=1010)

        # Max context length (must match ConversationFilter.MAX_CONTEXT + BOS/EOT overhead)
        self.max_context = 1024

        # TokenPacker: adds BOS/EOT, merges small entries, pads to max_context
        self.token_packer = TokenPacker(max_context=self.max_context)

        # ConversationValidator: runs immediately after BOS/EOT are added;
        # strict=True means the pipeline halts on any invalid sequence.
        self.conversation_validator = ConversationValidator(strict=True)

        # Dataset processors
        self.processors = {
            "dailydialog": DailyDialogProcessor(),
            "dolly": DollyProcessor(),
            "oasst": OASSTProcessor(),
            "openorca": OpenOrcaProcessor(),
            "ultrachat": UltraChatProcessor(),
        }
        
    def load_raw_data(self) -> dict:
        """
        Load raw data from all datasets.
        
        Returns:
            Dictionary with dataset names as keys and (train, validation) tuples as values
        """
        print("=" * 60)
        print("STEP 1: Loading raw data from all datasets")
        print("=" * 60)
        
        raw_data = {}
        
        for dataset_name, processor in self.processors.items():
            print(f"\nLoading {dataset_name}...")
            try:
                train_data = processor.get_train_data()
                validation_data = processor.get_validation_data()
                raw_data[dataset_name] = (train_data, validation_data)
                print(f"  ✓ Train: {len(train_data)} conversations")
                print(f"  ✓ Validation: {len(validation_data)} conversations")
            except Exception as e:
                print(f"  ✗ Error loading {dataset_name}: {e}")
                raw_data[dataset_name] = ([], [])
        
        return raw_data
    
    def sanitize_data(self, conversations: List[str]) -> List[str]:
        """
        Apply 3-step sanitization pipeline to conversations.
        
        Args:
            conversations: List of conversation strings
            
        Returns:
            List of sanitized conversation strings
        """
        if not conversations:
            return []
        
        # Step 1: DatasetCleaner - Remove HTML, unwanted tags, long sequences
        cleaned_conversations = self.dataset_cleaner.clean(conversations, min_length=41)
        
        # Step 2: ConversationFilter - Filter by context length
        filtered_conversations, long_conversations = self.conversation_filter.filter_conversations(
            cleaned_conversations
        )
        
        # Step 3: ConversationWindowProcessor - Apply sliding window to long conversations
        windowed_conversations = self.conversation_window_processor.process(long_conversations)
        
        # Combine filtered and windowed conversations
        all_sanitized = filtered_conversations + windowed_conversations
        
        return all_sanitized
    
    def sanitize_all_datasets(self, raw_data: dict) -> dict:
        """
        Sanitize data from all datasets.
        
        Args:
            raw_data: Dictionary with dataset names and (train, validation) tuples
            
        Returns:
            Dictionary with sanitized data
        """
        print("\n" + "=" * 60)
        print("STEP 2: Sanitizing data from all datasets")
        print("=" * 60)
        
        sanitized_data = {}
        
        for dataset_name, (train_data, validation_data) in raw_data.items():
            print(f"\nSanitizing {dataset_name}...")
            
            # Sanitize train data
            print(f"  Processing train data ({len(train_data)} conversations)...")
            sanitized_train = self.sanitize_data(train_data)
            
            # Sanitize validation data
            print(f"  Processing validation data ({len(validation_data)} conversations)...")
            sanitized_validation = self.sanitize_data(validation_data)
            
            sanitized_data[dataset_name] = (sanitized_train, sanitized_validation)
            
            print(f"  ✓ Train: {len(sanitized_train)} conversations")
            print(f"  ✓ Validation: {len(sanitized_validation)} conversations")
        
        return sanitized_data
    
    def calculate_total_tokens(self, conversations: List[str]) -> int:
        """
        Calculate total number of tokens in a list of conversations.
        
        Args:
            conversations: List of conversation strings
            
        Returns:
            Total token count
        """
        return sum(num_tokens(conv) for conv in conversations)
    
    def sample_by_token_target(
        self,
        conversations: List[str],
        target_tokens: int
    ) -> Tuple[List[str], int]:
        """
        Sample conversations to meet target token count.
        
        Args:
            conversations: List of conversation strings
            target_tokens: Target number of tokens
            
        Returns:
            Tuple of (sampled conversations, actual token count)
        """
        if not conversations:
            return [], 0
        
        # Shuffle conversations for randomness
        shuffled_conversations = conversations.copy()
        random.shuffle(shuffled_conversations)
        
        sampled = []
        current_tokens = 0
        
        for conv in shuffled_conversations:
            conv_tokens = num_tokens(conv)
            
            # Stop if we've reached the target
            if current_tokens >= target_tokens:
                break
            
            sampled.append(conv)
            current_tokens += conv_tokens
        
        return sampled, current_tokens
    
    def mix_datasets(self, sanitized_data: dict) -> Tuple[List[str], List[str]]:
        """
        Mix datasets according to target token counts.
        
        Args:
            sanitized_data: Dictionary with sanitized data
            
        Returns:
            Tuple of (final_train_data, final_validation_data)
        """
        print("\n" + "=" * 60)
        print("STEP 3: Mixing datasets according to token targets")
        print("=" * 60)
        
        final_train_data = []
        final_validation_data = []
        
        for dataset_name, target_tokens in TARGETS.items():
            print(f"\n{dataset_name}:")
            print(f"  Target tokens: {target_tokens:,}")
            
            if dataset_name not in sanitized_data:
                print(f"  ✗ Dataset not found, skipping")
                continue
            
            train_data, validation_data = sanitized_data[dataset_name]
            
            # Calculate proportional split (100% train, 10% validation)
            train_target = int(target_tokens * 1)
            validation_target = int(target_tokens * 0.1)
            
            # Sample train data
            sampled_train, actual_train_tokens = self.sample_by_token_target(
                train_data, train_target
            )
            
            # Sample validation data
            sampled_validation, actual_validation_tokens = self.sample_by_token_target(
                validation_data, validation_target
            )
            
            # Add to final datasets
            final_train_data.extend(sampled_train)
            final_validation_data.extend(sampled_validation)
            
            print(f"  Train: {len(sampled_train)} conversations, {actual_train_tokens:,} tokens")
            print(f"  Validation: {len(sampled_validation)} conversations, {actual_validation_tokens:,} tokens")
        
        # Shuffle final datasets
        random.shuffle(final_train_data)
        random.shuffle(final_validation_data)
        
        return final_train_data, final_validation_data

    def pack_data(
        self,
        train_data: List[str],
        validation_data: List[str],
    ) -> "Tuple[List[str], List[str]]":
        """
        Apply TokenPacker post-processing to both splits.

        For each split this:
          1. Wraps every entry with <|BOS|> and <|EOT|> text tokens.
          2. Sorts by num_tokens (ascending) then merges consecutive entries
             so the combined token count stays within max_context.
          3. Right-pads every sequence with <|PAD|> text tokens to exactly
             max_context tokens.

        No encoding is performed — strings are returned as-is so they can
        be saved to disk and encoded later.

        Args:
            train_data:      List of sanitised training conversation strings.
            validation_data: List of sanitised validation conversation strings.

        Returns:
            Tuple of (packed_train, packed_validation) where each element is
            a list of packed, padded conversation strings.
        """
        print("\n" + "=" * 60)
        print("STEP 4: Packing sequences (BOS/EOT → sort → merge → pad)")
        print("=" * 60)

        print("\nPacking train data...")
        packed_train = self.token_packer.process(train_data)

        print("\nPacking validation data...")
        packed_validation = self.token_packer.process(validation_data)

        return packed_train, packed_validation

    def validate_and_shard(
        self,
        packed_train: "List[str]",
        packed_validation: "List[str]",
        train_shard_dir: str = "shards/train",
        val_shard_dir: str = "shards/val",
    ) -> "Tuple[List[str], List[str]]":
        """
        Validate packed sequences then write NumPy shards for each split.

        This method must be called **after** pack_data() because it expects
        sequences that already carry <|BOS|> and <|EOT|> boundaries.

        Steps (per split):
          1. ConversationValidator checks all 5 structural rules on the
             tokenized representation.  Raises ValueError on failure.
          2. ShardWriter encodes each valid sequence to uint16 token IDs
             and writes fixed-size (2^25 token) .npy shard files while
             keeping every conversation intact within a single shard.

        Args:
            packed_train:      List of packed training conversation strings.
            packed_validation: List of packed validation conversation strings.
            train_shard_dir:   Output directory for training shards.
            val_shard_dir:     Output directory for validation shards.

        Returns:
            Tuple of (valid_packed_train, valid_packed_validation) — the
            subsets that passed validation (order preserved).
        """
        print("\n" + "=" * 60)
        print("STEP 5: Validating packed sequences")
        print("=" * 60)

        print("\nValidating train data...")
        valid_train = self.conversation_validator.validate(packed_train)

        print("\nValidating validation data...")
        valid_validation = self.conversation_validator.validate(packed_validation)

        print("\n" + "=" * 60)
        print("STEP 6: Writing NumPy shards")
        print(f"        Shard size : {SHARD_SIZE:,} tokens (2^25 ≈ 32 M)")
        print("=" * 60)

        print("\nWriting train shards...")
        train_writer = ShardWriter(output_dir=train_shard_dir, split="train")
        train_writer.write(valid_train)

        print("\nWriting validation shards...")
        val_writer = ShardWriter(output_dir=val_shard_dir, split="val")
        val_writer.write(valid_validation)

        return valid_train, valid_validation

    def run(self) -> "Tuple[List[str], List[str]]":
        """
        Run the complete data preparation pipeline.

        Returns:
            Tuple of (packed_train_data, packed_validation_data) where each
            element is a list of packed, padded conversation strings (with
            <|BOS|>, <|EOT|>, and <|PAD|> special tokens in text form).
            No encoding is performed.
        """
        print("\n" + "=" * 60)
        print("STARTING DATA PREPARATION PIPELINE")
        print("=" * 60)
        
        # Step 1: Load raw data
        raw_data = self.load_raw_data()
        
        # Step 2: Sanitize data
        sanitized_data = self.sanitize_all_datasets(raw_data)
        
        # Step 3: Mix datasets according to targets
        final_train_data, final_validation_data = self.mix_datasets(sanitized_data)

        # Intermediate statistics (pre-packing)
        print("\n" + "=" * 60)
        print("PRE-PACK STATISTICS")
        print("=" * 60)

        train_tokens = self.calculate_total_tokens(final_train_data)
        validation_tokens = self.calculate_total_tokens(final_validation_data)
        total_tokens = train_tokens + validation_tokens

        print(f"\nTrain Data (text):")
        print(f"  Conversations : {len(final_train_data):,}")
        print(f"  Tokens        : {train_tokens:,}")

        print(f"\nValidation Data (text):")
        print(f"  Conversations : {len(final_validation_data):,}")
        print(f"  Tokens        : {validation_tokens:,}")

        target_total = sum(TARGETS.values())
        print(f"\nTarget Total Tokens  : {target_total:,}")
        print(f"Achievement Rate     : {(total_tokens / target_total * 100):.2f}%")

        # Step 4: Pack sequences (BOS/EOT + sort + merge + pad)
        packed_train, packed_validation = self.pack_data(
            final_train_data, final_validation_data
        )

        # Post-pack statistics
        print("\n" + "=" * 60)
        print("POST-PACK STATISTICS")
        print("=" * 60)

        print(f"\nTrain Data:")
        print(f"  Sequences     : {len(packed_train):,}")
        print(f"  Tokens each   : {self.max_context}")
        print(f"  Total tokens  : {len(packed_train) * self.max_context:,}")

        print(f"\nValidation Data:")
        print(f"  Sequences     : {len(packed_validation):,}")
        print(f"  Tokens each   : {self.max_context}")
        print(f"  Total tokens  : {len(packed_validation) * self.max_context:,}")

        # Steps 5 + 6: Validate then shard
        valid_train, valid_validation = self.validate_and_shard(
            packed_train, packed_validation
        )

        # Final statistics
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)

        print(f"\nTrain Data (validated):")
        print(f"  Sequences     : {len(valid_train):,}")
        print(f"  Tokens each   : {self.max_context}")
        print(f"  Total tokens  : {len(valid_train) * self.max_context:,}")

        print(f"\nValidation Data (validated):")
        print(f"  Sequences     : {len(valid_validation):,}")
        print(f"  Tokens each   : {self.max_context}")
        print(f"  Total tokens  : {len(valid_validation) * self.max_context:,}")

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED")
        print("=" * 60)

        return valid_train, valid_validation


def main():
    """Main entry point."""
    # Initialize pipeline
    pipeline = DataPreparationPipeline(seed=SEED)

    # Run pipeline – returns packed, padded text strings (no encoding)
    packed_train, packed_validation = pipeline.run()

    # Save packed text strings: one packed sequence per line
    print("\nSaving packed sequences to files...")

    with open("final_train_data.txt", "w", encoding="utf-8") as f:
        for seq in packed_train:
            f.write(seq + "\n")

    with open("final_validation_data.txt", "w", encoding="utf-8") as f:
        for seq in packed_validation:
            f.write(seq + "\n")

    print("✓ Saved final_train_data.txt")
    print("✓ Saved final_validation_data.txt")

    return packed_train, packed_validation


if __name__ == "__main__":
    main()
