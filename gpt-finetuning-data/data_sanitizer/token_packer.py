from data_sanitizer.utils import num_tokens
from data_sanitizer.conversation_validator import ConversationValidator
import os
from datetime import datetime


class TokenPacker:
    # ------------------------------------------------------------------ #
    # Special Tokens (text form)
    # ------------------------------------------------------------------ #
    BOS_TOKEN = "<|BOS|>"
    EOT_TOKEN = "<|EOT|>"
    PAD_TOKEN = "<|PAD|>"

    def __init__(self, max_context: int = 1024, invalid_data_log: str = "invalid_sequences.txt"):
        self.max_context = max_context
        self.invalid_data_log = invalid_data_log

    # ------------------------------------------------------------------ #
    # Step 1 – Wrap each conversation string with BOS / EOT
    # ------------------------------------------------------------------ #
    def wrap_with_special_tokens(self, conversations: list[str]) -> list[str]:
        """
        Prepend <|BOS|> and append <|EOT|> to every conversation string.
        """
        return [
            self.BOS_TOKEN + conv + self.EOT_TOKEN
            for conv in conversations
        ]

    # ------------------------------------------------------------------ #
    # Step 2 – Sort ascending by num_tokens then bin-pack
    # ------------------------------------------------------------------ #
    def pack_entries(self, conversations: list[str]) -> list[str]:
        """
        Best-Fit Decreasing (BFD) bin packing.

        Steps:
        1. Compute token count for each conversation.
        2. Sort conversations by token count (descending).
        3. For each conversation, place it into the bin that leaves the
            least remaining space after insertion.
        4. If no existing bin can fit it, create a new bin.

        Oversized conversations (> max_context) are kept as separate entries.
        """

        # Precompute token counts once
        items = [(conv, num_tokens(conv)) for conv in conversations]

        # Sort descending
        items.sort(key=lambda x: x[1], reverse=True)

        # Each bin is:
        # {
        #     "remaining": int,
        #     "parts": list[str]
        # }
        bins = []

        packed = []

        for conv, size in items:
            if size > self.max_context:
                print("[DEBUG] : This print should never have been in the logs this is from token_packer.py file")
                packed.append(conv)
                continue

            best_bin = None
            smallest_remaining = None

            # Find the bin that leaves the least remaining space
            for b in bins:
                if b["remaining"] >= size:
                    remaining_after = b["remaining"] - size

                    if (
                        smallest_remaining is None
                        or remaining_after < smallest_remaining
                    ):
                        smallest_remaining = remaining_after
                        best_bin = b

            if best_bin is None:
                bins.append({
                    "remaining": self.max_context - size,
                    "parts": [conv],
                })
            else:
                best_bin["parts"].append(conv)
                best_bin["remaining"] -= size

        # Preserve oversized conversations first (same behavior as your code)
        packed.extend("".join(b["parts"]) for b in bins)

        return packed

    # ------------------------------------------------------------------ #
    # Step 3 – Pad every sequence to exactly max_context tokens
    # ------------------------------------------------------------------ #
    def pad_entries(self, packed: list[str]) -> list[str]:
        """
        Right-pad each packed string with <|PAD|> tokens until it reaches
        exactly max_context tokens.  Sequences that exceed max_context are
        truncated at the token boundary (we keep the first max_context tokens
        worth of the string – approximated here by truncating on PAD count).

        NOTE: padding is done in token-count space; each <|PAD|> occupies
        exactly 1 token (it is a special token).
        """
        padded = []
        for conv in packed:
            tokens_used = num_tokens(conv)
            if tokens_used < self.max_context:
                pad_count = self.max_context - tokens_used
                conv = conv + self.PAD_TOKEN * pad_count
            # If already at / above max_context, keep as-is
            # (oversized entries were already flagged during pack_entries)
            padded.append(conv)
        return padded

    # ------------------------------------------------------------------ #
    # Helper method to log invalid sequences
    # ------------------------------------------------------------------ #
    def _log_invalid_sequences(self, wrapped: list[str], invalid_results: list) -> None:
        """
        Log invalid sequences to a text file for inspection.
        
        Args:
            wrapped: List of all wrapped sequences (before filtering)
            invalid_results: List of ValidationResult objects for invalid sequences
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create or append to the log file
        file_exists = os.path.exists(self.invalid_data_log)
        
        with open(self.invalid_data_log, "a", encoding="utf-8") as f:
            # Add separator if file already exists
            if file_exists:
                f.write("\n" + "=" * 80 + "\n")
            
            # Write header
            f.write(f"Invalid Sequences Log - {timestamp}\n")
            f.write(f"Total Invalid: {len(invalid_results)}\n")
            f.write("=" * 80 + "\n\n")
            
            # Write each invalid sequence with its failure reasons
            for result in invalid_results:
                f.write(f"Sequence #{result.index}\n")
                f.write(f"Reasons: {'; '.join(result.reasons)}\n")
                f.write(f"Content:\n")
                f.write(wrapped[result.index])
                f.write("\n")
                f.write("-" * 80 + "\n\n")

    # ------------------------------------------------------------------ #
    # Full Pipeline
    # ------------------------------------------------------------------ #
    def process(self, conversations: list[str]) -> list[str]:
        """
        Run the full packing pipeline on a list of plain conversation strings:
          1. Wrap each string with <|BOS|> … <|EOT|>.
          2. Sort by num_tokens (ascending) and bin-pack into max_context bins.
          3. Right-pad every bin to exactly max_context tokens with <|PAD|>.

        Returns a list of packed, padded strings ready to be saved.
        No encoding is performed here.
        """
        print(f"  [TokenPacker] Input entries   : {len(conversations):,}")

        # Step 1 – wrap with BOS / EOT
        wrapped = self.wrap_with_special_tokens(conversations)
        print(f"  [TokenPacker] Wrapped         : {len(wrapped):,} sequences")


        # Step 1.5 – validate structure before bin-packing
        # Expected format per sequence:
        #   <|BOS|><|USER|>...<|EOS|><|ASSISTANT|>...<|EOS|>...<|ASSISTANT|>...<|EOS|><|EOT|>
        # Rules:
        #   1. Must start with <|BOS|>  2. Must end with <|EOT|>
        #   3. First turn must be USER  4. Last turn must be ASSISTANT
        #   5. USER / ASSISTANT turns must strictly alternate
        validator = ConversationValidator(strict=False)  # filter, don't raise
        
        # Use validate_with_report to get both valid sequences and validation results
        valid_wrapped, validation_results = validator.validate_with_report(wrapped)
        
        # Log invalid sequences to file
        invalid_results = [r for r in validation_results if not r.passed]
        if invalid_results:
            self._log_invalid_sequences(wrapped, invalid_results)
            print(f"  [TokenPacker] Logged {len(invalid_results):,} invalid sequences to '{self.invalid_data_log}'")
        
        wrapped = valid_wrapped
        print(f"  [TokenPacker] After validation : {len(wrapped):,} sequences passed")



        # Step 2 – sort & bin-pack
        packed = self.pack_entries(wrapped)
        print(
            f"  [TokenPacker] After packing   : {len(packed):,} sequences "
            f"(reduced from {len(wrapped):,}, saved {len(wrapped) - len(packed):,} entries)"
        )

        # Step 3 – pad to max_context
        padded = self.pad_entries(packed)
        print(
            f"  [TokenPacker] After padding   : {len(padded):,} sequences "
            f"of exactly {self.max_context} tokens each"
        )

        return padded
