"""
ConversationValidator
=====================
Validates packed conversation strings **after** BOS/EOT tokens have been
added by TokenPacker.wrap_with_special_tokens().

Expected structure of every packed sequence:

    <|BOS|>
    <|USER|>...<|EOS|>
    <|ASSISTANT|>...<|EOS|>
    <|USER|>...<|EOS|>
    <|ASSISTANT|>...<|EOS|>
    ...
    <|EOT|>

Rules enforced
--------------
1. Sequence must start with <|BOS|>.
2. Sequence must end with <|EOT|>.
3. First conversational turn must be USER  (BOS → USER → …).
4. Last  conversational turn must be ASSISTANT (… → ASSISTANT → EOT).
5. USER and ASSISTANT must strictly alternate — no consecutive same-role turns.

Validation is performed on the **tokenized representation** (token ID
sequences), not raw strings, so BOS/USER/ASSISTANT/EOS/EOT are detected
via their actual token IDs as registered in data_sanitizer.utils.

Usage
-----
    from data_sanitizer.conversation_validator import ConversationValidator

    validator = ConversationValidator()
    valid_sequences = validator.validate(packed_sequences)
    # Raises ValueError if any sequence is invalid.
    # Returns the original list unchanged when all sequences are valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

# Re-use the single encoder and token-ID map that the rest of the project uses.
# Never redefine special tokens here.
from data_sanitizer.utils import enc, special_tokens


# ---------------------------------------------------------------------------
# Token IDs for the special tokens we validate against
# ---------------------------------------------------------------------------
_BOS_ID       = special_tokens["<|BOS|>"]
_USER_ID      = special_tokens["<|USER|>"]
_ASSISTANT_ID = special_tokens["<|ASSISTANT|>"]
_EOS_ID       = special_tokens["<|EOS|>"]
_EOT_ID       = special_tokens["<|EOT|>"]

# Human-readable names for diagnostic messages
_TOKEN_NAMES = {
    _BOS_ID:       "BOS",
    _USER_ID:      "USER",
    _ASSISTANT_ID: "ASSISTANT",
    _EOS_ID:       "EOS",
    _EOT_ID:       "EOT",
}


# ---------------------------------------------------------------------------
# Validation result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    """Outcome of validating a single packed sequence."""

    index:   int          # position in the input list
    passed:  bool         # True → valid; False → invalid
    reasons: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        if self.passed:
            return f"[{status}] sequence #{self.index}"
        return f"[{status}] sequence #{self.index}: {'; '.join(self.reasons)}"


# ---------------------------------------------------------------------------
# ConversationValidator
# ---------------------------------------------------------------------------
class ConversationValidator:
    """
    Validates packed conversation strings after BOS/EOT tokens are added.

    The validator encodes each string to token IDs using the project's shared
    encoder, then checks the structural rules listed in the module docstring.

    Parameters
    ----------
    strict : bool
        When True (default) a ValueError is raised if any sequence is invalid.
        When False the caller receives the valid subset and must inspect the
        returned results to detect failures.
    """

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def validate(
        self,
        packed_sequences: List[str],
    ) -> List[str]:
        """
        Validate every packed sequence and return only the valid ones.

        Args:
            packed_sequences: List of packed conversation strings produced by
                              TokenPacker.  Each string must already contain
                              the <|BOS|> prefix and <|EOT|> suffix.

        Returns:
            The subset of ``packed_sequences`` that passed all validation
            rules (order is preserved).

        Raises:
            ValueError: If ``strict=True`` (the default) and one or more
                        sequences fail validation.  The error message lists
                        the failure reasons for every invalid sequence.
        """
        print(f"\n  [ConversationValidator] Validating {len(packed_sequences):,} sequences...")

        results: List[ValidationResult] = []

        for idx, seq in enumerate(packed_sequences):
            result = self._validate_single(idx, seq)
            results.append(result)

        invalid_results = [r for r in results if not r.passed]
        valid_sequences  = [
            seq
            for seq, r in zip(packed_sequences, results)
            if r.passed
        ]

        # ---- Report -------------------------------------------------------
        n_total   = len(packed_sequences)
        n_invalid = len(invalid_results)
        n_valid   = n_total - n_invalid

        print(f"  [ConversationValidator] Total    : {n_total:,}")
        print(f"  [ConversationValidator] Valid    : {n_valid:,}")
        print(f"  [ConversationValidator] Invalid  : {n_invalid:,}")

        if n_invalid > 0:
            self._report_invalid(invalid_results)

            if self.strict:
                summary = "\n".join(str(r) for r in invalid_results)
                raise ValueError(
                    f"\n[ConversationValidator] {n_invalid} invalid sequence(s) found.\n"
                    f"Processing halted to prevent corrupt shards.\n\n"
                    f"Invalid sequences:\n{summary}"
                )
        else:
            print("  [ConversationValidator] ✓ All sequences passed validation.")

        return valid_sequences

    # ------------------------------------------------------------------
    # Per-sequence validation logic
    # ------------------------------------------------------------------
    def _validate_single(self, idx: int, seq: str) -> ValidationResult:
        """Validate one packed sequence and return a ValidationResult."""
        token_ids = enc.encode(seq, allowed_special="all")
        reasons: List[str] = []

        # ---- Rule 1: must start with BOS ----------------------------------
        if not token_ids or token_ids[0] != _BOS_ID:
            first = _token_name(token_ids[0]) if token_ids else "<empty>"
            reasons.append(f"Rule 1 – does not start with BOS (starts with {first})")

        # ---- Rule 2: must end with EOT ------------------------------------
        if not token_ids or token_ids[-1] != _EOT_ID:
            last = _token_name(token_ids[-1]) if token_ids else "<empty>"
            reasons.append(f"Rule 2 – does not end with EOT (ends with {last})")

        # ---- Extract speaker-role token sequence --------------------------
        # Walk the token IDs and collect an ordered list of (USER|ASSISTANT)
        # markers so we can check alternation.
        speaker_tokens = [
            tok for tok in token_ids
            if tok in (_USER_ID, _ASSISTANT_ID)
        ]

        if not speaker_tokens:
            reasons.append("Rule 3/4/5 – no USER or ASSISTANT tokens found")
            return ValidationResult(index=idx, passed=False, reasons=reasons)

        # ---- Rule 3: must start with USER ---------------------------------
        if speaker_tokens[0] != _USER_ID:
            reasons.append(
                f"Rule 3 – first turn is {_token_name(speaker_tokens[0])}, "
                f"expected USER"
            )

        # ---- Rule 4: must end with ASSISTANT ------------------------------
        if speaker_tokens[-1] != _ASSISTANT_ID:
            reasons.append(
                f"Rule 4 – last turn is {_token_name(speaker_tokens[-1])}, "
                f"expected ASSISTANT"
            )

        # ---- Rule 5: turns must strictly alternate ------------------------
        for i in range(1, len(speaker_tokens)):
            if speaker_tokens[i] == speaker_tokens[i - 1]:
                role = _token_name(speaker_tokens[i])
                reasons.append(
                    f"Rule 5 – consecutive {role} turns at positions "
                    f"{i - 1} and {i} in speaker sequence"
                )
                break  # report only the first violation to avoid noise

        passed = len(reasons) == 0
        return ValidationResult(index=idx, passed=passed, reasons=reasons)

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _report_invalid(invalid_results: List[ValidationResult]) -> None:
        """Print a human-readable summary of all invalid sequences."""
        print("\n  [ConversationValidator] ── Invalid Sequences ──────────────────────")
        for r in invalid_results:
            print(f"    {r}")
        print("  [ConversationValidator] ────────────────────────────────────────────\n")

    # ------------------------------------------------------------------
    # Convenience: validate and return (valid, invalid) pair
    # ------------------------------------------------------------------
    def validate_with_report(
        self,
        packed_sequences: List[str],
    ) -> Tuple[List[str], List[ValidationResult]]:
        """
        Non-raising variant that always returns both the valid subset and the
        full list of ValidationResult objects.

        Useful when the caller wants to inspect or log invalid entries rather
        than halt immediately.

        Returns:
            (valid_sequences, results)  where results covers every input item.
        """
        results: List[ValidationResult] = [
            self._validate_single(idx, seq)
            for idx, seq in enumerate(packed_sequences)
        ]

        valid_sequences = [
            seq
            for seq, r in zip(packed_sequences, results)
            if r.passed
        ]

        return valid_sequences, results


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------
def _token_name(token_id: int) -> str:
    """Return a human-readable name for a token ID, or its numeric value."""
    return _TOKEN_NAMES.get(token_id, str(token_id))
