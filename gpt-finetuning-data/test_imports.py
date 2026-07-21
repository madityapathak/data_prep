"""
Test script to verify all imports work correctly.
"""

print("Testing imports...")

print("\n1. Testing dataset processors...")
try:
    from get_formatted_datasets.daily_dialog import DailyDialogProcessor
    print("  ✓ DailyDialogProcessor")
except Exception as e:
    print(f"  ✗ DailyDialogProcessor: {e}")

try:
    from get_formatted_datasets.databricks_dolly import DollyProcessor
    print("  ✓ DollyProcessor")
except Exception as e:
    print(f"  ✗ DollyProcessor: {e}")

try:
    from get_formatted_datasets.oasst import OASSTProcessor
    print("  ✓ OASSTProcessor")
except Exception as e:
    print(f"  ✗ OASSTProcessor: {e}")

try:
    from get_formatted_datasets.open_orca import OpenOrcaProcessor
    print("  ✓ OpenOrcaProcessor")
except Exception as e:
    print(f"  ✗ OpenOrcaProcessor: {e}")

try:
    from get_formatted_datasets.ultrachat import UltraChatProcessor
    print("  ✓ UltraChatProcessor")
except Exception as e:
    print(f"  ✗ UltraChatProcessor: {e}")

try:
    from get_formatted_datasets.utils import TARGETS
    print("  ✓ TARGETS dict")
except Exception as e:
    print(f"  ✗ TARGETS: {e}")

print("\n2. Testing sanitizers...")
try:
    from data_sanitizer.dataset_cleaner import DatasetCleaner
    print("  ✓ DatasetCleaner")
except Exception as e:
    print(f"  ✗ DatasetCleaner: {e}")

try:
    from data_sanitizer.conversation_filter import ConversationFilter
    print("  ✓ ConversationFilter")
except Exception as e:
    print(f"  ✗ ConversationFilter: {e}")

try:
    from data_sanitizer.conversation_window_processor import ConversationWindowProcessor
    print("  ✓ ConversationWindowProcessor")
except Exception as e:
    print(f"  ✗ ConversationWindowProcessor: {e}")

try:
    from data_sanitizer.utils import num_tokens
    print("  ✓ num_tokens")
except Exception as e:
    print(f"  ✗ num_tokens: {e}")

print("\n3. Testing main pipeline...")
try:
    from prepare_training_data import DataPreparationPipeline
    print("  ✓ DataPreparationPipeline")
except Exception as e:
    print(f"  ✗ DataPreparationPipeline: {e}")

print("\n✅ All imports tested!")
