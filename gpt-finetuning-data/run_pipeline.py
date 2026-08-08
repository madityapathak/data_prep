#!/usr/bin/env python3
"""
Simple runner script to execute the data preparation pipeline.

Usage:
    python run_pipeline.py
"""

from prepare_training_data import main

if __name__ == "__main__":
    print("\n🚀 Starting LLM Fine-Tuning Data Preparation Pipeline\n")
    
    try:
        packed_train, packed_validation = main()
        
        print("\n✅ Pipeline completed successfully!")
        print(f"\n📊 Results (packed, fixed-length sequences):")
        print(f"   - Training sequences  : {len(packed_train):,}")
        print(f"   - Validation sequences: {len(packed_validation):,}")
        print(f"\n💾 Output files (one token-id sequence per line):")
        print(f"   - final_train_data.txt")
        print(f"   - final_validation_data.txt")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error:")
        print(f"   {type(e).__name__}: {e}")
        raise
