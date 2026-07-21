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
        final_train_data, final_validation_data = main()
        
        print("\n✅ Pipeline completed successfully!")
        print(f"\n📊 Results:")
        print(f"   - Training samples: {len(final_train_data):,}")
        print(f"   - Validation samples: {len(final_validation_data):,}")
        print(f"\n💾 Output files:")
        print(f"   - final_train_data.txt")
        print(f"   - final_validation_data.txt")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error:")
        print(f"   {type(e).__name__}: {e}")
        raise
