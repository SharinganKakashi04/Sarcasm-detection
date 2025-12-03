from datasets import load_dataset
import pandas as pd
import os

def download_sarcasm_dataset():
    """Download and save sarcasm detection dataset"""
    
    print("Downloading dataset...")
    
    # Using the tweet_eval dataset's irony subset
    dataset = load_dataset("tweet_eval", "irony")
    
    # Convert to pandas for easier exploration
    train_df = pd.DataFrame(dataset['train'])
    val_df = pd.DataFrame(dataset['validation'])
    test_df = pd.DataFrame(dataset['test'])
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save as CSV
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    print(f"✓ Training samples: {len(train_df)}")
    print(f"✓ Validation samples: {len(val_df)}")
    print(f"✓ Test samples: {len(test_df)}")
    print(f"✓ Sarcastic samples in train: {train_df['label'].sum()}")
    print("\nDataset saved to data/ directory")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    download_sarcasm_dataset()