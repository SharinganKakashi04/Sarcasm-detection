import torch
import transformers
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

print("🔍 Checking installation...")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ Transformers version: {transformers.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# Test VADER
analyzer = SentimentIntensityAnalyzer()
test_text = "This is great!"
scores = analyzer.polarity_scores(test_text)    
print(f"✓ VADER working: {scores}")

# Check data
try:
    df = pd.read_csv('data/train.csv')
    print(f"✓ Dataset loaded: {len(df)} samples")
except:
    print("⚠ Dataset not found - run download_data.py first")

print("\n✅ Setup complete!")