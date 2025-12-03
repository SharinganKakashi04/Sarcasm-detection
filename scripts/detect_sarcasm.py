import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import sys

# ============================================================================
# MODEL DEFINITION (Must match your training code)
# ============================================================================

class DistilBERTClassifier(nn.Module):
    """Baseline DistilBERT classifier"""
    
    def __init__(self, num_classes=2, dropout=0.3):
        super(DistilBERTClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# ============================================================================
# SARCASM DETECTOR CLASS
# ============================================================================

class SarcasmDetector:
    """Easy-to-use sarcasm detection interface"""
    
    def __init__(self, model_path='models/baseline_best.pt', device=None):
        """
        Initialize the sarcasm detector
        
        Args:
            model_path: Path to saved model weights
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading model on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Load model
        self.model = DistilBERTClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print("✓ Model loaded successfully!")
    
    def predict(self, text, return_confidence=False):
        """
        Predict if text is sarcastic
        
        Args:
            text: Input text string
            return_confidence: If True, returns confidence scores
            
        Returns:
            If return_confidence=False: 'sarcastic' or 'non-sarcastic'
            If return_confidence=True: (prediction, confidence_dict)
        """
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)[0]
            prediction = torch.argmax(logits, dim=1).item()
        
        # Convert to label
        label = 'sarcastic' if prediction == 1 else 'non-sarcastic'
        
        if return_confidence:
            confidence = {
                'non-sarcastic': probabilities[0].item(),
                'sarcastic': probabilities[1].item()
            }
            return label, confidence
        else:
            return label
    
    def predict_batch(self, texts):
        """
        Predict multiple texts at once
        
        Args:
            texts: List of text strings
            
        Returns:
            List of predictions
        """
        predictions = []
        for text in texts:
            pred = self.predict(text)
            predictions.append(pred)
        return predictions

# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_mode(detector):
    """Interactive command-line interface"""
    print("\n" + "="*70)
    print("SARCASM DETECTION - INTERACTIVE MODE")
    print("="*70)
    print("\nEnter text to check for sarcasm.")
    print("Commands:")
    print("  - Type 'quit' or 'exit' to quit")
    print("  - Type 'batch' to enter batch mode")
    print("  - Type 'examples' to see example predictions")
    print("-"*70 + "\n")
    
    while True:
        text = input("Enter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        elif text.lower() == 'batch':
            batch_mode(detector)
            continue
        
        elif text.lower() == 'examples':
            show_examples(detector)
            continue
        
        elif not text:
            print("⚠️  Please enter some text\n")
            continue
        
        # Predict
        label, confidence = detector.predict(text, return_confidence=True)
        
        # Display result
        print(f"\n{'='*70}")
        print(f"Text: {text}")
        print(f"{'='*70}")
        print(f"Prediction: {label.upper()}")
        print(f"\nConfidence Scores:")
        print(f"  Non-sarcastic: {confidence['non-sarcastic']:.2%}")
        print(f"  Sarcastic:     {confidence['sarcastic']:.2%}")
        
        # Visual indicator
        if label == 'sarcastic':
            bar_length = int(confidence['sarcastic'] * 50)
            print(f"\n  {'█' * bar_length}{'░' * (50-bar_length)} {confidence['sarcastic']:.1%}")
        else:
            bar_length = int(confidence['non-sarcastic'] * 50)
            print(f"\n  {'█' * bar_length}{'░' * (50-bar_length)} {confidence['non-sarcastic']:.1%}")
        
        print()

def batch_mode(detector):
    """Batch prediction mode"""
    print("\n" + "="*70)
    print("BATCH MODE")
    print("="*70)
    print("Enter multiple texts (one per line)")
    print("Type 'done' when finished\n")
    
    texts = []
    while True:
        text = input(f"Text {len(texts)+1}: ").strip()
        if text.lower() == 'done':
            break
        if text:
            texts.append(text)
    
    if not texts:
        print("No texts entered.\n")
        return
    
    print(f"\nProcessing {len(texts)} texts...\n")
    
    results = []
    for text in texts:
        label, confidence = detector.predict(text, return_confidence=True)
        results.append((text, label, confidence))
    
    # Display results
    print("\n" + "="*70)
    print("BATCH RESULTS")
    print("="*70 + "\n")
    
    for i, (text, label, conf) in enumerate(results, 1):
        emoji = "😏" if label == 'sarcastic' else "😊"
        print(f"{i}. {emoji} [{label.upper()}] (confidence: {conf[label]:.1%})")
        print(f"   {text[:100]}{'...' if len(text) > 100 else ''}")
        print()

def show_examples(detector):
    """Show example predictions"""
    examples = [
        "I love waiting in long lines at the DMV!",
        "This project is due tomorrow and I haven't started. Perfect.",
        "Great! Another meeting that could have been an email.",
        "The weather is beautiful today!",
        "I'm really excited about this new opportunity.",
        "Yeah right, like that's ever going to happen.",
        "Oh wonderful, my code has 50 new bugs.",
        "I had a great time at the party yesterday."
    ]
    
    print("\n" + "="*70)
    print("EXAMPLE PREDICTIONS")
    print("="*70 + "\n")
    
    for text in examples:
        label, confidence = detector.predict(text, return_confidence=True)
        emoji = "😏" if label == 'sarcastic' else "😊"
        print(f"{emoji} [{label.upper()}] {text}")
        print(f"   Confidence: {confidence[label]:.1%}\n")

# ============================================================================
# SIMPLE FUNCTION INTERFACE
# ============================================================================

def detect_sarcasm(text, model_path='models/baseline_best.pt'):
    """
    Simple one-line function to detect sarcasm
    
    Args:
        text: Input text string
        model_path: Path to model weights
        
    Returns:
        'sarcastic' or 'non-sarcastic'
    
    Example:
        >>> result = detect_sarcasm("Oh great, another rainy day!")
        >>> print(result)
        'sarcastic'
    """
    detector = SarcasmDetector(model_path)
    return detector.predict(text)

# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Sarcasm Detection Tool')
    parser.add_argument('--text', type=str, help='Text to classify')
    parser.add_argument('--model', type=str, default='models/baseline_best.pt',
                       help='Path to model weights')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive mode')
    parser.add_argument('--batch', action='store_true',
                       help='Batch prediction mode')
    
    args = parser.parse_args()
    
    # Load detector
    try:
        detector = SarcasmDetector(args.model)
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at '{args.model}'")
        print("\nAvailable models:")
        print("  - models/baseline_best.pt")
        print("  - models/incongruity_best.pt")
        print("  - models/improved_incongruity_best.pt")
        print("  - models/ensemble_model_1.pt (or 2, 3)")
        return
    
    # Single text prediction
    if args.text:
        label, confidence = detector.predict(args.text, return_confidence=True)
        print(f"\nText: {args.text}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence[label]:.2%}")
        return
    
    # Batch mode
    if args.batch:
        batch_mode(detector)
        return
    
    # Interactive mode (default)
    interactive_mode(detector)

if __name__ == "__main__":
    # If run with no arguments, start interactive mode
    if len(sys.argv) == 1:
        try:
            detector = SarcasmDetector()
            interactive_mode(detector)
        except FileNotFoundError:
            print("❌ Error: No trained model found!")
            print("\nPlease train a model first or specify model path:")
            print("  python scripts/detect_sarcasm.py --model path/to/model.pt")
    else:
        main()