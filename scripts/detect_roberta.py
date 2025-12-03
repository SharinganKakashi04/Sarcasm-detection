import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
import sys

# ============================================================================
# MODEL DEFINITION (Must match training code)
# ============================================================================

class RobertaSarcasmClassifier(nn.Module):
    """RoBERTa-based sarcasm classifier"""
    
    def __init__(self, num_classes=2, dropout=0.3, hidden_size=256):
        super(RobertaSarcasmClassifier, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        roberta_hidden = 768
        
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(roberta_hidden, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln2 = nn.LayerNorm(hidden_size // 2)
        
        self.dropout3 = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size // 2, num_classes)
        
        self.num_samples = 5
    
    def forward(self, input_ids, attention_mask, training=False):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled = outputs.last_hidden_state[:, 0]
        
        # Single forward pass for inference
        x = self.dropout1(pooled)
        x = F.relu(self.ln1(self.fc1(x)))
        
        x = self.dropout2(x)
        x = F.relu(self.ln2(self.fc2(x)))
        
        x = self.dropout3(x)
        logits = self.classifier(x)
        
        return logits

# ============================================================================
# SARCASM DETECTOR
# ============================================================================

class RobertaSarcasmDetector:
    """Easy-to-use RoBERTa sarcasm detection interface"""
    
    def __init__(self, model_path='models/roberta_best.pt', device=None):
        """
        Initialize the sarcasm detector
        
        Args:
            model_path: Path to saved RoBERTa model weights
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🤖 Loading RoBERTa model on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
        # Load model
        self.model = RobertaSarcasmClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Model loaded successfully!\n")
    
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
            logits = self.model(input_ids, attention_mask, training=False)
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
            List of (text, prediction, confidence) tuples
        """
        results = []
        for text in texts:
            label, confidence = self.predict(text, return_confidence=True)
            results.append((text, label, confidence))
        return results

# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_mode(detector):
    """Interactive command-line interface"""
    print("="*80)
    print("🎭 SARCASM DETECTION - INTERACTIVE MODE (RoBERTa Model)")
    print("="*80)
    print("\n📝 Enter text to check for sarcasm.")
    print("\nCommands:")
    print("  • Type 'quit' or 'exit' to quit")
    print("  • Type 'batch' to test multiple texts")
    print("  • Type 'examples' to see example predictions")
    print("  • Type 'test' to test with your previous failing cases")
    print("-"*80 + "\n")
    
    while True:
        text = input("💬 Enter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        elif text.lower() == 'batch':
            batch_mode(detector)
            continue
        
        elif text.lower() == 'examples':
            show_examples(detector)
            continue
        
        elif text.lower() == 'test':
            test_previous_failures(detector)
            continue
        
        elif not text:
            print("⚠️  Please enter some text\n")
            continue
        
        # Predict
        label, confidence = detector.predict(text, return_confidence=True)
        
        # Display result with nice formatting
        print(f"\n{'='*80}")
        print(f"📄 Text: {text}")
        print(f"{'='*80}")
        
        # Emoji based on prediction
        emoji = "😏" if label == 'sarcastic' else "😊"
        
        print(f"\n{emoji} Prediction: {label.upper()}")
        print(f"\n📊 Confidence Scores:")
        print(f"   Non-sarcastic: {confidence['non-sarcastic']:.1%}")
        print(f"   Sarcastic:     {confidence['sarcastic']:.1%}")
        
        # Visual confidence bar
        if label == 'sarcastic':
            bar_length = int(confidence['sarcastic'] * 60)
            bar_color = '█'
        else:
            bar_length = int(confidence['non-sarcastic'] * 60)
            bar_color = '█'
        
        print(f"\n   {bar_color * bar_length}{'░' * (60-bar_length)} {confidence[label]:.1%}")
        print()

def batch_mode(detector):
    """Batch prediction mode"""
    print("\n" + "="*80)
    print("📦 BATCH MODE")
    print("="*80)
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
        print("❌ No texts entered.\n")
        return
    
    print(f"\n⏳ Processing {len(texts)} texts...\n")
    
    results = detector.predict_batch(texts)
    
    # Display results
    print("\n" + "="*80)
    print("📊 BATCH RESULTS")
    print("="*80 + "\n")
    
    for i, (text, label, conf) in enumerate(results, 1):
        emoji = "😏" if label == 'sarcastic' else "😊"
        confidence_pct = conf[label] * 100
        
        print(f"{i}. {emoji} [{label.upper():15}] ({confidence_pct:5.1f}%)")
        print(f"   {text[:100]}{'...' if len(text) > 100 else ''}")
        print()

def test_previous_failures(detector):
    """Test with cases that failed with the baseline model"""
    test_cases = [
        ("I like the weather today", "non-sarcastic"),
        ("I think that its too hot today", "non-sarcastic"),
        ("i love you", "non-sarcastic"),
        ("Oh great, another Monday morning", "sarcastic"),
        ("Yeah right, like that's gonna work", "sarcastic"),
        ("Perfect, just what I needed today", "sarcastic"),
        ("I'm so happy to be stuck in traffic", "sarcastic"),
        ("The weather is beautiful today", "non-sarcastic"),
        ("I had a great day at work", "non-sarcastic"),
    ]
    
    print("\n" + "="*80)
    print("🧪 TESTING PREVIOUS FAILURE CASES")
    print("="*80 + "\n")
    
    correct = 0
    total = len(test_cases)
    
    for text, expected in test_cases:
        label, confidence = detector.predict(text, return_confidence=True)
        
        is_correct = label == expected
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        emoji = "😏" if label == 'sarcastic' else "😊"
        
        print(f"{status} {emoji} [{label:15}] (conf: {confidence[label]:.1%}) | Expected: {expected}")
        print(f"   '{text}'")
        
        if not is_correct:
            print(f"   ⚠️  Misclassified! Confidence: {confidence[label]:.1%}")
        print()
    
    accuracy = (correct / total) * 100
    print(f"{'='*80}")
    print(f"📈 Accuracy on test cases: {correct}/{total} ({accuracy:.1f}%)")
    print(f"{'='*80}\n")

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
        "I had a great time at the party yesterday.",
        "Sure, because that makes total sense.",
        "Thanks for nothing."
    ]
    
    print("\n" + "="*80)
    print("💡 EXAMPLE PREDICTIONS")
    print("="*80 + "\n")
    
    for text in examples:
        label, confidence = detector.predict(text, return_confidence=True)
        emoji = "😏" if label == 'sarcastic' else "😊"
        print(f"{emoji} [{label.upper():15}] (conf: {confidence[label]:.1%})")
        print(f"   '{text}'")
        print()

# ============================================================================
# SIMPLE FUNCTION INTERFACE
# ============================================================================

def detect_sarcasm(text, model_path='models/roberta_best.pt'):
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
    detector = RobertaSarcasmDetector(model_path)
    return detector.predict(text)

# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='RoBERTa Sarcasm Detection Tool')
    parser.add_argument('--text', type=str, help='Text to classify')
    parser.add_argument('--model', type=str, default='models/roberta_best.pt',
                       help='Path to RoBERTa model weights')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive mode')
    parser.add_argument('--batch', action='store_true',
                       help='Batch prediction mode')
    parser.add_argument('--test', action='store_true',
                       help='Test with previous failing cases')
    
    args = parser.parse_args()
    
    # Load detector
    try:
        detector = RobertaSarcasmDetector(args.model)
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at '{args.model}'")
        print("\nMake sure you've trained the RoBERTa model:")
        print("  python scripts/roberta_model.py")
        return
    
    # Single text prediction
    if args.text:
        label, confidence = detector.predict(args.text, return_confidence=True)
        emoji = "😏" if label == 'sarcastic' else "😊"
        print(f"\n{emoji} Text: {args.text}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence[label]:.1%}\n")
        return
    
    # Test mode
    if args.test:
        test_previous_failures(detector)
        return
    
    # Batch mode
    if args.batch:
        batch_mode(detector)
        return
    
    # Interactive mode (default)
    interactive_mode(detector)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        try:
            detector = RobertaSarcasmDetector()
            interactive_mode(detector)
        except FileNotFoundError:
            print("❌ Error: No trained RoBERTa model found!")
            print("\nPlease train the model first:")
            print("  python scripts/roberta_model.py")
    else:
        main()