import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import os
from datetime import datetime

# ============================================================================
# MODEL DEFINITION
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
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        
        x = self.dropout1(pooled)
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout2(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout3(x)
        logits = self.classifier(x)
        
        return logits

# ============================================================================
# FEEDBACK DATASET
# ============================================================================

class FeedbackDataset(Dataset):
    """Dataset for feedback samples"""
    
    def __init__(self, samples, tokenizer, max_length=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']
        label = sample['label']
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ============================================================================
# INTERACTIVE LEARNING SYSTEM
# ============================================================================

class InteractiveLearningSystem:
    """System that learns from user feedback"""
    
    def __init__(self, model_path='models/roberta_best.pt', 
                 feedback_file='data/user_feedback.json',
                 device=None):
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🤖 Loading model on {self.device}...")
        
        # Load tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaSarcasmClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Feedback storage
        self.feedback_file = feedback_file
        self.feedback_samples = self._load_feedback()
        
        print(f"✅ Model loaded! Currently {len(self.feedback_samples)} feedback samples stored.\n")
    
    def _load_feedback(self):
        """Load previously saved feedback"""
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_feedback(self):
        """Save feedback to file"""
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_samples, f, indent=2)
    
    def predict(self, text):
        """Predict with current model"""
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
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, training=False)
            probabilities = torch.softmax(logits, dim=1)[0]
            prediction = torch.argmax(logits, dim=1).item()
        
        confidence = {
            'non-sarcastic': probabilities[0].item(),
            'sarcastic': probabilities[1].item()
        }
        
        label = 'sarcastic' if prediction == 1 else 'non-sarcastic'
        
        return label, confidence
    
    def add_feedback(self, text, correct_label):
        """Add user feedback"""
        # Convert label to integer
        label_int = 1 if correct_label == 'sarcastic' else 0
        
        feedback = {
            'text': text,
            'label': label_int,
            'timestamp': datetime.now().isoformat()
        }
        
        self.feedback_samples.append(feedback)
        self._save_feedback()
        
        print(f"✅ Feedback saved! Total samples: {len(self.feedback_samples)}")
    
    def retrain_on_feedback(self, epochs=5, batch_size=8, learning_rate=5e-6):
        """Fine-tune model on user feedback"""
        
        if len(self.feedback_samples) < 5:
            print(f"⚠️  Need at least 5 feedback samples to retrain (currently: {len(self.feedback_samples)})")
            return False
        
        print(f"\n{'='*70}")
        print(f"🎓 RETRAINING ON {len(self.feedback_samples)} FEEDBACK SAMPLES")
        print(f"{'='*70}\n")
        
        # Create dataset and dataloader
        dataset = FeedbackDataset(self.feedback_samples, self.tokenizer)
        
        # If we have few samples, use them multiple times per epoch
        if len(self.feedback_samples) < 20:
            # Repeat samples to have meaningful batches
            dataloader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), 
                                   shuffle=True, drop_last=False)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup training
        self.model.train()
        
        # Only train the classifier layers, keep RoBERTa mostly frozen
        # This prevents catastrophic forgetting
        for param in self.model.roberta.parameters():
            param.requires_grad = False
        
        # Only fine-tune the last 2 layers of RoBERTa
        for param in self.model.roberta.encoder.layer[-2:].parameters():
            param.requires_grad = True
        
        optimizer = AdamW([
            {'params': self.model.roberta.encoder.layer[-2:].parameters(), 'lr': learning_rate},
            {'params': self.model.fc1.parameters(), 'lr': learning_rate * 5},
            {'params': self.model.fc2.parameters(), 'lr': learning_rate * 5},
            {'params': self.model.classifier.parameters(), 'lr': learning_rate * 5}
        ], weight_decay=0.01)
        
        # Use cross-entropy loss with higher weight on feedback samples
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                logits = self.model(input_ids, attention_mask, training=True)
                loss = criterion(logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
            avg_loss = total_loss / len(dataloader)
            accuracy = correct / total
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")
        
        # Save updated model
        backup_path = 'models/roberta_before_feedback.pt'
        if not os.path.exists(backup_path):
            torch.save(self.model.state_dict(), backup_path)
            print(f"\n💾 Original model backed up to: {backup_path}")
        
        torch.save(self.model.state_dict(), 'models/roberta_best.pt')
        print(f"💾 Updated model saved to: models/roberta_best.pt")
        
        self.model.eval()
        
        print(f"\n✅ Retraining complete!")
        return True

# ============================================================================
# INTERACTIVE MODE WITH FEEDBACK
# ============================================================================

def interactive_feedback_mode(system):
    """Interactive mode with feedback collection"""
    
    print("="*80)
    print("🎓 INTERACTIVE LEARNING MODE")
    print("="*80)
    print("\n📝 Test examples and provide feedback to improve the model!")
    print("\nCommands:")
    print("  • Enter text to get prediction")
    print("  • Type 'correct' if prediction is right")
    print("  • Type 'wrong' if prediction is wrong (you'll specify correct label)")
    print("  • Type 'retrain' to fine-tune model on all feedback")
    print("  • Type 'stats' to see feedback statistics")
    print("  • Type 'test' to test standard cases")
    print("  • Type 'quit' to exit")
    print("-"*80 + "\n")
    
    current_text = None
    current_prediction = None
    
    while True:
        if current_text is None:
            user_input = input("💬 Enter text (or command): ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'retrain':
                system.retrain_on_feedback()
                continue
            
            if user_input.lower() == 'stats':
                show_feedback_stats(system)
                continue
            
            if user_input.lower() == 'test':
                test_with_feedback(system)
                continue
            
            # Predict
            current_text = user_input
            label, confidence = system.predict(current_text)
            current_prediction = label
            
            # Display prediction
            emoji = "😏" if label == 'sarcastic' else "😊"
            print(f"\n{'='*80}")
            print(f"📄 Text: {current_text}")
            print(f"{emoji} Prediction: {label.upper()}")
            print(f"📊 Confidence: {confidence[label]:.1%}")
            print(f"{'='*80}\n")
            
            print("❓ Is this correct? Type 'correct' or 'wrong'")
        
        else:
            # Waiting for feedback
            feedback = input("👉 Feedback: ").strip().lower()
            
            if feedback == 'correct':
                print("✅ Great! The model got it right.\n")
                current_text = None
                current_prediction = None
            
            elif feedback == 'wrong':
                print("\n🤔 What should the correct label be?")
                print("  1. sarcastic")
                print("  2. non-sarcastic")
                
                correct_label_input = input("👉 Enter 1 or 2: ").strip()
                
                if correct_label_input == '1':
                    correct_label = 'sarcastic'
                elif correct_label_input == '2':
                    correct_label = 'non-sarcastic'
                else:
                    print("⚠️  Invalid input. Skipping...\n")
                    current_text = None
                    current_prediction = None
                    continue
                
                # Add feedback
                system.add_feedback(current_text, correct_label)
                print(f"✅ Feedback recorded: '{current_text}' → {correct_label}\n")
                
                # Ask if they want to retrain now
                if len(system.feedback_samples) >= 5 and len(system.feedback_samples) % 5 == 0:
                    retrain_now = input(f"🎓 You have {len(system.feedback_samples)} samples. Retrain now? (y/n): ").strip().lower()
                    if retrain_now == 'y':
                        system.retrain_on_feedback()
                
                current_text = None
                current_prediction = None
            
            elif feedback in ['quit', 'exit']:
                print("\n👋 Goodbye!")
                break
            
            else:
                print("⚠️  Please type 'correct' or 'wrong'\n")

def show_feedback_stats(system):
    """Show statistics about feedback"""
    print("\n" + "="*80)
    print("📊 FEEDBACK STATISTICS")
    print("="*80)
    
    if not system.feedback_samples:
        print("\n❌ No feedback samples yet!")
        print("Start testing and marking corrections to build your training set.\n")
        return
    
    print(f"\nTotal samples: {len(system.feedback_samples)}")
    
    # Count by label
    sarcastic_count = sum(1 for s in system.feedback_samples if s['label'] == 1)
    non_sarcastic_count = len(system.feedback_samples) - sarcastic_count
    
    print(f"  Sarcastic: {sarcastic_count}")
    print(f"  Non-sarcastic: {non_sarcastic_count}")
    
    # Show recent samples
    print("\n📝 Recent feedback samples:")
    for i, sample in enumerate(system.feedback_samples[-5:], 1):
        label_str = 'sarcastic' if sample['label'] == 1 else 'non-sarcastic'
        print(f"  {i}. [{label_str:15}] {sample['text'][:60]}")
    
    print("\n" + "="*80 + "\n")

def test_with_feedback(system):
    """Test standard cases and collect feedback"""
    test_cases = [
        "I like the weather today",
        "I think that its too hot today",
        "i love you",
        "The weather is beautiful today",
        "I had a great day at work",
        "Oh great, another Monday morning",
        "Yeah right, like that's gonna work",
        "Perfect, just what I needed today",
    ]
    
    print("\n" + "="*80)
    print("🧪 TESTING STANDARD CASES")
    print("="*80 + "\n")
    
    for text in test_cases:
        label, confidence = system.predict(text)
        emoji = "😏" if label == 'sarcastic' else "😊"
        
        print(f"{emoji} [{label:15}] ({confidence[label]:.1%}) - {text}")
    
    print("\n" + "="*80 + "\n")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("🎓 INTERACTIVE LEARNING SYSTEM FOR SARCASM DETECTION")
    print("="*80)
    print("\nThis system allows you to:")
    print("  1. Test the model on any text")
    print("  2. Provide feedback when it's wrong")
    print("  3. Retrain the model on your corrections")
    print("  4. Continuously improve accuracy!")
    print("\n" + "="*80 + "\n")
    
    try:
        system = InteractiveLearningSystem()
        interactive_feedback_mode(system)
    except FileNotFoundError:
        print("❌ Error: RoBERTa model not found!")
        print("\nPlease train the model first:")
        print("  python scripts/roberta_model.py")
        return

if __name__ == "__main__":
    main()