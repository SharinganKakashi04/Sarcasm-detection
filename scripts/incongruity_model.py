import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel, get_linear_schedule_with_warmup
# FIX: AdamW is now imported from torch.optim or transformers.optimization
from torch.optim import AdamW 
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# --- Global Seed for Reproducibility ---
torch.manual_seed(42)
np.random.seed(42)

# --- EnhancedSarcasmDataset Class (No Change) ---

class EnhancedSarcasmDataset(Dataset):
    """Dataset with both text and explicit sentiment features"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vader = SentimentIntensityAnalyzer()
        
        # Pre-compute linguistic features
        print("Computing linguistic features...")
        self.sentiment_features = self._extract_features()
    
    def _extract_features(self):
        """Extract explicit linguistic features for incongruity detection"""
        features = []
        
        for text in tqdm(self.texts, desc="Extracting features"):
            text_str = str(text)
            
            # 1. VADER sentiment scores (literal sentiment)
            vader_scores = self.vader.polarity_scores(text_str)
            
            # 2. Punctuation features
            exclamations = text_str.count('!')
            questions = text_str.count('?')
            ellipsis = 1 if ('...' in text_str or '…' in text_str) else 0
            
            # 3. Capitalization (intensity markers)
            words = text_str.split()
            caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / max(len(words), 1)
            
            # 4. Text statistics
            word_count = len(words)
            avg_word_length = np.mean([len(w) for w in words]) if words else 0
            
            # Combine into feature vector
            feature_vector = [
                vader_scores['compound'],    # Overall sentiment
                vader_scores['pos'],         # Positive sentiment
                vader_scores['neg'],         # Negative sentiment  
                vader_scores['neu'],         # Neutral sentiment
                exclamations / max(word_count, 1),  # Normalized exclamations
                questions / max(word_count, 1),     # Normalized questions
                ellipsis,
                caps_ratio,
                word_count,
                avg_word_length
            ]
            
            features.append(feature_vector)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
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
            'sentiment_features': self.sentiment_features[idx],
            'label': torch.tensor(label, dtype=torch.long)
        }

# --- IncongruityAttention Class (No Change) ---

class IncongruityAttention(nn.Module):
    """Attention mechanism to weight incongruity signals"""
    
    def __init__(self, hidden_size):
        super(IncongruityAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, hidden_size)
        attention_weights = torch.softmax(self.attention(x), dim=0)
        return attention_weights * x

# --- SentimentIncongruityModel Class (No Change) ---

class SentimentIncongruityModel(nn.Module):
    """
    Novel model that explicitly detects sentiment incongruity
    """
    
    def __init__(self, num_classes=2, dropout=0.3, feature_dim=10):
        super(SentimentIncongruityModel, self).__init__()
        
        # Contextual understanding branch (DistilBERT)
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        bert_hidden_size = self.distilbert.config.hidden_size  # 768
        
        # Sentiment feature branch (explicit features)
        self.sentiment_encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Incongruity detection module (THIS IS THE NOVEL PART)
        self.incongruity_detector = nn.Sequential(
            nn.Linear(bert_hidden_size + 256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.Tanh(),  # Tanh to capture bidirectional incongruity
            nn.Dropout(dropout)
        )
        
        # Attention to weight incongruity signals
        self.attention = IncongruityAttention(256)
        
        # Final classifier combining all signals
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden_size + 256 + 256, 256),  # BERT + sentiment + incongruity
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, sentiment_features):
        # 1. Extract contextual understanding from BERT
        bert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        contextual_embedding = bert_output.last_hidden_state[:, 0]  # [CLS] token
        
        # 2. Encode explicit sentiment features
        sentiment_embedding = self.sentiment_encoder(sentiment_features)
        
        # 3. Detect incongruity between literal and contextual meaning
        combined = torch.cat([contextual_embedding, sentiment_embedding], dim=1)
        incongruity_signal = self.incongruity_detector(combined)
        
        # 4. Apply attention to incongruity signals
        weighted_incongruity = self.attention(incongruity_signal)
        
        # 5. Final classification using all signals
        final_features = torch.cat([
            contextual_embedding,
            sentiment_embedding,
            weighted_incongruity
        ], dim=1)
        
        logits = self.classifier(final_features)
        
        return logits, weighted_incongruity  # Return incongruity for analysis

# --- IncongruityTrainer Class (No Change) ---

class IncongruityTrainer:
    """Training and evaluation for incongruity model"""
    
    def __init__(self, model, train_loader, val_loader, device, learning_rate=2e-5, epochs=4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        total_steps = len(train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            sentiment_features = batch['sentiment_features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            logits, _ = self.model(input_ids, attention_mask, sentiment_features)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self, return_incongruity=False):
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        incongruity_scores = [] if return_incongruity else None
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                sentiment_features = batch['sentiment_features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits, incongruity = self.model(input_ids, attention_mask, sentiment_features)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                
                if return_incongruity:
                    incongruity_scores.extend(incongruity.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        
        result = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions,
            'true_labels': true_labels
        }
        
        if return_incongruity:
            result['incongruity_scores'] = incongruity_scores
        
        return result
    
    def train(self):
        print("Starting training with incongruity detection...")
        best_f1 = 0
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print("-" * 50)
            
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            val_metrics = self.evaluate()
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            self.val_f1_scores.append(val_metrics['f1'])
            
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}")
            print(f"Val Recall: {val_metrics['recall']:.4f}")
            
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save(self.model.state_dict(), 'models/incongruity_best.pt')
                print(f"✓ Saved new best model (F1: {best_f1:.4f})")
        
        return val_metrics
    
    def plot_training_history(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        epochs_range = range(1, len(self.train_losses) + 1)
        
        # Loss
        axes[0].plot(epochs_range, self.train_losses, label='Train Loss', marker='o')
        axes[0].plot(epochs_range, self.val_losses, label='Val Loss', marker='o')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(epochs_range, self.val_accuracies, label='Val Accuracy', 
                    marker='o', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        # F1 Score
        axes[2].plot(epochs_range, self.val_f1_scores, label='Val F1', 
                    marker='o', color='purple')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1 Score')
        axes[2].set_title('Validation F1 Score')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/incongruity_training_history.png', dpi=150, bbox_inches='tight')
        plt.show()

def compare_models():
    """Compare baseline vs incongruity model"""
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    # This will be populated after training
    baseline_results = {
        'accuracy': 0.67,
        'f1': 0.67,
        'precision': 0.68,
        'recall': 0.67
    }
    
    print("\nBaseline Model (DistilBERT only):")
    print(f"  Accuracy:  {baseline_results['accuracy']:.4f}")
    print(f"  F1 Score:  {baseline_results['f1']:.4f}")
    print(f"  Precision: {baseline_results['precision']:.4f}")
    print(f"  Recall:    {baseline_results['recall']:.4f}")
    
    print("\n→ Now training Incongruity Model...")

def plot_confusion_matrix(true_labels, predictions, title="Confusion Matrix"):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Non-Sarcastic', 'Sarcastic'],
                yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(f'results/{title.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()

# --- MODIFIED main() FUNCTION ---

def main():
    BATCH_SIZE = 16
    MAX_LENGTH = 128
    LEARNING_RATE = 2e-5
    EPOCHS = 4  # One extra epoch for the enhanced model
    
    print("="*70)
    print("SENTIMENT INCONGRUITY MODEL")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    print("\nLoading data...")
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    print("\nInitializing tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    print("\nCreating enhanced datasets with sentiment features...")
    train_dataset = EnhancedSarcasmDataset(
        train_df['text'].values,
        train_df['label'].values,
        tokenizer,
        MAX_LENGTH
    )
    
    val_dataset = EnhancedSarcasmDataset(
        val_df['text'].values,
        val_df['label'].values,
        tokenizer,
        MAX_LENGTH
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    print("\nInitializing Sentiment Incongruity Model...")
    model = SentimentIncongruityModel(feature_dim=10)
    
    # --- Checkpoint Loading Logic ADDED HERE ---
    checkpoint_path = 'models/incongruity_best.pt'
    if os.path.exists(checkpoint_path):
        print(f"*** RESUMING: Loading saved best model weights from {checkpoint_path} ***")
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("Successfully loaded model state.")
        except RuntimeError as e:
            # This handles cases where the model architecture might change slightly
            print(f"Error loading state dict: {e}. Starting training from scratch.")
    else:
        print("No checkpoint found. Starting training from scratch.")
    # ----------------------------------------
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    compare_models()
    
    trainer = IncongruityTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS
    )
    
    final_metrics = trainer.train()
    
    print("\nGenerating plots...")
    trainer.plot_training_history()
    plot_confusion_matrix(final_metrics['true_labels'], final_metrics['predictions'],
                         "Incongruity Model - Confusion Matrix")
    
    print("\n" + "="*70)
    print("FINAL RESULTS - INCONGRUITY MODEL")
    print("="*70)
    print("\nClassification Report:")
    print(classification_report(
        final_metrics['true_labels'],
        final_metrics['predictions'],
        target_names=['Non-Sarcastic', 'Sarcastic']
    ))
    
    print("\n✅ Training complete! Model saved to models/incongruity_best.pt")
    
    # Compare with baseline
    print("\n" + "="*70)
    print("IMPROVEMENT OVER BASELINE")
    print("="*70)
    baseline_f1 = 0.67
    improvement = ((final_metrics['f1'] - baseline_f1) / baseline_f1) * 100
    print(f"Baseline F1:     {baseline_f1:.4f}")
    print(f"Incongruity F1:  {final_metrics['f1']:.4f}")
    print(f"Improvement:     {improvement:+.2f}%")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    main()