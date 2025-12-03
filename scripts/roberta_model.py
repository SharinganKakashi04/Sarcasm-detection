import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# DATASET
# ============================================================================

class RobertaSarcasmDataset(Dataset):
    """Dataset for RoBERTa"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
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
            'label': torch.tensor(label, dtype=torch.long)
        }

# ============================================================================
# IMPROVED ROBERTA MODEL
# ============================================================================

class RobertaSarcasmClassifier(nn.Module):
    """
    RoBERTa-based classifier with improvements:
    - Better architecture
    - Multi-sample dropout for robustness
    - Layer normalization
    """
    
    def __init__(self, num_classes=2, dropout=0.3, hidden_size=256):
        super(RobertaSarcasmClassifier, self).__init__()
        
        # RoBERTa encoder
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        roberta_hidden = 768
        
        # Multi-layer classifier with residual-like connections
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(roberta_hidden, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln2 = nn.LayerNorm(hidden_size // 2)
        
        self.dropout3 = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size // 2, num_classes)
        
        # Multi-sample dropout (apply dropout multiple times for robustness)
        self.num_samples = 5
    
    def forward(self, input_ids, attention_mask, training=True):
        # Get RoBERTa outputs
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use CLS token
        pooled = outputs.last_hidden_state[:, 0]
        
        if training and self.training:
            # Multi-sample dropout during training
            logits_list = []
            for _ in range(self.num_samples):
                x = self.dropout1(pooled)
                x = F.relu(self.ln1(self.fc1(x)))
                
                x = self.dropout2(x)
                x = F.relu(self.ln2(self.fc2(x)))
                
                x = self.dropout3(x)
                logits = self.classifier(x)
                logits_list.append(logits)
            
            # Average logits
            logits = torch.mean(torch.stack(logits_list), dim=0)
        else:
            # Single forward pass during inference
            x = self.dropout1(pooled)
            x = F.relu(self.ln1(self.fc1(x)))
            
            x = self.dropout2(x)
            x = F.relu(self.ln2(self.fc2(x)))
            
            x = self.dropout3(x)
            logits = self.classifier(x)
        
        return logits

# ============================================================================
# LABEL SMOOTHING LOSS
# ============================================================================

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing to prevent overconfidence
    Helps with generalization
    """
    
    def __init__(self, num_classes=2, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
    
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_label = self.smoothing / (self.num_classes - 1)
        
        log_probs = F.log_softmax(pred, dim=1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(smooth_label)
            true_dist.scatter_(1, target.unsqueeze(1), confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))

# ============================================================================
# IMPROVED TRAINER
# ============================================================================

class ImprovedTrainer:
    """Trainer with best practices for better generalization"""
    
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=1e-5, epochs=5, warmup_ratio=0.1,
                 use_label_smoothing=True):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        
        # Label smoothing for better generalization
        if use_label_smoothing:
            self.criterion = LabelSmoothingLoss(smoothing=0.1)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Separate learning rates for RoBERTa and classifier
        roberta_params = list(model.roberta.parameters())
        classifier_params = list(model.fc1.parameters()) + \
                           list(model.fc2.parameters()) + \
                           list(model.classifier.parameters())
        
        self.optimizer = AdamW([
            {'params': roberta_params, 'lr': learning_rate},
            {'params': classifier_params, 'lr': learning_rate * 10}  # Higher LR for classifier
        ], weight_decay=0.01)
        
        # Scheduler with warmup
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Track metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        self.best_val_f1 = 0
        
        # Early stopping
        self.patience = 3
        self.patience_counter = 0
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            logits = self.model(input_ids, attention_mask, training=True)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(input_ids, attention_mask, training=False)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions,
            'true_labels': true_labels,
            'probabilities': np.array(all_probs)
        }
    
    def train(self):
        print("Starting training with RoBERTa...")
        
        for epoch in range(self.epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print('='*70)
            
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
            
            # Save best model
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                torch.save(self.model.state_dict(), 'models/roberta_best.pt')
                print(f"✓ Saved new best model (F1: {self.best_val_f1:.4f})")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                print(f"No improvement ({self.patience_counter}/{self.patience})")
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('models/roberta_best.pt'))
        final_metrics = self.evaluate()
        
        return final_metrics
    
    def plot_training_history(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        epochs_range = range(1, len(self.train_losses) + 1)
        
        # Loss
        axes[0].plot(epochs_range, self.train_losses, label='Train', marker='o')
        axes[0].plot(epochs_range, self.val_losses, label='Val', marker='o')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(epochs_range, self.val_accuracies, marker='o', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].grid(True)
        
        # F1
        axes[2].plot(epochs_range, self.val_f1_scores, marker='o', color='purple')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1 Score')
        axes[2].set_title('Validation F1 Score')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/roberta_training_history.png', dpi=150, bbox_inches='tight')
        plt.show()

def plot_confusion_matrix(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu',
                xticklabels=['Non-Sarcastic', 'Sarcastic'],
                yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('RoBERTa Model - Confusion Matrix')
    plt.savefig('results/roberta_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

def main():
    BATCH_SIZE = 16
    MAX_LENGTH = 128
    LEARNING_RATE = 1e-5  # Lower LR for RoBERTa
    EPOCHS = 5
    
    print("="*70)
    print("ROBERTA-BASED SARCASM DETECTION")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')
    
    print(f"Train: {len(train_df)}")
    print(f"Val: {len(val_df)}")
    
    # Check class distribution
    train_counts = Counter(train_df['label'])
    print(f"\nTrain distribution: {train_counts}")
    print(f"Class balance ratio: {train_counts[0]/train_counts[1]:.2f}:1")
    
    # Initialize tokenizer
    print("\nLoading RoBERTa tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = RobertaSarcasmDataset(
        train_df['text'].values,
        train_df['label'].values,
        tokenizer,
        MAX_LENGTH
    )
    
    val_dataset = RobertaSarcasmDataset(
        val_df['text'].values,
        val_df['label'].values,
        tokenizer,
        MAX_LENGTH
    )
    
    # Create weighted sampler for class imbalance
    class_counts = [train_counts[0], train_counts[1]]
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = weights[train_df['label'].values]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    print("\nInitializing RoBERTa model...")
    model = RobertaSarcasmClassifier(dropout=0.3, hidden_size=256)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train
    trainer = ImprovedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        warmup_ratio=0.1,
        use_label_smoothing=True
    )
    
    final_metrics = trainer.train()
    
    # Plot results
    print("\nGenerating visualizations...")
    trainer.plot_training_history()
    plot_confusion_matrix(final_metrics['true_labels'], final_metrics['predictions'])
    
    # Final report
    print("\n" + "="*70)
    print("FINAL RESULTS - ROBERTA MODEL")
    print("="*70)
    print(classification_report(
        final_metrics['true_labels'],
        final_metrics['predictions'],
        target_names=['Non-Sarcastic', 'Sarcastic']
    ))
    
    # Complete comparison
    print("\n" + "="*70)
    print("COMPLETE MODEL COMPARISON")
    print("="*70)
    print(f"Baseline (DistilBERT):             F1 = 0.6700")
    print(f"Incongruity v1 (VADER):            F1 = 0.6894 (+2.89%)")
    print(f"Incongruity v2 (Deep Sentiment):   F1 = 0.6786 (+1.28%)")
    print(f"Ensemble + Augmentation:           F1 = 0.6984 (+4.24%)")
    print(f"RoBERTa (Improved):                F1 = {final_metrics['f1']:.4f} ({((final_metrics['f1'] - 0.67) / 0.67 * 100):+.2f}%)")
    
    improvement = final_metrics['f1'] - 0.67
    if final_metrics['f1'] > 0.72:
        print(f"\n🎉 EXCELLENT! Achieved {improvement:.4f} improvement ({improvement/0.67*100:.2f}%)")
    elif final_metrics['f1'] > 0.70:
        print(f"\n✅ GOOD! Achieved {improvement:.4f} improvement ({improvement/0.67*100:.2f}%)")
    else:
        print(f"\n⚠️  Moderate improvement of {improvement:.4f} ({improvement/0.67*100:.2f}%)")
    
    print("\n✅ Training complete! Model saved to models/roberta_best.pt")
    print("\nTest your model with:")
    print("  python scripts/detect_sarcasm.py --model models/roberta_best.pt")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    main()