import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SarcasmDataset(Dataset):
    """Dataset class for sarcasm detection"""
    
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

class DistilBERTClassifier(nn.Module):
    """DistilBERT-based sarcasm classifier"""
    
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
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

class BaselineTrainer:
    """Training and evaluation for baseline model"""
    
    def __init__(self, model, train_loader, val_loader, device, learning_rate=2e-5, epochs=3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        
        # Loss function with class weights to handle imbalance
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        total_steps = len(train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Track metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
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
            'true_labels': true_labels
        }
    
    def train(self):
        """Full training loop"""
        print("Starting training...")
        best_f1 = 0
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print("-" * 50)
            
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            val_metrics = self.evaluate()
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}")
            print(f"Val Recall: {val_metrics['recall']:.4f}")
            
            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save(self.model.state_dict(), 'models/baseline_best.pt')
                print(f"✓ Saved new best model (F1: {best_f1:.4f})")
        
        return val_metrics
    
    def plot_training_history(self):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        epochs_range = range(1, len(self.train_losses) + 1)
        axes[0].plot(epochs_range, self.train_losses, label='Train Loss', marker='o')
        axes[0].plot(epochs_range, self.val_losses, label='Val Loss', marker='o')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy curve
        axes[1].plot(epochs_range, self.val_accuracies, label='Val Accuracy', 
                    marker='o', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/baseline_training_history.png', dpi=150, bbox_inches='tight')
        plt.show()

def plot_confusion_matrix(true_labels, predictions):
    """Plot confusion matrix"""
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Sarcastic', 'Sarcastic'],
                yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Baseline Model')
    plt.savefig('results/baseline_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    # Configuration
    BATCH_SIZE = 16  # Small batch size for CPU
    MAX_LENGTH = 128
    LEARNING_RATE = 2e-5
    EPOCHS = 3
    
    print("="*50)
    print("BASELINE MODEL TRAINING")
    print("="*50)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SarcasmDataset(
        train_df['text'].values,
        train_df['label'].values,
        tokenizer,
        MAX_LENGTH
    )
    
    val_dataset = SarcasmDataset(
        val_df['text'].values,
        val_df['label'].values,
        tokenizer,
        MAX_LENGTH
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    print("\nInitializing model...")
    model = DistilBERTClassifier()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = BaselineTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS
    )
    
    # Train
    final_metrics = trainer.train()
    
    # Plot results
    print("\nGenerating plots...")
    trainer.plot_training_history()
    plot_confusion_matrix(final_metrics['true_labels'], final_metrics['predictions'])
    
    # Print final classification report
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print("\nClassification Report:")
    print(classification_report(
        final_metrics['true_labels'],
        final_metrics['predictions'],
        target_names=['Non-Sarcastic', 'Sarcastic']
    ))
    
    print("\n✅ Training complete! Model saved to models/baseline_best.pt")

if __name__ == "__main__":
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    main()