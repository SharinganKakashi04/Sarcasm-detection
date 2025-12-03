import torch
import torch.nn as nn
import torch.nn.functional as F
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
import random
from collections import Counter

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class SarcasmAugmenter:
    """Data augmentation for sarcasm detection"""
    
    def __init__(self):
        # Sarcasm markers to inject
        self.sarcasm_markers = [
            "Yeah right", "Sure", "Oh great", "How wonderful", 
            "Just perfect", "Absolutely fantastic", "Oh yeah"
        ]
        
        # Intensifiers to add
        self.intensifiers = [
            "so", "very", "really", "totally", "absolutely", "completely"
        ]
    
    def synonym_replacement(self, text):
        """Simple synonym replacement"""
        replacements = {
            'good': ['great', 'nice', 'fine', 'excellent'],
            'bad': ['terrible', 'awful', 'horrible', 'poor'],
            'happy': ['glad', 'pleased', 'delighted', 'joyful'],
            'sad': ['unhappy', 'miserable', 'disappointed', 'upset'],
        }
        
        words = text.split()
        new_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in replacements and random.random() < 0.3:
                new_words.append(random.choice(replacements[word_lower]))
            else:
                new_words.append(word)
        
        return ' '.join(new_words)
    
    def add_intensifier(self, text):
        """Add intensifiers to make sarcasm more pronounced"""
        words = text.split()
        adjectives_idx = []
        
        # Simple heuristic: words followed by nouns might be adjectives
        for i, word in enumerate(words):
            if i < len(words) - 1 and len(word) > 3:
                if random.random() < 0.3:
                    adjectives_idx.append(i)
        
        if adjectives_idx:
            idx = random.choice(adjectives_idx)
            intensifier = random.choice(self.intensifiers)
            words.insert(idx, intensifier)
        
        return ' '.join(words)
    
    def add_punctuation(self, text):
        """Add sarcastic punctuation"""
        if random.random() < 0.3:
            text = text.rstrip('.')
            punct = random.choice(['!', '!!', '...', '?!'])
            text += punct
        return text
    
    def augment(self, text, label):
        """Apply random augmentation"""
        if label == 0:  # Non-sarcastic - less augmentation
            if random.random() < 0.3:
                text = self.synonym_replacement(text)
        else:  # Sarcastic - more aggressive augmentation
            aug_type = random.choice(['synonym', 'intensifier', 'punctuation', 'combined'])
            
            if aug_type == 'synonym':
                text = self.synonym_replacement(text)
            elif aug_type == 'intensifier':
                text = self.add_intensifier(text)
            elif aug_type == 'punctuation':
                text = self.add_punctuation(text)
            else:  # combined
                text = self.synonym_replacement(text)
                text = self.add_intensifier(text)
                text = self.add_punctuation(text)
        
        return text

def augment_dataset(df, augmenter, target_samples=None):
    """Augment dataset with balanced samples"""
    print("Augmenting dataset...")
    
    # Separate by class
    sarcastic = df[df['label'] == 1]
    non_sarcastic = df[df['label'] == 0]
    
    # Calculate how many augmented samples we need
    if target_samples is None:
        target_samples = max(len(sarcastic), len(non_sarcastic))
    
    augmented_data = []
    
    # Augment sarcastic samples
    sarc_needed = max(0, target_samples - len(sarcastic))
    for _ in tqdm(range(sarc_needed), desc="Augmenting sarcastic"):
        sample = sarcastic.sample(1).iloc[0]
        aug_text = augmenter.augment(sample['text'], 1)
        augmented_data.append({'text': aug_text, 'label': 1})
    
    # Augment non-sarcastic samples
    non_sarc_needed = max(0, target_samples - len(non_sarcastic))
    for _ in tqdm(range(non_sarc_needed), desc="Augmenting non-sarcastic"):
        sample = non_sarcastic.sample(1).iloc[0]
        aug_text = augmenter.augment(sample['text'], 0)
        augmented_data.append({'text': aug_text, 'label': 0})
    
    # Combine original + augmented
    augmented_df = pd.DataFrame(augmented_data)
    combined_df = pd.concat([df, augmented_df], ignore_index=True)
    
    print(f"Original: {len(df)}, Augmented: {len(augmented_df)}, Total: {len(combined_df)}")
    print(f"Class distribution: {Counter(combined_df['label'])}")
    
    return combined_df

# ============================================================================
# IMPROVED DATASET
# ============================================================================

class AugmentedSarcasmDataset(Dataset):
    """Dataset for augmented training"""
    
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
# ENHANCED BASELINE MODEL WITH FOCAL LOSS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss to handle class imbalance better"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class EnhancedDistilBERT(nn.Module):
    """Enhanced DistilBERT with better architecture"""
    
    def __init__(self, num_classes=2, dropout=0.4):
        super(EnhancedDistilBERT, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        hidden_size = 768
        
        # Multi-layer classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        return logits

# ============================================================================
# ENSEMBLE TRAINER
# ============================================================================

class EnsembleTrainer:
    """Train multiple models and ensemble them"""
    
    def __init__(self, models, train_loader, val_loader, device, 
                 learning_rate=2e-5, epochs=4, use_focal_loss=True):
        self.models = [m.to(device) for m in models]
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        
        # Focal loss for better class handling
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Separate optimizer for each model
        self.optimizers = []
        self.schedulers = []
        
        for model in self.models:
            optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            self.optimizers.append(optimizer)
            
            total_steps = len(train_loader) * epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=total_steps // 10,
                num_training_steps=total_steps
            )
            self.schedulers.append(scheduler)
        
        self.train_losses = [[] for _ in models]
        self.val_accuracies = [[] for _ in models]
        self.ensemble_accuracies = []
    
    def train_epoch(self, model_idx):
        """Train one model for one epoch"""
        model = self.models[model_idx]
        optimizer = self.optimizers[model_idx]
        scheduler = self.schedulers[model_idx]
        
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Training Model {model_idx+1}')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def evaluate_single(self, model_idx):
        """Evaluate single model"""
        model = self.models[model_idx]
        model.eval()
        
        predictions = []
        true_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(input_ids, attention_mask)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        accuracy = accuracy_score(true_labels, predictions)
        return accuracy, predictions, true_labels, np.vstack(all_probs)
    
    def evaluate_ensemble(self):
        """Evaluate ensemble of all models"""
        for model in self.models:
            model.eval()
        
        all_probs_per_model = [[] for _ in self.models]
        true_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                for i, model in enumerate(self.models):
                    outputs = model(input_ids, attention_mask)
                    probs = F.softmax(outputs, dim=1)
                    all_probs_per_model[i].append(probs.cpu().numpy())
                
                true_labels.extend(labels.cpu().numpy())
        
        # Average predictions from all models
        ensemble_probs = []
        for i in range(len(self.models)):
            model_probs = np.vstack(all_probs_per_model[i])
            ensemble_probs.append(model_probs)
        
        ensemble_probs = np.mean(ensemble_probs, axis=0)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        accuracy = accuracy_score(true_labels, ensemble_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, ensemble_preds, average='binary'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': ensemble_preds,
            'true_labels': true_labels
        }
    
    def train(self):
        """Train all models"""
        print(f"Training ensemble of {len(self.models)} models...")
        
        for epoch in range(self.epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print('='*70)
            
            # Train each model
            for i in range(len(self.models)):
                print(f"\n--- Model {i+1} ---")
                train_loss = self.train_epoch(i)
                self.train_losses[i].append(train_loss)
                
                acc, _, _, _ = self.evaluate_single(i)
                self.val_accuracies[i].append(acc)
                
                print(f"Model {i+1} - Train Loss: {train_loss:.4f}, Val Acc: {acc:.4f}")
            
            # Evaluate ensemble
            print(f"\n--- Ensemble Evaluation ---")
            ensemble_metrics = self.evaluate_ensemble()
            self.ensemble_accuracies.append(ensemble_metrics['accuracy'])
            
            print(f"Ensemble Accuracy: {ensemble_metrics['accuracy']:.4f}")
            print(f"Ensemble F1: {ensemble_metrics['f1']:.4f}")
            print(f"Ensemble Precision: {ensemble_metrics['precision']:.4f}")
            print(f"Ensemble Recall: {ensemble_metrics['recall']:.4f}")
        
        # Final evaluation
        final_metrics = self.evaluate_ensemble()
        
        # Save best models
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), f'models/ensemble_model_{i+1}.pt')
        
        return final_metrics
    
    def plot_results(self):
        """Plot ensemble training results"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        epochs = range(1, self.epochs + 1)
        
        # Individual model accuracies
        for i in range(len(self.models)):
            axes[0].plot(epochs, self.val_accuracies[i], 
                        marker='o', label=f'Model {i+1}', alpha=0.6)
        axes[0].plot(epochs, self.ensemble_accuracies, 
                    marker='s', linewidth=3, label='Ensemble', color='red')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Performance Comparison')
        axes[0].legend()
        axes[0].grid(True)
        
        # Training losses
        for i in range(len(self.models)):
            axes[1].plot(epochs, self.train_losses[i], 
                        marker='o', label=f'Model {i+1}', alpha=0.6)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/ensemble_training.png', dpi=150, bbox_inches='tight')
        plt.show()

def plot_confusion_matrix(true_labels, predictions, title):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=['Non-Sarcastic', 'Sarcastic'],
                yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(f'results/{title.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

def main():
    BATCH_SIZE = 16
    MAX_LENGTH = 128
    LEARNING_RATE = 2e-5
    EPOCHS = 4
    NUM_MODELS = 3  # Ensemble of 3 models
    
    print("="*70)
    print("DATA AUGMENTATION + ENSEMBLE APPROACH")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')
    
    print(f"Original train: {len(train_df)}")
    print(f"Original val: {len(val_df)}")
    print(f"Original class distribution: {Counter(train_df['label'])}")
    
    # Augment training data
    augmenter = SarcasmAugmenter()
    train_df_augmented = augment_dataset(train_df, augmenter, target_samples=2000)
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Create datasets
    train_dataset = AugmentedSarcasmDataset(
        train_df_augmented['text'].values,
        train_df_augmented['label'].values,
        tokenizer,
        MAX_LENGTH
    )
    
    val_dataset = AugmentedSarcasmDataset(
        val_df['text'].values,
        val_df['label'].values,
        tokenizer,
        MAX_LENGTH
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Create ensemble of models with different initializations
    print(f"\nInitializing ensemble of {NUM_MODELS} models...")
    models = []
    for i in range(NUM_MODELS):
        torch.manual_seed(42 + i)  # Different seed for each model
        model = EnhancedDistilBERT(dropout=0.3 + i*0.05)  # Slightly different dropout
        models.append(model)
        print(f"Model {i+1} initialized")
    
    # Train ensemble
    trainer = EnsembleTrainer(
        models=models,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        use_focal_loss=True
    )
    
    final_metrics = trainer.train()
    
    # Plot results
    print("\nGenerating visualizations...")
    trainer.plot_results()
    plot_confusion_matrix(final_metrics['true_labels'], final_metrics['predictions'],
                         "Ensemble Model - Confusion Matrix")
    
    # Final report
    print("\n" + "="*70)
    print("FINAL RESULTS - ENSEMBLE MODEL")
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
    print(f"Ensemble + Augmentation:           F1 = {final_metrics['f1']:.4f} ({((final_metrics['f1'] - 0.67) / 0.67 * 100):+.2f}%)")
    
    improvement = final_metrics['f1'] - 0.67
    if improvement > 0.03:
        print(f"\n✅ SUCCESS! Achieved {improvement:.4f} improvement ({improvement/0.67*100:.2f}%)")
    else:
        print(f"\n⚠️  Modest improvement of {improvement:.4f} ({improvement/0.67*100:.2f}%)")
        print("Consider pivoting to analysis paper focusing on interpretability")
    
    print("\n✅ Training complete! Models saved to models/ensemble_model_*.pt")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    main()