import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (DistilBertTokenizer, DistilBertModel, 
                          AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

torch.manual_seed(42)
np.random.seed(42)

class ImprovedSarcasmDataset(Dataset):
    """Enhanced dataset with better sentiment extraction"""
    
    def __init__(self, texts, labels, sarcasm_tokenizer, sentiment_tokenizer, 
                 sentiment_model, device, max_length=128):
        self.texts = texts
        self.labels = labels
        self.sarcasm_tokenizer = sarcasm_tokenizer
        self.sentiment_tokenizer = sentiment_tokenizer
        self.max_length = max_length
        
        # Pre-compute sentiment embeddings using cardiffnlp twitter sentiment model
        print("Computing deep sentiment embeddings...")
        self.sentiment_embeddings = self._extract_sentiment_embeddings(sentiment_model, device)
        
        # Extract linguistic features
        print("Extracting linguistic features...")
        self.linguistic_features = self._extract_linguistic_features()
    
    def _extract_sentiment_embeddings(self, sentiment_model, device):
        """Extract sentiment using a fine-tuned sentiment model instead of VADER"""
        sentiment_model.eval()
        embeddings = []
        
        with torch.no_grad():
            for text in tqdm(self.texts, desc="Sentiment extraction"):
                text_str = str(text)
                
                # Tokenize for sentiment model
                inputs = self.sentiment_tokenizer(
                    text_str,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=True
                ).to(device)
                
                # Get sentiment logits
                outputs = sentiment_model(**inputs)
                sentiment_probs = F.softmax(outputs.logits, dim=-1)[0]
                
                # Extract: [negative, neutral, positive] probabilities
                embeddings.append(sentiment_probs.cpu().numpy())
        
        return torch.tensor(embeddings, dtype=torch.float32)
    
    def _extract_linguistic_features(self):
        """Extract enhanced linguistic features"""
        features = []
        
        for text in tqdm(self.texts, desc="Linguistic features"):
            text_str = str(text).lower()
            words = text_str.split()
            
            # 1. Punctuation intensity
            exclamation_ratio = text_str.count('!') / max(len(text_str), 1)
            question_ratio = text_str.count('?') / max(len(text_str), 1)
            ellipsis = 1 if ('...' in text_str or '…' in text_str) else 0
            multiple_punct = len(re.findall(r'[!?]{2,}', text_str))
            
            # 2. Capitalization patterns
            all_caps_words = sum(1 for w in text_str.split() if w.isupper() and len(w) > 1)
            caps_ratio = all_caps_words / max(len(words), 1)
            
            # 3. Emoji analysis (common sarcasm indicators)
            emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                "]+", flags=re.UNICODE)
            emoji_count = len(emoji_pattern.findall(text_str))
            
            # 4. Interjections and markers
            interjections = ['yeah', 'right', 'sure', 'wow', 'great', 'cool', 'nice', 'perfect']
            interjection_count = sum(1 for word in interjections if word in text_str)
            
            # 5. Intensifiers
            intensifiers = ['so', 'very', 'really', 'totally', 'absolutely', 'completely']
            intensifier_count = sum(text_str.count(word) for word in intensifiers)
            
            # 6. Negation patterns
            negations = ['not', "n't", 'no', 'never', 'nothing', 'nobody']
            negation_count = sum(text_str.count(word) for word in negations)
            
            # 7. Contrast markers
            contrasts = ['but', 'however', 'though', 'although', 'yet']
            contrast_count = sum(text_str.count(word) for word in contrasts)
            
            # 8. Quote marks (often used sarcastically)
            quote_marks = text_str.count('"') + text_str.count("'")
            
            feature_vector = [
                exclamation_ratio * 100,
                question_ratio * 100,
                ellipsis,
                multiple_punct,
                caps_ratio * 100,
                emoji_count,
                interjection_count,
                intensifier_count,
                negation_count,
                contrast_count,
                quote_marks,
                len(words)  # text length
            ]
            
            features.append(feature_vector)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize for sarcasm model
        encoding = self.sarcasm_tokenizer.encode_plus(
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
            'sentiment_embedding': self.sentiment_embeddings[idx],
            'linguistic_features': self.linguistic_features[idx],
            'label': torch.tensor(label, dtype=torch.long)
        }

class CrossAttention(nn.Module):
    """Cross-attention between contextual and sentiment representations"""
    
    def __init__(self, hidden_size):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = np.sqrt(hidden_size)
    
    def forward(self, context, sentiment):
        Q = self.query(context)
        K = self.key(sentiment)
        V = self.value(sentiment)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended = torch.matmul(attention_weights, V)
        
        return attended, attention_weights

class ImprovedIncongruityModel(nn.Module):
    """
    Enhanced Sentiment Incongruity Model v2
    
    Key improvements:
    1. Uses deep sentiment model (cardiffnlp) instead of VADER
    2. Cross-attention between context and sentiment
    3. Richer linguistic features
    4. Contrastive incongruity detection
    """
    
    def __init__(self, num_classes=2, dropout=0.3):
        super(ImprovedIncongruityModel, self).__init__()
        
        # Main contextual encoder
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        bert_hidden_size = 768
        
        # Sentiment representation encoder
        # Input: 3D sentiment probs + 12 linguistic features = 15 features
        self.sentiment_encoder = nn.Sequential(
            nn.Linear(15, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, bert_hidden_size)  # Project to same dim as BERT
        )
        
        # Cross-attention: How does context attend to sentiment?
        self.cross_attention = CrossAttention(bert_hidden_size)
        
        # Incongruity detector - detects mismatch between signals
        self.incongruity_detector = nn.Sequential(
            nn.Linear(bert_hidden_size * 3, 512),  # context + sentiment + attended
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        
        # Gating mechanism: learn when to trust incongruity signal
        self.gate = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden_size + 256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, sentiment_embedding, linguistic_features):
        # 1. Extract contextual representation
        bert_outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        context_repr = bert_outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # 2. Encode sentiment + linguistic features
        combined_sentiment = torch.cat([sentiment_embedding, linguistic_features], dim=1)
        sentiment_repr = self.sentiment_encoder(combined_sentiment)
        
        # 3. Cross-attention: how does context relate to sentiment?
        attended_sentiment, attention_weights = self.cross_attention(
            context_repr.unsqueeze(1), 
            sentiment_repr.unsqueeze(1)
        )
        attended_sentiment = attended_sentiment.squeeze(1)
        
        # 4. Detect incongruity
        incongruity_input = torch.cat([context_repr, sentiment_repr, attended_sentiment], dim=1)
        incongruity_signal = self.incongruity_detector(incongruity_input)
        
        # 5. Gate the incongruity signal
        gate_weight = self.gate(incongruity_signal)
        gated_incongruity = gate_weight * incongruity_signal
        
        # 6. Final classification
        final_repr = torch.cat([context_repr, gated_incongruity], dim=1)
        logits = self.classifier(final_repr)
        
        return logits, incongruity_signal, gate_weight

class ContrastiveLoss(nn.Module):
    """Contrastive loss to maximize separation between sarcastic and non-sarcastic incongruity"""
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, incongruity_signal, labels):
        # Split into sarcastic and non-sarcastic
        sarcastic_mask = labels == 1
        non_sarcastic_mask = labels == 0
        
        if sarcastic_mask.sum() == 0 or non_sarcastic_mask.sum() == 0:
            return torch.tensor(0.0, device=incongruity_signal.device)
        
        # Compute mean incongruity for each class
        sarcastic_incong = incongruity_signal[sarcastic_mask].mean(dim=0)
        non_sarcastic_incong = incongruity_signal[non_sarcastic_mask].mean(dim=0)
        
        # Distance between class centroids
        distance = F.pairwise_distance(
            sarcastic_incong.unsqueeze(0), 
            non_sarcastic_incong.unsqueeze(0)
        )
        
        # Maximize distance (minimize negative distance)
        loss = F.relu(self.margin - distance)
        
        return loss

class ImprovedTrainer:
    """Trainer with contrastive learning"""
    
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=2e-5, epochs=5, contrastive_weight=0.1):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.contrastive_weight = contrastive_weight
        
        # Two loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = ContrastiveLoss(margin=1.0)
        
        # Optimizer with weight decay
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Scheduler
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
        total_cls_loss = 0
        total_cont_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            sentiment_emb = batch['sentiment_embedding'].to(self.device)
            linguistic_feat = batch['linguistic_features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            logits, incongruity, _ = self.model(input_ids, attention_mask, 
                                               sentiment_emb, linguistic_feat)
            
            # Classification loss
            cls_loss = self.classification_loss(logits, labels)
            
            # Contrastive loss on incongruity signal
            cont_loss = self.contrastive_loss(incongruity, labels)
            
            # Combined loss
            loss = cls_loss + self.contrastive_weight * cont_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_cont_loss += cont_loss.item()
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'cls': cls_loss.item(),
                'cont': cont_loss.item()
            })
        
        avg_loss = total_loss / len(self.train_loader)
        print(f"  Avg Classification Loss: {total_cls_loss/len(self.train_loader):.4f}")
        print(f"  Avg Contrastive Loss: {total_cont_loss/len(self.train_loader):.4f}")
        
        return avg_loss
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                sentiment_emb = batch['sentiment_embedding'].to(self.device)
                linguistic_feat = batch['linguistic_features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits, _, _ = self.model(input_ids, attention_mask, 
                                         sentiment_emb, linguistic_feat)
                loss = self.classification_loss(logits, labels)
                
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
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
        print("Starting training with improved incongruity detection + contrastive learning...")
        best_f1 = 0
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print("-" * 60)
            
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
                torch.save(self.model.state_dict(), 'models/improved_incongruity_best.pt')
                print(f"✓ Saved new best model (F1: {best_f1:.4f})")
        
        return val_metrics
    
    def plot_training_history(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        epochs_range = range(1, len(self.train_losses) + 1)
        
        axes[0].plot(epochs_range, self.train_losses, label='Train', marker='o')
        axes[0].plot(epochs_range, self.val_losses, label='Val', marker='o')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curves')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(epochs_range, self.val_accuracies, marker='o', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].grid(True)
        
        axes[2].plot(epochs_range, self.val_f1_scores, marker='o', color='purple')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1 Score')
        axes[2].set_title('Validation F1')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/improved_training_history.png', dpi=150, bbox_inches='tight')
        plt.show()

def plot_confusion_matrix(true_labels, predictions, title="Confusion Matrix"):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Sarcastic', 'Sarcastic'],
                yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(f'results/{title.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    BATCH_SIZE = 16
    MAX_LENGTH = 128
    LEARNING_RATE = 2e-5
    EPOCHS = 5
    CONTRASTIVE_WEIGHT = 0.1
    
    print("="*70)
    print("IMPROVED SENTIMENT INCONGRUITY MODEL V2")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Initialize tokenizers
    print("\nInitializing tokenizers...")
    sarcasm_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Load sentiment model (Twitter-tuned RoBERTa for better sentiment)
    print("Loading sentiment model (cardiffnlp/twitter-roberta-base-sentiment)...")
    sentiment_tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        'cardiffnlp/twitter-roberta-base-sentiment-latest'
    ).to(device)
    
    # Create datasets
    print("\nCreating enhanced datasets...")
    train_dataset = ImprovedSarcasmDataset(
        train_df['text'].values,
        train_df['label'].values,
        sarcasm_tokenizer,
        sentiment_tokenizer,
        sentiment_model,
        device,
        MAX_LENGTH
    )
    
    val_dataset = ImprovedSarcasmDataset(
        val_df['text'].values,
        val_df['label'].values,
        sarcasm_tokenizer,
        sentiment_tokenizer,
        sentiment_model,
        device,
        MAX_LENGTH
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    print("\nInitializing Improved Incongruity Model...")
    model = ImprovedIncongruityModel()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train
    trainer = ImprovedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        contrastive_weight=CONTRASTIVE_WEIGHT
    )
    
    final_metrics = trainer.train()
    
    # Plot results
    print("\nGenerating visualizations...")
    trainer.plot_training_history()
    plot_confusion_matrix(final_metrics['true_labels'], final_metrics['predictions'],
                         "Improved Model - Confusion Matrix")
    
    # Final report
    print("\n" + "="*70)
    print("FINAL RESULTS - IMPROVED MODEL")
    print("="*70)
    print(classification_report(
        final_metrics['true_labels'],
        final_metrics['predictions'],
        target_names=['Non-Sarcastic', 'Sarcastic']
    ))
    
    # Compare all models
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"Baseline (DistilBERT only):        F1 = 0.6700")
    print(f"Incongruity v1 (VADER):            F1 = 0.6894 (+2.89%)")
    print(f"Incongruity v2 (Improved):         F1 = {final_metrics['f1']:.4f} ({((final_metrics['f1'] - 0.67) / 0.67 * 100):+.2f}%)")
    
    print("\n✅ Training complete! Model saved to models/improved_incongruity_best.pt")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    main()                                  