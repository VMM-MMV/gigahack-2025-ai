import os
import torch
import logging
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback  
)

import numpy as np
from seqeval.metrics import classification_report


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bert_ner_trainer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration"""
    model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 512
    batch_size: int = 1000
    num_epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    output_dir: str = "./bert_ner_model"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100

class CoNLLDataset(Dataset):
    """Dataset class for CoNLL-2003 format"""
    
    def __init__(self, file_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentences, self.labels = self._read_conll_file(file_path)
        self.label_to_id, self.id_to_label = self._create_label_mappings()
        
    def _read_conll_file(self, file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
        """Read CoNLL format file"""
        sentences = []
        labels = []
        current_sentence = []
        current_labels = []
        
        logger.info(f"Reading CoNLL file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line == "" or line.startswith("-DOCSTART-"):
                    if current_sentence:
                        sentences.append(current_sentence)
                        labels.append(current_labels)
                        current_sentence = []
                        current_labels = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        word = parts[0]
                        label = parts[-1]  # Last column is the label
                        current_sentence.append(word)
                        current_labels.append(label)
        
        # Add last sentence if exists
        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)
            
        logger.info(f"Loaded {len(sentences)} sentences")
        return sentences, labels
    
    def _create_label_mappings(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Create label to ID mappings"""
        unique_labels = set()
        for label_seq in self.labels:
            unique_labels.update(label_seq)
        
        # Sort labels for consistency
        sorted_labels = sorted(list(unique_labels))
        label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}
        
        logger.info(f"Found {len(unique_labels)} unique labels: {sorted_labels}")
        return label_to_id, id_to_label
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        labels = self.labels[idx]
        
        # Tokenize and align labels
        encoding = self.tokenizer(
            sentence,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Align labels with tokenized input
        labels_aligned = self._align_labels(labels, encoding)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels_aligned, dtype=torch.long)
        }
    
    def _align_labels(self, labels: List[str], encoding) -> List[int]:
        """Align labels with tokenized input"""
        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100 (ignored in loss)
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # First subtoken of a word gets the label
                aligned_labels.append(self.label_to_id[labels[word_idx]])
            else:
                # Subsequent subtokens get -100 (ignored in loss)
                aligned_labels.append(-100)
            previous_word_idx = word_idx
            
        return aligned_labels

class BERTNERTrainer:
    """Main trainer class for BERT NER"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None
        
    def prepare_data(self, train_file: str, eval_file: Optional[str] = None):
        """Prepare training and evaluation datasets"""
        logger.info("Preparing data...")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Load datasets
        self.train_dataset = CoNLLDataset(train_file, self.tokenizer, self.config.max_length)
        
        if eval_file:
            self.eval_dataset = CoNLLDataset(eval_file, self.tokenizer, self.config.max_length)
        else:
            # Split train dataset for evaluation
            train_size = int(0.8 * len(self.train_dataset))
            eval_size = len(self.train_dataset) - train_size
            self.train_dataset, self.eval_dataset = torch.utils.data.random_split(
                self.train_dataset, [train_size, eval_size]
            )
        
        # Save label mappings
        self._save_label_mappings()
        
    def _save_label_mappings(self):
        """Save label mappings for inference"""
        mappings = {
            'label_to_id': self.train_dataset.label_to_id if hasattr(self.train_dataset, 'label_to_id') else self.train_dataset.dataset.label_to_id,
            'id_to_label': self.train_dataset.id_to_label if hasattr(self.train_dataset, 'id_to_label') else self.train_dataset.dataset.id_to_label
        }
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        with open(os.path.join(self.config.output_dir, 'label_mappings.json'), 'w') as f:
            json.dump(mappings, f, indent=2)
        
        logger.info(f"Saved label mappings to {self.config.output_dir}/label_mappings.json")
    
    def setup_model(self):
        """Setup the BERT model"""
        logger.info(f"Setting up model: {self.config.model_name}")
        
        # Get number of labels
        if hasattr(self.train_dataset, 'label_to_id'):
            num_labels = len(self.train_dataset.label_to_id)
        else:
            num_labels = len(self.train_dataset.dataset.label_to_id)
        
        # Load model
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config.model_name,
            num_labels=num_labels
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("Model moved to GPU")
        else:
            logger.info("Using CPU")
    

    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation using seqeval"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        # Load id_to_label mapping
        # This logic handles if the dataset is a Subset or the full Dataset
        id_to_label = self.train_dataset.dataset.id_to_label if hasattr(self.train_dataset, 'dataset') else self.train_dataset.id_to_label

        # Remove ignored index (special tokens) and convert IDs to labels
        true_predictions = [
            [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Use seqeval's classification_report
        report = classification_report(true_labels, true_predictions, output_dict=True)
        
        # Return main metrics
        return {
            "precision": report["micro avg"]["precision"],
            "recall": report["micro avg"]["recall"],
            "f1": report["micro avg"]["f1-score"],
        }
    
    # Inside the BERTNERTrainer class

    def train(self):
        """Train the model"""
        logger.info("Starting training...")
        
        # Training arguments are already well-configured for this
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs, # This now acts as a maximum
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir='./logs',
            logging_steps=self.config.logging_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            load_best_model_at_end=True,      # ✅ Crucial for early stopping
            metric_for_best_model="f1",       # ✅ Tells it which metric to watch
            greater_is_better=True,           # ✅ Tells it that a higher f1 is better
            save_total_limit=2,
            dataloader_pin_memory=False,
        )
        
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        
        # Initialize trainer WITH THE CALLBACK
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            # 👇 ADD THIS LINE 👇
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
        )
        
        # Train
        self.trainer.train()
        
        # Save final model (which will be the best one)
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info(f"Training completed. Model saved to {self.config.output_dir}")
    
    def evaluate(self):
        """Evaluate the trained model"""
        if self.trainer is None:
            logger.error("Model not trained yet. Call train() first.")
            return
        
        logger.info("Evaluating model...")
        results = self.trainer.evaluate()
        
        logger.info("Evaluation Results:")
        for key, value in results.items():
            logger.info(f"{key}: {value:.4f}")
        
        return results

class BERTNERInference:
    """Inference class for trained BERT NER model"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        
        # Load label mappings
        with open(os.path.join(model_path, 'label_mappings.json'), 'r') as f:
            mappings = json.load(f)
        self.label_to_id = mappings['label_to_id']
        self.id_to_label = {int(k): v for k, v in mappings['id_to_label'].items()}
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.model.eval()
        logger.info(f"Loaded model from {model_path}")
    
    def predict(self, text: str) -> List[Tuple[str, str]]:
        """Predict NER tags for input text"""
        # Tokenize (returns a BatchEncoding object)
        encoding = self.tokenizer(
            text.split(),
            is_split_into_words=True,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )

        # Move tensors to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in encoding.items()}
        else:
            inputs = dict(encoding)

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)

        # Align predictions with words
        word_ids = encoding.word_ids()   # <- keep using encoding here
        words = text.split()
        results = []

        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                if word_idx < len(words):
                    label = self.id_to_label[predictions[0][i].item()]
                    results.append((words[word_idx], label))
            previous_word_idx = word_idx

        return results

    
    def tokenize_sensitive_info(self, text: str) -> str:
        """Replace sensitive information with generic tokens"""
        predictions = self.predict(text)
        
        masked_words = []
        current_entity_words = []
        current_entity_type = None
        entity_counters = {}

        for word, label in predictions:
            if label.startswith('B-'):
                # If we were in the middle of another entity, save it first
                if current_entity_words:
                    entity_type_key = current_entity_type.split('-')[-1]
                    if entity_type_key not in entity_counters:
                        entity_counters[entity_type_key] = 0
                    entity_counters[entity_type_key] += 1
                    masked_words.append(f"[{entity_type_key}_{entity_counters[entity_type_key]}]")
                    current_entity_words = []

                # Start a new entity
                current_entity_type = label
                current_entity_words.append(word)

            elif label.startswith('I-') and current_entity_type and label.split('-')[-1] == current_entity_type.split('-')[-1]:
                # Continue the current entity
                current_entity_words.append(word)

            else: # Label is 'O' or a new B- tag
                # End of any current entity, so save it
                if current_entity_words:
                    entity_type_key = current_entity_type.split('-')[-1]
                    if entity_type_key not in entity_counters:
                        entity_counters[entity_type_key] = 0
                    entity_counters[entity_type_key] += 1
                    masked_words.append(f"[{entity_type_key}_{entity_counters[entity_type_key]}]")
                
                # Reset and add the current 'O' word
                current_entity_words = []
                current_entity_type = None
                masked_words.append(word)

        # Append any leftover entity at the end of the sentence
        if current_entity_words:
            entity_type_key = current_entity_type.split('-')[-1]
            if entity_type_key not in entity_counters:
                entity_counters[entity_type_key] = 0
            entity_counters[entity_type_key] += 1
            masked_words.append(f"[{entity_type_key}_{entity_counters[entity_type_key]}]")

        return ' '.join(masked_words)

def main():
    """Main training function"""
    config = TrainingConfig(
        model_name="answerdotai/ModernBERT-base",
        batch_size=16,
        num_epochs=10,  
        learning_rate=3e-5,  
        output_dir="./bert_ner_model",
        warmup_steps=200,
        save_steps=100,
        eval_steps=100,
        logging_steps=50
    )
    
    trainer = BERTNERTrainer(config)
    train_file = "data/ner_dataset_conll.txt"
    
    try:
        trainer.prepare_data(train_file)
        trainer.setup_model()
        trainer.train()
        trainer.evaluate()
        
        logger.info("Training completed successfully!")
        
        # SKIP THE INFERENCE TEST FOR NOW
        print("✅ Training completed! Model saved to:", config.output_dir)
        print("🧪 Run inference separately after training")
        
        # Comment out or remove the inference test:
        inference = BERTNERInference(config.output_dir)
        sample_text = "John Doe lives at 123 Main Street"
        tokenized = inference.tokenize_sensitive_info(sample_text)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__=="__main__":
    main()