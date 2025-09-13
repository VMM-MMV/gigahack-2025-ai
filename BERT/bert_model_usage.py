from BERT_Trainer import BERTNERTrainer, BERTNERInference, TrainingConfig
import logging

def train_model():
    """Train the NER model"""
    
    # Configure training parameters
    config = TrainingConfig(
        model_name="answerdotai/ModernBERT-base",
        max_length=512,
        batch_size=1000,          # Adjust based on your GPU memory
        num_epochs=10,           # Increase for better performance
        learning_rate=2e-5,
        output_dir="./trained_ner_model",
        save_steps=500,
        eval_steps=500,
        logging_steps=100
    )
    
    # Initialize trainer
    trainer = BERTNERTrainer(config)
    
    # Your CoNLL training file path
    train_file = "data/ner_dataset_conll.txt"  # Replace with your file path
    eval_file = None  # Optional: separate evaluation file
    
    try:
        print("Starting NER model training...")
        
        # Prepare data
        trainer.prepare_data(train_file, eval_file)
        
        # Setup model
        trainer.setup_model()
        
        # Train
        trainer.train()
        
        # Evaluate
        results = trainer.evaluate()
        
        print("Training completed successfully!")
        print(f"Model saved to: {config.output_dir}")
        print(f"F1 Score: {results.get('eval_f1', 'N/A'):.4f}")
        
        return config.output_dir
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

def test_inference(model_path):
    """Test the trained model"""
    
    # Load trained model
    inference = BERTNERInference(model_path)
    
    # Test samples (replace with your actual test data)
    test_samples = [
        "John Smith lives at 456 Oak Avenue and his bank account is 1234567890",
        "Mary Johnson works at ABC Corporation on 789 Pine Street",
        "The client Robert Brown has account number 9876543210"
    ]
    
    print("\n" + "="*60)
    print("TESTING INFERENCE")
    print("="*60)
    
    for i, text in enumerate(test_samples, 1):
        print(f"\nTest {i}:")
        print(f"Original: {text}")
        
        # Get predictions
        predictions = inference.predict(text)
        print("Predictions:")
        for word, label in predictions:
            if label != 'O':
                print(f"  {word} -> {label}")
        
        # Get tokenized version
        tokenized = inference.tokenize_sensitive_info(text)
        print(f"Tokenized: {tokenized}")

def quick_inference_example():
    """Quick example for inference only (if model already trained)"""
    
    model_path = "./trained_ner_model"  # Path to your trained model
    
    try:
        inference = BERTNERInference(model_path)
        
        # Example text
        text = "John Doe lives at 123 Main Street with account 555-123-4567"
        
        # Tokenize sensitive info
        tokenized = inference.tokenize_sensitive_info(text)
        
        print(f"Original: {text}")
        print(f"Tokenized: {tokenized}")
        
    except Exception as e:
        print(f"Inference failed: {str(e)}")
        print("Make sure the model is trained first!")

if __name__ == "__main__":
    # For training
    print("Starting BERT NER Training...")
    model_path = train_model()
    
    # Test the trained model
    test_inference(model_path)
    
    print("\n" + "="*60)
    print("TRAINING AND TESTING COMPLETED!")
    print("="*60)

# ===================================
# DEPLOYMENT SCRIPT: deploy_ner.py
# ===================================

import os
import torch
from flask import Flask, request, jsonify
from BERT_Trainer import BERTNERInference

app = Flask(__name__)

# Load model once at startup
MODEL_PATH = "./trained_ner_model"
ner_model = None

def load_model():
    """Load the NER model"""
    global ner_model
    if os.path.exists(MODEL_PATH):
        ner_model = BERTNERInference(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": ner_model is not None})

@app.route('/tokenize', methods=['POST'])
def tokenize_text():
    """Tokenize sensitive information in text"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Tokenize
        tokenized = ner_model.tokenize_sensitive_info(text)
        
        return jsonify({
            "original": text,
            "tokenized": tokenized,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_entities():
    """Predict entities in text"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Predict
        predictions = ner_model.predict(text)
        
        # Format results
        entities = [{"word": word, "label": label} for word, label in predictions if label != 'O']
        
        return jsonify({
            "text": text,
            "entities": entities,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)



import os
import json
from pathlib import Path
from BERT_Trainer import BERTNERInference
from tqdm import tqdm

def process_documents_batch(input_dir: str, output_dir: str, model_path: str):
    """Process multiple documents in batch"""
    
    # Load model
    inference = BERTNERInference(model_path)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process all text files in input directory
    input_path = Path(input_dir)
    text_files = list(input_path.glob("*.txt"))
    
    results = []
    
    for file_path in tqdm(text_files, desc="Processing documents"):
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Tokenize
            tokenized = inference.tokenize_sensitive_info(text)
            
            # Save tokenized version
            output_file = Path(output_dir) / f"{file_path.stem}_tokenized.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(tokenized)
            
            results.append({
                "input_file": str(file_path),
                "output_file": str(output_file),
                "status": "success"
            })
            
        except Exception as e:
            results.append({
                "input_file": str(file_path),
                "error": str(e),
                "status": "failed"
            })
    
    # Save processing report
    with open(Path(output_dir) / "processing_report.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Batch processing completed. Results saved to {output_dir}")
    return results

# Example usage:
process_documents_batch("./input_docs", "./output_docs", "./trained_ner_model")