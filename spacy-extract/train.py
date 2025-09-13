import spacy
from spacy.training.example import Example
import random
from sklearn.model_selection import train_test_split
from utils.io_service import load_data
from spacy.scorer import Scorer
from pathlib import Path
from utils.logger import logger
import time
import torch


def evaluate(nlp, data):
    scorer = Scorer()
    examples = []
    for text, annotations in data:
        doc = nlp(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)
    return scorer.score(examples)


def train(
    train_data_path,
    model_path,
    number_of_epochs: int = 10,
    use_early_stopping: bool = False,
    min_delta: float = 0.001,
    patience: int = 2,
    batch_size: int = 256,
    dropout: float = 0.2,
    use_gpu: bool = True,  # ← Now we respect this
):
    # --- STEP 1: ENABLE GPU VIA PYTORCH (THIS IS THE ONLY WAY SPACY 3.X USES GPU) ---
    if use_gpu and torch.cuda.is_available():
        try:
            spacy.require_gpu()  # 👈 THIS IS THE KEY LINE!
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"✅ Successfully enabled GPU: {device_name}")
        except Exception as e:
            logger.error(f"❌ Failed to enable GPU: {e}")
            logger.warning("⚠️ Falling back to CPU.")
            use_gpu = False
    else:
        logger.warning("⚠️ GPU disabled or unavailable. Using CPU.")
        if not torch.cuda.is_available():
            logger.warning("   → PyTorch CUDA not available. Install torch with CUDA support.")

    # --- STEP 2: INIT MODEL ---
    nlp = spacy.blank("ro")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    # --- STEP 3: LOAD AND PREPARE DATA ---
    all_data = load_data(train_data_path)
    train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)

    LABEL_INDEX = 2
    for _, annotations in train_data:
        for ent in annotations.get("entities", []):
            ner.add_label(ent[LABEL_INDEX])

    nlp.initialize()  # ← This will now use GPU if spacy.require_gpu() was called

    best_f = 0.0
    epochs_no_improve = 0
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    # --- STEP 4: TRAINING LOOP ---
    for epoch in range(number_of_epochs):
        start_time = time.time()
        random.shuffle(train_data)
        losses = {}

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                examples.append(Example.from_dict(doc, annotations))

            nlp.update(examples, drop=dropout, losses=losses)

            if (i // batch_size) % 100 == 0:
                processed = i + len(batch)
                pct = (processed / len(train_data)) * 100
                logger.info(f"Epoch {epoch+1}: {processed}/{len(train_data)} ({pct:.1f}%) done")

        # Evaluate
        scores = evaluate(nlp, test_data)
        f_score = scores["ents_f"]
        epoch_time = time.time() - start_time

        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.1f}s | F1 = {f_score:.4f}")

        if f_score > best_f:
            best_f = f_score
            save_path = model_path / f"best_model_epoch{epoch+1}"
            nlp.to_disk(save_path)
            logger.info(f"🚀 Saved best model: {save_path} (F1={f_score:.4f})")

        # Early stopping
        if use_early_stopping:
            if f_score <= best_f - min_delta:
                epochs_no_improve += 1
                logger.warning(f"No improvement for {epochs_no_improve} epoch(s).")
            else:
                epochs_no_improve = 0

            if epochs_no_improve >= patience:
                logger.info(f"🛑 Early stopping triggered. Best F1={best_f:.4f}")
                break

    logger.info(f"🏆 Final best F1: {best_f:.4f}")


if __name__ == "__main__":
    train_data_path = r"train2.jsonl"
    model_path = "models/spacy_metadata_extraction_model1.0"
    NUMBER_OF_EPOCHS = 5
    train(train_data_path, model_path, NUMBER_OF_EPOCHS, use_gpu=True)