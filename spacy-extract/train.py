import spacy
from spacy.training.example import Example
import random
from sklearn.model_selection import train_test_split
from utils.io_service import load_data
from spacy.training import offsets_to_biluo_tags
from spacy.scorer import Scorer
from pathlib import Path

def sanity_check(nlp, train_data):
    SAMPLE_INDEX = 0
    print("Sanity check on training Data:")
    sample_text, _ = train_data[SAMPLE_INDEX]
    print("Sample text: "+sample_text)
    doc = nlp(sample_text)
    for ent in doc.ents:
        print(ent.text, ent.label_)

def check_alignment(nlp, text, entities, filename=None):
    doc = nlp.make_doc(text)
    tags = offsets_to_biluo_tags(doc, entities)
    if '-' in tags:
        file_info = f" in file: {filename}" if filename else ""
        print(f"⚠ Misaligned entities detected{file_info}")
        print("Text:", text)
        print("Entities:", entities)
        print("BILUO tags:", tags)
    return tags

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
    patience: int = 2
):
    """
    Train a spaCy NER model, optionally using early stopping.

    Args:
        train_data_path (str): Path to dataset
        model_path (str): Where to save the best model
        number_of_epochs (int): Max epochs if using early stopping, or total epochs if not
        use_early_stopping (bool): If True, stop early when F1 improvement is small
        min_delta (float): Minimum improvement in F1 for early stopping
        patience (int): Epochs to wait without improvement before stopping
    """
    # if spacy.prefer_gpu():
    #     print("Using GPU")
    # else:
    #     print("GPU not available, using CPU")

    nlp = spacy.blank("ro")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    all_data = load_data(train_data_path)
    train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)

    LABEL_INDEX = 2
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[LABEL_INDEX])

    nlp.initialize()

    best_f = 0.0
    epochs_no_improve = 0

    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(number_of_epochs):
        random.shuffle(train_data)
        losses = {}
        for i, (text, annotations) in enumerate(train_data):
            print(i)
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.3, losses=losses)

        scores = evaluate(nlp, test_data)
        f_score = scores["ents_f"]

        print(f"Epoch {epoch + 1}: Losses {losses}, F1 = {f_score:.4f}")

        # Save if best
        if f_score > best_f:
            best_f = f_score
            nlp.to_disk(model_path / f"best_model_epoch{epoch+1}")
            print(f"New best model saved (F1={f_score:.4f})")

        # Handle early stopping
        if use_early_stopping:
            if f_score <= best_f - min_delta:
                epochs_no_improve += 1
                print(f"⚠ No significant improvement for {epochs_no_improve} epoch(s).")
            else:
                epochs_no_improve = 0

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered. Best F1={best_f:.4f}")
                break


if __name__ == "__main__":
    train_data_path = r"data/ner_dataset_spacy-1200.jsonl"
    model_path = "models/spacy_metadata_extraction_model1.0"
    NUMBER_OF_EPOCHS = 10
    train(train_data_path, model_path, NUMBER_OF_EPOCHS)
