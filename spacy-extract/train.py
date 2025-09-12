import spacy
from spacy.training.example import Example
import random
from sklearn.model_selection import train_test_split
from utils.io_service import load_data
from spacy.training import offsets_to_biluo_tags

# Add this import
spacy.prefer_gpu()

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

def train(train_data_path, model_path, number_of_epochs:int):
    # Use GPU if available
    if spacy.prefer_gpu():
        print("Using GPU")
    else:
        print("GPU not available, using CPU")

    # Load blank or pre-trained model
    nlp = spacy.blank("ro")  # Romanian language model
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    all_data = load_data(train_data_path)
    train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)

    # Add labels to NER
    LABEL_INDEX = 2
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[LABEL_INDEX])

    nlp.initialize()

    # Train model
    for epoch in range(number_of_epochs):
        random.shuffle(train_data)
        losses = {}
        for i, (text, annotations) in enumerate(train_data):
            print(i)
            entities = annotations.get("entities", [])
            check_alignment(nlp, text, entities)
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.3, losses=losses)

        print(f"Epoch {epoch + 1}: Losses {losses}")

    nlp.to_disk(model_path)

if __name__ == "__main__":
    train_data_path = r"C:\Users\mihai_vieru\Desktop\gigahack-2025-ai\data\ner_dataset_spacy.jsonl"
    model_path = "spacy_metadata_extraction_model1.0"
    NUMBER_OF_EPOCHS = 10
    train(train_data_path, model_path, NUMBER_OF_EPOCHS)
