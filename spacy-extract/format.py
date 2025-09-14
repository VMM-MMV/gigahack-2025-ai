import json

# Path to your JSONL file
file_path = "data/ner_dataset_spacy.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        # Each line is a JSON object
        entry = json.loads(line.strip())
        text = entry[0]  # the main text
        entities = entry[1]["entities"]

        print("Text:", text[:100], "...")  # show only first 100 chars for brevity
        for start, end, label in entities:
            print(f"{label}: |{text[start:end]}|")
        print("-" * 50)
