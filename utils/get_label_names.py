import json
from collections import defaultdict

def get_all_labels(input_json_path):
    """Extract all unique NER labels from the dataset"""
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    labels = set()
    
    for entry in data:
        ner_tags = entry.get("ner_tags", [])
        for tag in ner_tags:
            # Remove BIO prefix (B-, I-) and keep only the label
            if tag.startswith(("B-", "I-")):
                label = tag[2:]
            else:
                label = tag
            
            if label != "O":  # Skip "Outside" tags
                labels.add(label)
    
    return sorted(list(labels))

def get_all_labels_spacy(input_jsonl_path):
    """Extract all unique labels from spaCy format"""
    labels = set()
    
    with open(input_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            items = json.loads(line)
            if isinstance(items, list) and len(items) == 2:
                _, entity_info = items
                for start, end, label in entity_info.get("entities", []):
                    labels.add(label)
    
    return sorted(list(labels))

# Usage example
if __name__ == "__main__":
    # For BIO format
    # labels = get_all_labels("synthetic_moldova_pii_data.json")
    
    # For spaCy format
    labels = get_all_labels_spacy("data/ner_dataset_spacy.jsonl")
    
    print("All NER labels:")
    for label in labels:
        print(f"- {label}")
    
    print(f"\nTotal labels: {len(labels)}")