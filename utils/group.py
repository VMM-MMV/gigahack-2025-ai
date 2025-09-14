import json
from collections import defaultdict

def extract_and_save_entities_original(input_json_path, output_json_path, remove_duplicates=True):
    """
    Load a JSON file with tokenized data, group tokens by NER tags,
    combine entities of the same type, and save to a new JSON file.

    Args:
        input_json_path (str): Path to the input JSON file.
        output_json_path (str): Path to save the grouped entities JSON.
        remove_duplicates (bool): Whether to remove duplicate entity values.
    """
    # Load JSON file
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def group_tokens_by_tag(tokens, ner_tags):
        grouped = []
        current_tokens = []
        current_tag = None

        for tok, tag in zip(tokens, ner_tags):
            normalized_tag = tag[2:] if tag.startswith(("B-", "I-")) else tag

            if normalized_tag == "O":
                if current_tokens:
                    grouped.append((current_tag, " ".join(current_tokens)))
                    current_tokens = []
                    current_tag = None
            else:
                if normalized_tag != current_tag:
                    if current_tokens:
                        grouped.append((current_tag, " ".join(current_tokens)))
                    current_tag = normalized_tag
                    current_tokens = [tok]
                else:
                    current_tokens.append(tok)

        if current_tokens:
            grouped.append((current_tag, " ".join(current_tokens)))

        return grouped

    # Combine all entries into a dictionary
    entities_dict = defaultdict(list)

    for entry in data:
        tokens = entry.get("tokens", [])
        ner_tags = entry.get("ner_tags", [])
        grouped_entities = group_tokens_by_tag(tokens, ner_tags)

        for label, value in grouped_entities:
            entities_dict[label].append(value)

    # Optionally remove duplicates
    if remove_duplicates:
        entities_dict = {label: list(dict.fromkeys(values)) for label, values in entities_dict.items()}

    # Save to JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(entities_dict, f, ensure_ascii=False, indent=2)

    print(f"Grouped entities saved to {output_json_path}")
    return entities_dict

def extract_and_save_entities_spacy(input_jsonl_path, output_json_path, remove_duplicates=True):
    entities_dict = defaultdict(list)

    with open(input_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            items = json.loads(line)  # Should be a list: [text, {"entities": [...]}]
            if isinstance(items, list) and len(items) == 2:
                text, entity_info = items
                for start, end, label in entity_info.get("entities", []):
                    entity_value = text[start:end]
                    entities_dict[label].append(entity_value)
            # else:
            #     # fallback: handle normal dict format
            #     if isinstance(items, dict):
            #         text = items.get("text", "")
            #         for start, end, label in items.get("entities", []):
            #             entity_value = text[start:end]
            #             entities_dict[label].append(entity_value)
            #     else:
            #         print("Unexpected line format:", line)

    # Save grouped entities to JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(entities_dict, f, ensure_ascii=False, indent=2)

    print(f"Grouped entities saved to {output_json_path}")
    return entities_dict


if __name__ == "__main__":
    entities = extract_and_save_entities_spacy(
        "data/saas/ner_dataset_spacy-nw.jsonl",
        "grouped_entities.json"
    )

    # Example usage
    # entities = extract_and_save_entities("synthetic_moldova_pii_data.json", "grouped_entities.json")
