import json
from pathlib import Path
import re

def convert_token_dataset_to_spacy(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    converted = []

    for sample in raw_data:
        tokens = sample["tokens"]
        ner_tags = sample["ner_tags"]

        text = ""
        token_offsets = []
        pos = 0

        for i, token in enumerate(tokens):
            start = pos
            text += token
            end = start + len(token)
            token_offsets.append((start, end))
            pos = end

            # Decide whether to add a space
            if i + 1 < len(tokens):
                next_token = tokens[i + 1]

                # Skip space after '@'
                if token == "@":
                    continue

                # Skip space after '.' if next token starts with uppercase letter
                if token in [".", "-"] and next_token and not next_token[0].isupper():
                    continue

                if token in [","] and next_token and next_token[0].isdigit():
                    continue

                # Otherwise, add space only if next token starts with alphanumeric
                if next_token and next_token[0].isalnum():
                    text += " "
                    pos += 1

        # --- Convert BIO tags to entities ---
        entities = []
        entity_start, entity_label = None, None
        prev_end = None

        for (start, end), tag in zip(token_offsets, ner_tags):
            if tag == "O":
                if entity_start is not None:
                    entities.append((entity_start, prev_end, entity_label))
                    entity_start, entity_label = None, None
            elif tag.startswith("B-"):
                if entity_start is not None:
                    entities.append((entity_start, prev_end, entity_label))
                entity_start = start
                entity_label = tag[2:]
            elif tag.startswith("I-") and entity_label == tag[2:]:
                pass  # continuation
            else:
                # inconsistent tag (treat as B-)
                if entity_start is not None:
                    entities.append((entity_start, prev_end, entity_label))
                entity_start = start
                entity_label = tag[2:] if "-" in tag else tag
            prev_end = end

        if entity_start is not None:
            entities.append((entity_start, prev_end, entity_label))

        tags_with_space = re.findall(r"<[^>]+>\s", text)
        if tags_with_space: print("Tags followed by space:", tags_with_space)

        # Remove those tags from the text
        clean_text = re.sub(r"<[^>]+>\s", "", text)

        converted.append((text, {"entities": entities}))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # only the folder
    with open(output_path, "w", encoding="utf-8") as f_out:
        for item in converted:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Converted {len(converted)} samples to {output_path}")

if __name__ == "__main__":
    convert_token_dataset_to_spacy(
        input_path=r"C:\Users\mihai_vieru\Desktop\gigahack-2025-ai\mock_subset_200.json",
        output_path=r"data\ner_dataset_spacy.jsonl"
    )
