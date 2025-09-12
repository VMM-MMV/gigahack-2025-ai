import json
from pathlib import Path
import re

def convert_token_dataset_to_spacy(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    converted = []

    print(len(raw_data))

    for sample in raw_data:
        tokens = sample["tokens"]
        ner_tags = sample["ner_tags"]

        # Remove HTML tags and their corresponding NER tags
        cleaned_tokens = []
        cleaned_ner_tags = []
        all_tag_tokens = []
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Check if current token is start of HTML tag
            if token == "<":
                # Look ahead to find the complete HTML tag
                tag_tokens = ["<"]
                j = i + 1
                
                # Collect tokens until we find the closing ">"
                while j < len(tokens) and tokens[j] != ">":
                    tag_tokens.append(tokens[j])
                    j += 1
                
                # Add the closing ">" if found
                if j < len(tokens) and tokens[j] == ">":
                    tag_tokens.append(">")
                    j += 1
                
                # Skip all tokens that were part of the HTML tag
                i = j
                # print(tag_tokens)
                all_tag_tokens.extend(tag_tokens)
                continue
            else:
                # Keep non-HTML tag tokens
                cleaned_tokens.append(token)
                cleaned_ner_tags.append(ner_tags[i])
                i += 1

        if len(all_tag_tokens) >= 1: 
            # print(all_tag_tokens)
            continue

        # if all_tag_tokens != []: print(all_tag_tokens)

        # If all tokens were HTML tags, skip this sample
        if not cleaned_tokens:
            continue

        tokens = cleaned_tokens
        ner_tags = cleaned_ner_tags

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

        converted.append((text, {"entities": entities}))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # only the folder
    with open(output_path, "w", encoding="utf-8") as f_out:
        for item in converted:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Converted {len(converted)} samples to {output_path}")

if __name__ == "__main__":
    convert_token_dataset_to_spacy(
        input_path=r"C:\Users\mihai_vieru\Desktop\FinTech - PII_MD\synthetic_moldova_pii_data.json",
        output_path=r"data\ner_dataset_spacy.jsonl"
    )