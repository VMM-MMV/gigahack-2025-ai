import json

def convert_token_dataset_to_spacy(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    converted = []
    for sample in raw_data:
        tokens = sample["tokens"]
        ner_tags = sample["ner_tags"]
        space_after = sample.get("space_after", [True] * len(tokens))  # fallback if missing

        # Rebuild text with spaces
        text_parts = []
        token_offsets = []
        pos = 0
        for token, has_space in zip(tokens, space_after):
            text_parts.append(token)
            start = pos
            end = pos + len(token)
            token_offsets.append((start, end))
            pos = end + (1 if has_space else 0)
            if has_space:
                text_parts.append(" ")

        text = "".join(text_parts).rstrip()  # remove trailing space

        # Collect entities from BIO/BILOU-style tags
        entities = []
        entity_start, entity_label = None, None

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
                # continuation of entity
                pass
            else:
                # Handle inconsistent tags (start a new entity)
                if entity_start is not None:
                    entities.append((entity_start, prev_end, entity_label))
                entity_start = start
                entity_label = tag[2:] if "-" in tag else tag

            prev_end = end

        # Close last entity if open
        if entity_start is not None:
            entities.append((entity_start, prev_end, entity_label))

        converted.append((text, {"entities": entities}))

    # Write to spaCy JSONL format
    with open(output_path, "w", encoding="utf-8") as f_out:
        for item in converted:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Converted {len(converted)} samples to {output_path}")

if __name__ == "__main__":
    convert_token_dataset_to_spacy(
        input_path="../Data/new_format_dataset.json",
        output_path="../Data/ner_dataset_spacy.jsonl"
    )
