import json
from pathlib import Path
import re
from wordfreq import zipf_frequency

def should_merge(tok1, tok2, lang="ro"):
    if not tok1.isalnum() or not tok2.isalnum():
        return False
    merged = (tok1 + tok2).lower()
    return zipf_frequency(merged, lang) > 0

def fix_broken_tokens(tokens: list[str], space_after: list[bool]) -> tuple[list[str], list[bool]]:
    merged_tokens: list[str] = []
    merged_space: list[bool] = []
    i = 0

    while i < len(tokens):
        if i + 1 < len(tokens) and should_merge(tokens[i], tokens[i + 1]):
            merged = tokens[i] + tokens[i + 1]
            merged_tokens.append(merged)
            merged_space.append(space_after[i + 1])  # use spacing of last part
            i += 2
        else:
            merged_tokens.append(tokens[i])
            merged_space.append(space_after[i])
            i += 1

    return merged_tokens, merged_space

def remove_adjacent_duplicates(tokens, tags):
    """Collapse duplicates like ['Andrew','Andrew'] with same tag -> ['Andrew']."""
    cleaned_tokens, cleaned_tags = [], []
    for t, tag in zip(tokens, tags):
        if cleaned_tokens and cleaned_tokens[-1] == t and cleaned_tags[-1] == tag:
            continue
        cleaned_tokens.append(t)
        cleaned_tags.append(tag)
    return cleaned_tokens, cleaned_tags

def convert_token_dataset_to_spacy(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    converted = []

    print(len(raw_data))

    for sample in raw_data:
        tokens = sample["tokens"]
        ner_tags = sample["ner_tags"]

        # --- Step 1: Remove HTML tags ---
        cleaned_tokens = []
        cleaned_ner_tags = []
        all_tag_tokens = []
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token == "<":
                tag_tokens = ["<"]
                j = i + 1
                while j < len(tokens) and tokens[j] != ">":
                    tag_tokens.append(tokens[j])
                    j += 1
                if j < len(tokens) and tokens[j] == ">":
                    tag_tokens.append(">")
                    j += 1
                i = j
                all_tag_tokens.extend(tag_tokens)
                continue
            else:
                cleaned_tokens.append(token)
                cleaned_ner_tags.append(ner_tags[i])
                i += 1

        if len(all_tag_tokens) >= 1:
            print("Removed HTML tags:", " ".join(all_tag_tokens))
        if not cleaned_tokens:
            continue

        tokens, ner_tags = cleaned_tokens, cleaned_ner_tags

        # --- Step 2: Merge split names ---
        tokens, ner_tags = fix_broken_tokens(tokens, ner_tags)

        # --- Step 3: Remove adjacent duplicates ---
        tokens, ner_tags = remove_adjacent_duplicates(tokens, ner_tags)

        # --- Step 4: Rebuild text with spaces ---
        text = ""
        token_offsets = []
        pos = 0

        for i, token in enumerate(tokens):
            start = pos
            text += token
            end = start + len(token)
            token_offsets.append((start, end))
            pos = end

            if i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if token == "@":
                    continue
                if token in [".", "-"] and next_token and not next_token[0].isupper():
                    continue
                if token in [","] and next_token and next_token[0].isdigit():
                    continue
                if next_token and next_token[0].isalnum():
                    text += " "
                    pos += 1

        # --- Step 5: Convert BIO tags to entities ---
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
                pass
            else:
                if entity_start is not None:
                    entities.append((entity_start, prev_end, entity_label))
                entity_start = start
                entity_label = tag[2:] if "-" in tag else tag
            prev_end = end

        if entity_start is not None:
            entities.append((entity_start, prev_end, entity_label))

        converted.append((text, {"entities": entities}))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f_out:
        for item in converted:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Converted {len(converted)} samples to {output_path}")

if __name__ == "__main__":
    convert_token_dataset_to_spacy(
        input_path=r"/home/serveruser/gigahack-2025-ai/mock_subset_200.json",
        output_path=r"data/ner_dataset_spacy.jsonl"
    )
