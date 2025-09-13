import json
from pathlib import Path
import re

def convert_token_dataset_to_conll(input_path, output_path):
    """
    Convert token dataset to CoNLL format for BERT transformer training.
    CoNLL format: token\tBIO_tag per line, with empty lines between sentences.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    converted_samples = []
    print(f"Processing {len(raw_data)} samples...")

    for sample_idx, sample in enumerate(raw_data):
        tokens = sample["tokens"]
        ner_tags = sample["ner_tags"]

        # Remove HTML tags and their corresponding NER tags
        cleaned_tokens = []
        cleaned_ner_tags = []
        html_tag_found = False
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Check if current token is start of HTML tag
            if token == "<":
                html_tag_found = True
                # Look ahead to find the complete HTML tag
                j = i + 1
                
                # Collect tokens until we find the closing ">"
                while j < len(tokens) and tokens[j] != ">":
                    j += 1
                
                # Add the closing ">" if found
                if j < len(tokens) and tokens[j] == ">":
                    j += 1
                
                # Skip all tokens that were part of the HTML tag
                i = j
                continue
            else:
                # Keep non-HTML tag tokens
                cleaned_tokens.append(token)
                cleaned_ner_tags.append(ner_tags[i])
                i += 1

        # Skip samples that contained HTML tags (as in original code)
        if html_tag_found:
            continue

        # If all tokens were HTML tags, skip this sample
        if not cleaned_tokens:
            continue

        # Validate that tokens and tags have the same length
        if len(cleaned_tokens) != len(cleaned_ner_tags):
            print(f"Warning: Sample {sample_idx} has mismatched tokens and tags length. Skipping.")
            continue

        # Add the cleaned sample
        converted_samples.append({
            'tokens': cleaned_tokens,
            'ner_tags': cleaned_ner_tags
        })

    # Write to CoNLL format
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        for sample_idx, sample in enumerate(converted_samples):
            tokens = sample['tokens']
            ner_tags = sample['ner_tags']
            
            # Write each token-tag pair on a separate line
            for token, tag in zip(tokens, ner_tags):
                # Clean the token (remove extra whitespace, newlines)
                clean_token = token.strip().replace('\n', ' ').replace('\t', ' ')
                if clean_token:  # Only write non-empty tokens
                    f_out.write(f"{clean_token}\t{tag}\n")
            
            # Add empty line between sentences (except after the last sample)
            if sample_idx < len(converted_samples) - 1:
                f_out.write("\n")

    print(f"Converted {len(converted_samples)} samples to CoNLL format: {output_path}")
    
    # Print some statistics
    total_tokens = sum(len(sample['tokens']) for sample in converted_samples)
    unique_tags = set()
    for sample in converted_samples:
        unique_tags.update(sample['ner_tags'])
    
    print(f"Total tokens: {total_tokens}")
    print(f"Unique NER tags: {sorted(unique_tags)}")

def validate_conll_file(file_path):
    """
    Validate the generated CoNLL file format.
    """
    print(f"\nValidating CoNLL file: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    sentence_count = 0
    token_count = 0
    empty_line_count = 0
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        
        if not line:  # Empty line (sentence separator)
            empty_line_count += 1
            continue
        
        parts = line.split('\t')
        if len(parts) != 2:
            print(f"Warning: Line {line_num} has {len(parts)} parts instead of 2: {line}")
            continue
        
        token, tag = parts
        token_count += 1
        
        # Count sentences (assuming each non-empty block is a sentence)
        if line_num == 1 or (line_num > 1 and lines[line_num-2].strip() == ""):
            sentence_count += 1
    
    print(f"Sentences: {sentence_count}")
    print(f"Tokens: {token_count}")
    print(f"Empty lines: {empty_line_count}")
    print("Validation complete!")

if __name__ == "__main__":
    input_file = "synthetic_moldova_pii_data.json"
    output_file = "data/ner_dataset_conll.txt"
    
    # Convert to CoNLL format
    convert_token_dataset_to_conll(input_file, output_file)
    
    # Validate the output file
    validate_conll_file(output_file)