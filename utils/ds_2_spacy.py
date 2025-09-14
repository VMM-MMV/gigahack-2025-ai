import json
from pathlib import Path
import re
from wordfreq import zipf_frequency

def should_merge(tok1, tok2, lang="ro"):
    # First check if tokens form any known entity patterns
    merged = tok1 + tok2
    
    # Personal identification patterns
    if re.match(r'^[0-9]{13}$', merged):  # CNP
        return True
    if re.match(r'^[A-Za-z]{2}[0-9]{6,10}$', merged):  # Passport numbers
        return True
    if re.match(r'^[0-9]{6,10}$', merged) and len(merged) >= 6:  # Identity card numbers
        return True
    
    # Contact information patterns
    if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', merged):  # Email
        return True
    if re.match(r'^[0-9]{9,10}$', merged):  # Phone numbers
        return True
    if re.match(r'^MD-[0-9]{4}$', merged):  # Postal codes
        return True
    
    # Financial patterns
    if re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]{16,32}$', merged):  # IBAN
        return True
    if re.match(r'^[0-9]{16}$', merged):  # Card numbers (full)
        return True
    if re.match(r'^\*+[0-9]{4}$', merged):  # Masked card numbers
        return True
    if re.match(r'^[0-9]{4,20}$', merged):  # Account numbers
        return True
    
    # Contract and document patterns
    if re.match(r'^[A-Z]{3}-[0-9]{4}-[0-9]{4,8}$', merged):  # Contract numbers
        return True
    if re.match(r'^[A-Z]{3}[0-9]{3,6}[A-Z]{0,2}$', merged):  # License plates
        return True
    if re.match(r'^[A-Z]{2}[0-9]{7,9}$', merged):  # License numbers
        return True
    
    # Medical patterns
    if re.match(r'^AM[0-9]{10}$', merged):  # Insurance numbers
        return True
    if re.match(r'^[ABO][+-]$', merged):  # Blood types
        return True
    
    # Digital patterns
    if re.match(r'^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$', merged):  # IP addresses
        return True
    if re.match(r'^DEV[0-9]{9}$', merged):  # Device IDs
        return True
    
    # Handle hyphenated names
    if re.match(r'^[A-Z][a-z]+-[A-Z][a-z]+$', merged):  # Hyphenated names
        return True
    
    # Handle specific known entities
    if merged.lower() in ["moldtelecom", "gmail"]:
        return True
    
    # Finally, check if it's a valid Romanian word
    if not tok1.isalnum() or not tok2.isalnum():
        return False
    
    return zipf_frequency(merged.lower(), lang) > 0

def fix_broken_tokens(tokens: list[str], ner_tags: list[str]) -> tuple[list[str], list[str]]:
    """
    Merge tokens that should be together and update their corresponding NER tags.
    When merging, we keep the tag from the first token.
    """
    merged_tokens: list[str] = []
    merged_tags: list[str] = []
    i = 0

    while i < len(tokens):
        if i + 1 < len(tokens) and should_merge(tokens[i], tokens[i + 1]):
            merged = tokens[i] + tokens[i + 1]
            merged_tokens.append(merged)
            # Keep the tag from the first token, but if it's 'O' and second is not, use second
            if ner_tags[i] == 'O' and ner_tags[i + 1] != 'O':
                merged_tags.append(ner_tags[i + 1])
            else:
                merged_tags.append(ner_tags[i])
            i += 2
        else:
            merged_tokens.append(tokens[i])
            merged_tags.append(ner_tags[i])
            i += 1

    return merged_tokens, merged_tags

def remove_adjacent_duplicates(tokens, tags):
    """Collapse duplicates like ['Andrew','Andrew'] with same tag -> ['Andrew']."""
    cleaned_tokens, cleaned_tags = [], []
    for t, tag in zip(tokens, tags):
        if cleaned_tokens and cleaned_tokens[-1] == t and cleaned_tags[-1] == tag:
            continue
        cleaned_tokens.append(t)
        cleaned_tags.append(tag)
    return cleaned_tokens, cleaned_tags

def should_add_space(current_token, next_token):
    """Determine if a space should be added between two tokens."""
    if current_token == "@":
        return False
    if current_token in [".", "-"] and next_token and not next_token[0].isupper():
        return False
    if current_token == "," and next_token and next_token[0].isdigit():
        return False
    if next_token and next_token[0].isalnum():
        return True
    return False

def remove_html_tags(tokens, ner_tags):
    """
    Remove HTML tags by matching specific token patterns and remove all '>' characters.
    """
    cleaned_tokens = []
    cleaned_ner_tags = []
    removed_html_tags = []
    
    valid_html_tags = {
        'a', 'abbr', 'address', 'area', 'article', 'aside', 'audio', 'b', 'base', 
        'bdi', 'bdo', 'blockquote', 'body', 'br', 'button', 'canvas', 'caption', 
        'cite', 'code', 'col', 'colgroup', 'data', 'datalist', 'dd', 'del', 
        'details', 'dfn', 'dialog', 'div', 'dl', 'dt', 'em', 'embed', 'fieldset', 
        'figcaption', 'figure', 'footer', 'form', 'h1', 'h2', 'h3', 'h4', 'h5', 
        'h6', 'head', 'header', 'hr', 'html', 'i', 'iframe', 'img', 'input', 
        'ins', 'kbd', 'label', 'legend', 'li', 'link', 'main', 'map', 'mark', 
        'meta', 'meter', 'nav', 'noscript', 'object', 'ol', 'optgroup', 'option', 
        'output', 'p', 'param', 'picture', 'pre', 'progress', 'q', 'rp', 'rt', 
        'ruby', 's', 'samp', 'script', 'section', 'select', 'small', 'source', 
        'span', 'strong', 'style', 'sub', 'summary', 'sup', 'table', 'tbody', 
        'td', 'template', 'textarea', 'tfoot', 'th', 'thead', 'time', 'title', 
        'tr', 'track', 'u', 'ul', 'var', 'video', 'wbr'
    }
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # Remove all ">" characters
        if token == ">":
            i += 1
            continue
            
        # Check for HTML tag pattern starting with "<"
        if token == "<":
            tag_start = i
            i += 1
            
            if i < len(tokens) and tokens[i] == "/":
                i += 1
            
            # HTML tag name
            if i < len(tokens) and tokens[i].lower() in valid_html_tags:
                i += 1
                
                if i < len(tokens) and tokens[i] == "/":
                    i += 1
                
                removed_html_tags.extend(tokens[tag_start:i])
                
        else:
            cleaned_tokens.append(token)
            cleaned_ner_tags.append(ner_tags[i])
            i += 1
    
    # Print removed tags if any were found
    # if len(removed_html_tags) > 1:
    #     print(f"Removed valid HTML tags: {' '.join(removed_html_tags)}")
    
    return cleaned_tokens, cleaned_ner_tags

def remove_new_line(tokens, ner_tags):
    """
    Remove HTML tags by matching specific token patterns and remove all '>' characters.
    """
    cleaned_tokens = []
    cleaned_ner_tags = []
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if i < len(tokens) and i+1 < len(tokens):
            # if token in ["\\", "n"]:
            #     print(ner_tags[i])

            if token == "\\" and tokens[i+1] and tokens[i+1][0] == "n":
                if len(tokens[i+1]) > 1:
                    tokens[i+1] = tokens[i+1][1:]
                    i += 1
                    continue
                i += 2
            # elif token == "\\":
            #     i += 1
        cleaned_tokens.append(token)
        cleaned_ner_tags.append(ner_tags[i])
        i += 1
        
    return cleaned_tokens, cleaned_ner_tags

def convert_token_dataset_to_spacy(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    converted = []
    print(f"Processing {len(raw_data)} samples...")

    for sample_idx, sample in enumerate(raw_data):
        tokens = sample["tokens"]
        ner_tags = sample["ner_tags"]

        tokens, ner_tags = remove_new_line(tokens, ner_tags)

        tokens, ner_tags = remove_html_tags(tokens, ner_tags)
        
        if not tokens:
            continue

        # --- Step 2: Merge split tokens (like dates, emails, etc.) ---
        tokens, ner_tags = fix_broken_tokens(tokens, ner_tags)

        # --- Step 3: Remove adjacent duplicates ---
        tokens, ner_tags = remove_adjacent_duplicates(tokens, ner_tags)

        # --- Step 4: Rebuild text with proper spacing ---
        text = ""
        token_offsets = []
        pos = 0

        for i, token in enumerate(tokens):
            start = pos
            text += token
            end = start + len(token)
            token_offsets.append((start, end))
            pos = end

            # Add space if needed
            if i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if should_add_space(token, next_token):
                    text += " "
                    pos += 1

        # --- Step 5: Convert BIO tags to entities ---
        entities = []
        entity_start, entity_label, entity_end = None, None, None

        for i, ((start, end), tag) in enumerate(zip(token_offsets, ner_tags)):
            if tag == "O":
                if entity_start is not None:
                    entities.append((entity_start, entity_end, entity_label))
                    entity_start, entity_label, entity_end = None, None, None
            elif tag.startswith("B-"):
                if entity_start is not None:
                    entities.append((entity_start, entity_end, entity_label))
                entity_start = start
                entity_label = tag[2:]
                entity_end = end
            elif tag.startswith("I-"):
                current_label = tag[2:]
                if entity_start is not None and entity_label == current_label:
                    # Continue current entity - update the end position
                    entity_end = end
                else:
                    # Start new entity if no current entity or different label
                    if entity_start is not None:
                        entities.append((entity_start, entity_end, entity_label))
                    entity_start = start
                    entity_label = current_label
                    entity_end = end
            else:
                # Handle non-BIO tags (direct labels without B-/I- prefix)
                if entity_start is not None:
                    entities.append((entity_start, entity_end, entity_label))
                # For direct labels, treat as single-token entity
                entity_start = start
                entity_label = tag
                entity_end = end

        # Close any remaining entity
        if entity_start is not None:
            entities.append((entity_start, entity_end, entity_label))

        converted.append((text, {"entities": entities}))

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f_out:
        for item in converted:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Converted {len(converted)} samples to {output_path}")

if __name__ == "__main__":
    convert_token_dataset_to_spacy(
        # input_path=r"mock_subset_200.json",
        input_path=r"synthetic_moldova_pii_data.json",
        output_path=r"data/ner_dataset_spacy.jsonl"
    )