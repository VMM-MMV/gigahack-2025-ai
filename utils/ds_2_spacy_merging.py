import json
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def bio2_json_to_spacy_jsonl(json_filepath: str, output_filepath: str):
    with open(json_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_filepath, 'w', encoding='utf-8') as out_f:
        for sentence in data:
            # Handle Format A: List of token dicts
            if isinstance(sentence, list) and len(sentence) > 0 and isinstance(sentence[0], dict):
                tokens = [t['token'] for t in sentence]
                tags = [t['ner_tag'] for t in sentence]
            
            # Handle Format B: Dict with tokens/ner_tags lists
            elif isinstance(sentence, dict) and 'tokens' in sentence and 'ner_tags' in sentence:
                tokens = sentence['tokens']
                tags = sentence['ner_tags']
            
            else:
                logging.warning(f"Skipping malformed sentence: {sentence}")
                continue
            
            # Validate data
            if not tokens or not tags or len(tokens) != len(tags):
                logging.warning(f"Skipping sentence with mismatched tokens/tags (tokens: {len(tokens)}, tags: {len(tags)})")
                continue

            # =============================
            # NEW: ROBUST HTML TAG CLEANING (handles < br / >, < / p >, etc.)
            # =============================
            VALID_TAG_NAMES = {
                'br', 'p', 'div', 'span', 'u', 'b', 'i', 'strong', 'em', 'a', 'img', 'hr',
                'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'font', 'center', 'table', 'tr', 'td', 'th'
            }

            cleaned_tokens = []
            cleaned_tags = []
            i = 0
            n = len(tokens)

            while i < n:
                token = tokens[i]
                tag = tags[i]

                # === CASE 1: Start of potential HTML tag: current token is '<' ===
                if token == '<':
                    j = i + 1
                    end_pos = -1
                    while j < n and tokens[j].strip() == '':
                        j += 1
                    if j < n and tokens[j] in VALID_TAG_NAMES:
                        tag_name = tokens[j]
                        k = j + 1
                        while k < n:
                            if tokens[k] == '>':
                                end_pos = k
                                break
                            elif tokens[k] == '/' and k + 1 < n and tokens[k + 1] == '>':
                                end_pos = k + 1
                                break
                            elif tokens[k].strip() == '':
                                k += 1
                                continue
                            else:
                                break
                        if end_pos != -1:
                            i = end_pos + 1
                            continue
                        else:
                            i = j + 1
                            continue
                    cleaned_tokens.append(token)
                    cleaned_tags.append(tag)
                    i += 1

                # === CASE 2: Start of potential closing tag: current token is '/' ===
                elif token == '/':
                    j = i + 1
                    while j < n and tokens[j].strip() == '':
                        j += 1
                    if j < n and tokens[j] in VALID_TAG_NAMES:
                        k = j + 1
                        while k < n:
                            if tokens[k] == '>':
                                end_pos = k
                                break
                            elif tokens[k].strip() == '':
                                k += 1
                                continue
                            else:
                                break
                        if k < n and tokens[k] == '>':
                            i = k + 1
                            continue
                        else:
                            i = j + 1
                            continue
                    cleaned_tokens.append(token)
                    cleaned_tags.append(tag)
                    i += 1

                # === CASE 3: End of potential tag: current token is '>' ===
                elif token == '>':
                    j = i - 1
                    while j >= 0 and tokens[j].strip() == '':
                        j -= 1
                    if j >= 0 and tokens[j] in VALID_TAG_NAMES:
                        i += 1
                        continue
                    cleaned_tokens.append(token)
                    cleaned_tags.append(tag)
                    i += 1

                # === CASE 4: Token is a known tag name — check context ===
                elif token in VALID_TAG_NAMES:
                    cleaned_tokens.append(token)
                    cleaned_tags.append(tag)
                    i += 1

                # === CASE 5: Isolated '<', '>', '/' not part of any structure — remove if alone ===
                elif token in ['<', '>', '/']:
                    prev_is_alnum = (i > 0 and tokens[i-1].isalnum())
                    next_is_alnum = (i < n - 1 and tokens[i+1].isalnum())
                    if not prev_is_alnum and not next_is_alnum:
                        i += 1
                        continue
                    else:
                        cleaned_tokens.append(token)
                        cleaned_tags.append(tag)
                        i += 1

                else:
                    cleaned_tokens.append(token)
                    cleaned_tags.append(tag)
                    i += 1

            # Final pass: remove any remaining isolated '<', '>', '/'
            final_tokens = []
            final_tags = []
            i = 0
            while i < len(cleaned_tokens):
                token = cleaned_tokens[i]
                tag = cleaned_tags[i]
                if token in ['<', '>', '/']:
                    prev_is_alnum = (i > 0 and cleaned_tokens[i-1].isalnum())
                    next_is_alnum = (i < len(cleaned_tokens)-1 and cleaned_tokens[i+1].isalnum())
                    if not prev_is_alnum and not next_is_alnum:
                        i += 1
                        continue
                final_tokens.append(token)
                final_tags.append(tag)
                i += 1

            tokens = final_tokens
            tags = final_tags

            if not tokens or not tags or len(tokens) != len(tags):
                logging.warning(f"Skipping sentence after HTML cleaning due to mismatch")
                continue
            
            # =============================
            # NEW: EMAIL MERGING (handles "ion . popescu @ gmail . com")
            # =============================
            merged_tokens = []
            merged_tags = []
            i = 0
            n = len(tokens)

            while i < n:
                token = tokens[i]
                tag = tags[i]

                # Look for potential email start: alphanumeric token followed by '.' or '@'
                if re.match(r'^[a-zA-Z0-9_+-]+$', token) and i + 1 < n:
                    # Check if this is the start of an email pattern: word . word @ word . word
                    candidate_tokens = [token]
                    candidate_tags = [tag]
                    j = i + 1
                    seen_at = False
                    valid_email_pattern = True

                    # Scan ahead for email structure: [word][.][word]*[@][word][.][word]+
                    while j < n:
                        next_token = tokens[j]
                        next_tag = tags[j]

                        # Accept '.' or '@' only if they are separators
                        if next_token == '.':
                            # Must be followed by a valid email part (alphanumeric)
                            if j + 1 < n and re.match(r'^[a-zA-Z0-9_+-]+$', tokens[j + 1]):
                                candidate_tokens.append(next_token)
                                candidate_tags.append(next_tag)
                                j += 1
                                continue
                            else:
                                break  # Invalid: '.' not followed by word → not email

                        elif next_token == '@':
                            # Must have seen at least one part before @
                            if len(candidate_tokens) > 0 and not seen_at:
                                candidate_tokens.append(next_token)
                                candidate_tags.append(next_tag)
                                seen_at = True
                                j += 1
                                continue
                            else:
                                break  # Invalid: @ appears too early or twice

                        elif re.match(r'^[a-zA-Z0-9_+-]+$', next_token):
                            # Valid part of email (word)
                            if seen_at:  # After @
                                candidate_tokens.append(next_token)
                                candidate_tags.append(next_tag)
                                j += 1
                                continue
                            else:
                                # Before @ — only allow if followed by @ or .
                                if j + 1 < n and tokens[j + 1] in ['.', '@']:
                                    candidate_tokens.append(next_token)
                                    candidate_tags.append(next_tag)
                                    j += 1
                                    continue
                                else:
                                    break  # Word not followed by . or @ → stop
                        else:
                            break  # Invalid token in email sequence

                    # Validate final structure: must have at least one part before @, one after @
                    if seen_at and len(candidate_tokens) >= 4:  # e.g., [a,.,b,@,c,.,d]
                        # Merge into single email token
                        email = ''.join(t for t in candidate_tokens if t not in ['.', '@'])
                        # But preserve structure: replace . and @ with actual delimiters
                        email_corrected = ""
                        for t in candidate_tokens:
                            if t == '.':
                                email_corrected += '.'
                            elif t == '@':
                                email_corrected += '@'
                            else:
                                email_corrected += t

                        # Ensure it looks like a real email: has exactly one @ and at least one dot after
                        if email_corrected.count('@') == 1 and '.' in email_corrected.split('@')[1]:
                            merged_tokens.append(email_corrected)
                            merged_tags.append(candidate_tags[0])  # Use B-EMAIL from first token
                            i = j  # Skip all processed tokens
                            continue

                # If not part of email, just append normally
                merged_tokens.append(token)
                merged_tags.append(tag)
                i += 1

            tokens = merged_tokens
            tags = merged_tags
            # =============================
            # NEW: RECOMBINE SPLIT NUMBERS, IP ADDRESSES, TIMES, CURRENCY
            # =============================
            # We'll traverse tokens and merge sequences like:
            #   ["12", ".", "500"] → "12.500"
            #   ["192", ".", "168", ".", "1", ".", "100"] → "192.168.1.100"
            #   ["10", ":", "00"] → "10:00"
            #   ["12", ".", "500", "MDL"] → ["12.500", "MDL"] (keep currency separate)
            #
            # Rules:
            # - Only merge if token is digit, '.', ':', or ',' (for decimals)
            # - Do NOT merge if between non-digit tokens (e.g., "word . word")
            # - Keep standalone '.' or ':' as separate if not part of number/time

            merged_tokens = []
            merged_tags = []
            i = 0
            n = len(tokens)

            while i < n:
                token = tokens[i]
                tag = tags[i]

                # If current token is a digit, start checking for possible number/IP/time sequence
                if token.isdigit():
                    sequence = [token]
                    sequence_tags = [tag]
                    j = i + 1

                    # Look ahead for optional separators: '.', ':', ',' followed by digits
                    while j < n:
                        next_token = tokens[j]
                        next_tag = tags[j]

                        # Accept separator only if followed by digit
                        if next_token in ['.', ':', ','] and j + 1 < n and tokens[j + 1].isdigit():
                            sequence.append(next_token)
                            sequence_tags.append(next_tag)
                            j += 1  # skip separator
                        # Stop if next is not digit or separator
                        elif next_token.isdigit():
                            sequence.append(next_token)
                            sequence_tags.append(next_tag)
                            j += 1
                        else:
                            break

                    # If we found a sequence of ≥2 elements (e.g., "12" alone is kept as-is)
                    if len(sequence) > 1:
                        # Merge into single token
                        merged = ''.join(sequence)
                        merged_tokens.append(merged)
                        merged_tags.append(sequence_tags[0])  # Use first tag (B- or I-)
                        i = j  # skip all processed tokens
                    else:
                        # Just one digit — no merge needed
                        merged_tokens.append(token)
                        merged_tags.append(tag)
                        i += 1

                # Special case: handle standalone ":" or "." that might be part of time/decimal
                # But if not preceded/followed by digit, leave as-is
                elif token in ['.', ':', ','] and i > 0 and i < n - 1:
                    prev_is_digit = tokens[i-1].isdigit()
                    next_is_digit = tokens[i+1].isdigit()

                    if prev_is_digit and next_is_digit:
                        # This is part of number/time — will be handled above during digit scan
                        # Skip here — don't process separately
                        i += 1
                        continue
                    else:
                        # Not part of number → keep as separate token
                        merged_tokens.append(token)
                        merged_tags.append(tag)
                        i += 1

                else:
                    # Normal token — keep as-is
                    merged_tokens.append(token)
                    merged_tags.append(tag)
                    i += 1

            tokens = merged_tokens
            tags = merged_tags

            # =============================
            # NEW: Hyphen-aware token merging (handles - : / next to digits)
            # =============================
            punctuation_symbols = ['-', ':', '/', '+']
            merged_tokens = []
            merged_tags = []
            i = 0
            while i < len(tokens):
                current_token = tokens[i]
                current_tag = tags[i]
                
                # Check for punctuation symbols that should merge when next to digits
                if current_token in punctuation_symbols and i < len(tokens) - 1:
                    prev_token = tokens[i-1] if i > 0 else None
                    next_token = tokens[i+1]
                    # Merge if previous or next token is a digit
                    if (prev_token and prev_token.isdigit()) or next_token.isdigit():
                        if i > 0 and merged_tokens:
                            # Merge with previous token in merged_tokens
                            prev_merged_token = merged_tokens.pop()
                            prev_merged_tag = merged_tags.pop()
                            merged = prev_merged_token + current_token + next_token
                            merged_tokens.append(merged)
                            merged_tags.append(prev_merged_tag)
                        else:
                            # No previous token (e.g., "-987654")
                            merged = current_token + next_token
                            merged_tokens.append(merged)
                            merged_tags.append(current_tag)
                        i += 2  # Skip next token
                        continue

                # Handle tokens ending with '-' (e.g., "CH-" followed by "987654")
                if current_token.endswith('-') and i + 1 < len(tokens):
                    merged = current_token + tokens[i+1]
                    merged_tokens.append(merged)
                    merged_tags.append(current_tag)
                    i += 2
                    continue

                # Handle standalone starting hyphens (unlikely but safe)
                if current_token.startswith('-') and i > 0 and not tokens[i-1].endswith('-'):
                    merged_tokens.append(current_token)
                    merged_tags.append(current_tag)
                    i += 1
                    continue

                # Default: add token normally
                merged_tokens.append(current_token)
                merged_tags.append(current_tag)
                i += 1

            tokens = merged_tokens
            tags = merged_tags

            # Validate again after merging
            if not tokens or not tags or len(tokens) != len(tags):
                logging.warning(f"Skipping sentence after merging due to mismatch")
                continue
            
            # =============================
            # NEW: CONT_DIGITAL - Merge branded services with emails (e.g., "PayPal ion.p@gmail.com" → "PayPal:ion.p@gmail.com")
            # =============================
            known_services = {
                'PayPal', 'Skrill', 'Revolut', 'Wise', 'Stripe', 'Google', 'Apple', 'Microsoft',
                'Coinbase', 'TransferWise', 'Bitcoin', 'Ethereum', 'Crypto', 'Paysera', 'Monese',
                'N26', 'Starling', 'Plaid', 'Adyen', 'Square', 'Venmo', 'Zelle', 'CashApp'
            }

            merged_tokens = []
            merged_tags = []
            i = 0
            n = len(tokens)

            while i < n:
                token = tokens[i]
                tag = tags[i]

                # Look for an email token (must contain @ and at least one dot after @)
                if '@' in token and '.' in token.split('@')[1] if len(token.split('@')) > 1 else False:
                    # Check if previous token is a known service (and was NOT part of an email itself)
                    if i > 0:
                        prev_token = tokens[i-1]
                        prev_tag = tags[i-1]

                        # If previous token is a single word, uppercase/capitalized, and in known services
                        if (prev_token in known_services or
                            (prev_token.isalpha() and prev_token[0].isupper() and prev_token.capitalize() in known_services)):
                            
                            # Combine: "PayPal" + "ion.p@gmail.com" → "PayPal:ion.p@gmail.com"
                            combined = f"{prev_token}:{token}"
                            merged_tokens.append(combined)
                            merged_tags.append('B-CONT_DIGITAL')  # Always B- since it's a new composite entity
                            i += 1  # skip current email token
                            continue  # don't add prev_token separately — we've consumed it

                # If not part of a service+email combo, just append normally
                merged_tokens.append(token)
                merged_tags.append(tag)
                i += 1

            tokens = merged_tokens
            tags = merged_tags
            
            # =============================
            # Build text and calculate character offsets
            # =============================
            text = " ".join(tokens)
            char_offsets = []
            current = 0
            for token in tokens:
                start = current
                end = current + len(token)
                char_offsets.append((start, end))
                current = end + 1  # space between tokens

            # Convert BIO tags to entity spans
            entities = []
            current_start = None
            current_label = None

            for i, tag in enumerate(tags):
                if tag == 'O':
                    if current_start is not None:
                        start_char = char_offsets[current_start][0]
                        end_char = char_offsets[i-1][1]
                        entities.append([start_char, end_char, current_label])
                        current_start = None
                        current_label = None
                elif tag.startswith('B-'):
                    if current_start is not None:
                        start_char = char_offsets[current_start][0]
                        end_char = char_offsets[i-1][1]
                        entities.append([start_char, end_char, current_label])
                    current_start = i
                    current_label = tag[2:]
                elif tag.startswith('I-'):
                    pass

            # Handle open entity at end
            if current_start is not None:
                start_char = char_offsets[current_start][0]
                end_char = char_offsets[-1][1]
                entities.append([start_char, end_char, current_label])

            # Write to JSONL
            doc_json = [text, {"entities": entities}]
            out_f.write(json.dumps(doc_json) + '\n')
    
    print(f"Successfully converted {len(data)} sentences to {output_filepath}")

# Run the conversion
bio2_json_to_spacy_jsonl("synthetic_moldova_pii_data.json", "train2.jsonl")