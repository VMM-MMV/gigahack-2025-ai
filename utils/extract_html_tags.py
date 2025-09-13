import json
import re

def filter_entries_with_lt(input_path, output_path):
    # Load JSON (list of entries or single entry)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure data is a list (wrap single dict into a list)
    if isinstance(data, dict):
        data = [data]

    # Keep only entries with "<" in tokens
    filtered = [entry for entry in data if any("<" in tok for tok in entry.get("tokens", []))]

    # Save to output JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(filtered)} entries with '<' in tokens to {output_path}")

def save_combined_tokens(input_path, output_path):
    """Save combined tokens as plain text entries in JSON."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    combined_texts = [" ".join(entry.get("tokens", [])) for entry in data]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined_texts, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(combined_texts)} combined token entries to {output_path}")

def extract_contents_inside_tags(input_path, output_path):
    """Extract and save the contents inside <...> tags from tokens."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    extracted = []

    for entry in data:
        text = " ".join(entry.get("tokens", []))
        # Find all things inside <...>
        matches = re.findall(r"<(.*?)>", text)
        if matches:
            matches_str = "".join(matches)
            if len(matches_str) > 25:
                print({
                    "id": entry.get("id"),
                    "inside_tags": matches_str
                })

if __name__ == "__main__":
    # Example usage
    # filter_entries_with_lt("synthetic_moldova_pii_data.json", "output.json")
    extract_contents_inside_tags("errors.json", "combined_tokens.json")
