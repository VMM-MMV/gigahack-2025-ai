import json
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from typing import List, Dict, Tuple


def normalize_text(text: str) -> str:
    """Normalize Romanian text (Unicode NFC + lowercase)."""
    return unicodedata.normalize("NFC", text.strip().lower())


def is_outlier(strings, target, threshold=0.5):
    """
    Check if a target string is an outlier compared to a list of strings.
    
    Args:
        strings (list[str]): List of strings.
        target (str): The string to check.
        threshold (float): Minimum average similarity (0-1). 
                           If below -> considered outlier.
    
    Returns:
        tuple[bool, float]: (True if outlier, False otherwise, average similarity score).
    """
    if not strings:
        return True, 0.0  # no comparison possible

    # Normalize Romanian text
    target_norm = normalize_text(target)
    strings_norm = [normalize_text(s) for s in strings]

    similarities = []
    for s in strings_norm:
        sim = SequenceMatcher(None, target_norm, s).ratio()
        similarities.append(sim)

    if not similarities:
        return True, 0.0

    avg_similarity = sum(similarities) / len(similarities)
    return avg_similarity < threshold, avg_similarity


def detect_outliers(label, values, threshold=0.5):
    """
    Detect outliers for a specific label.
    
    Args:
        label (str): The field/label name.
        values (list[tuple]): List of (index, value) pairs.
        threshold (float): Similarity threshold.
    
    Returns:
        list[dict]: Outliers with value + index.
    """
    outliers = []
    values_no_index = [x[1] for x in values]

    for pos, value in values:
        is_out, confidence = is_outlier(values_no_index, value, threshold)
        if is_out:
            outliers.append({
                "value": value,
                "index": pos,
                "similarity": confidence
            })
    return outliers


def extract_and_save_entities_bio2(
    input_json_path: str,
    output_json_path: str,
    remove_duplicates: bool = True
) -> Dict[str, List[Tuple[int, str]]]:
    """
    Extract entities from a BIO2-format JSON file and save them grouped by entity type.

    Args:
        input_json_path (str): Path to the input JSON file in BIO2 format.
        output_json_path (str): Path to save the grouped entities JSON.
        remove_duplicates (bool): Whether to remove duplicate entity values.

    Returns:
        dict: A dictionary mapping entity labels to lists of (index, entity_string) tuples.
    """
    entities_dict = defaultdict(list)

    # --- Load JSON ---
    with open(input_json_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)  # list of dicts in BIO2 format

    # --- Extract entities ---
    for id, record in enumerate(data):
        tokens = record.get("tokens", [])
        ner_tags = record.get("ner_tags", [])
        entity_tokens = []
        entity_label = None

        for token, tag in zip(tokens, ner_tags):
            if tag == "O":
                # Save the previous entity if any
                if entity_tokens:
                    entity_value = " ".join(entity_tokens)
                    entities_dict[entity_label].append((id, entity_value))
                    entity_tokens = []
                    entity_label = None
                continue

            # BIO2: B-XXX starts a new entity, I-XXX continues it
            prefix, label = tag.split("-", 1)
            if prefix == "B":
                if entity_tokens:
                    # Save previous entity
                    entity_value = " ".join(entity_tokens)
                    entities_dict[entity_label].append((id, entity_value))
                entity_tokens = [token]
                entity_label = label
            elif prefix == "I" and label == entity_label:
                entity_tokens.append(token)
            else:
                # Misaligned I-tag: start new entity
                if entity_tokens:
                    entity_value = " ".join(entity_tokens)
                    entities_dict[entity_label].append((id, entity_value))
                entity_tokens = [token]
                entity_label = label

        # Save last entity in the sentence
        if entity_tokens:
            entity_value = " ".join(entity_tokens)
            entities_dict[entity_label].append((id, entity_value))

    # --- Save JSON ---
    with open(output_json_path, "w", encoding="utf-8") as outfile:
        json.dump(entities_dict, outfile, ensure_ascii=False, indent=2)

    # --- Logging ---
    total_entities = sum(len(values) for values in entities_dict.values())
    print(f"✅ Saved grouped entities → {output_json_path}")
    print(f"📌 Entity types: {list(entities_dict.keys())}")
    print(f"📊 Total entities: {total_entities}")

    return entities_dict


def process_all_labels(input_file="grouped_values.json", output_file="simplified_outliers.json", threshold=0.5):
    """Process all labels and detect outliers in dataset."""
    with open(input_file, "r", encoding="utf-8") as f:
        entities = json.load(f)

    all_outliers = {}

    print("🔍 Simplified Outlier Detection")
    print("=" * 50)

    for label in sorted(entities.keys()):
        outliers = detect_outliers(label, entities[label], threshold)
        if outliers:
            all_outliers[label] = outliers
            print(f"⚠️ {label}: {len(outliers)} outliers found")
        else:
            print(f"✅ {label}: No outliers found")

    results = {
        "outliers": all_outliers,
        "summary": {
            "total_outliers": sum(len(v) for v in all_outliers.values()),
            "labels_with_outliers": len(all_outliers),
            "total_labels_checked": len(entities)
        }
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Results saved to {output_file}")
    return results


def remove_outliers_from_dataset(original_json_path, outliers_json_path, output_json_path):
    """
    Remove outlier entries from the original dataset based on outliers file.
    
    Args:
        original_json_path: Path to the original dataset (BIO2 format)
        outliers_json_path: Path to the outliers file
        output_json_path: Path to save the cleaned dataset
    """
    
    # Load the outliers file
    with open(outliers_json_path, "r", encoding="utf-8") as f:
        outliers_data = json.load(f)
    
    # Collect all outlier indices to remove
    outlier_indices = set()
    
    for entity_type, outliers_list in outliers_data["outliers"].items():
        for outlier in outliers_list:
            outlier_indices.add(outlier["index"])
            print(f"Marking for removal - Index: {outlier['index']}, Entity: {entity_type}, Value: '{outlier['value']}'")
    
    # Load the original dataset
    with open(original_json_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    
    print(f"\nOriginal dataset size: {len(original_data)}")
    print(f"Outliers to remove: {len(outlier_indices)}")
    
    # Remove outlier entries
    cleaned_data = []
    removed_count = 0
    
    for i, record in enumerate(original_data):
        if i not in outlier_indices:
            cleaned_data.append(record)
        else:
            removed_count += 1
            print(f"Removed record at index {i}")
    
    print(f"Cleaned dataset size: {len(cleaned_data)}")
    print(f"Actually removed: {removed_count} records")
    
    # Save the cleaned dataset
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    print(f"Cleaned dataset saved to: {output_json_path}")
    
    return cleaned_data


def verify_removal(original_json_path, outliers_json_path, cleaned_json_path):
    """
    Verify that the outliers were properly removed by checking if they still exist in the cleaned dataset.
    """
    
    # Load files
    with open(outliers_json_path, "r", encoding="utf-8") as f:
        outliers_data = json.load(f)
    
    with open(cleaned_json_path, "r", encoding="utf-8") as f:
        cleaned_data = json.load(f)
    
    print("\n=== VERIFICATION ===")
    
    # Re-extract entities from cleaned dataset to verify removal
    entities_dict = defaultdict(list)
    
    for id, record in enumerate(cleaned_data):
        tokens = record.get("tokens", [])
        ner_tags = record.get("ner_tags", [])
        entity_tokens = []
        entity_label = None

        for token, tag in zip(tokens, ner_tags):
            if tag == "O":
                # Save the previous entity if any
                if entity_tokens:
                    entity_value = " ".join(entity_tokens)
                    entities_dict[entity_label].append((id, entity_value))
                    entity_tokens = []
                    entity_label = None
                continue
            
            if tag.startswith("B-"):
                # Save the previous entity if any
                if entity_tokens:
                    entity_value = " ".join(entity_tokens)
                    entities_dict[entity_label].append((id, entity_value))
                
                # Start new entity
                entity_label = tag[2:]  # Remove "B-" prefix
                entity_tokens = [token]
            elif tag.startswith("I-"):
                # Continue current entity
                if entity_label == tag[2:]:  # Same entity type
                    entity_tokens.append(token)
                else:
                    # Entity type mismatch, start new entity
                    if entity_tokens:
                        entity_value = " ".join(entity_tokens)
                        entities_dict[entity_label].append((id, entity_value))
                    
                    entity_label = tag[2:]  # Remove "I-" prefix
                    entity_tokens = [token]
        
        # Don't forget the last entity
        if entity_tokens:
            entity_value = " ".join(entity_tokens)
            entities_dict[entity_label].append((id, entity_value))
    
    # Check if any outlier values still exist
    still_exists = []
    for entity_type, outliers_list in outliers_data["outliers"].items():
        for outlier in outliers_list:
            outlier_value = outlier["value"]
            # Check if this value still exists in the cleaned dataset
            if entity_type in entities_dict:
                for _, value in entities_dict[entity_type]:
                    if value == outlier_value:
                        still_exists.append((entity_type, outlier_value))
                        break
    
    if still_exists:
        print("⚠️  WARNING: Some outlier values still exist in the cleaned dataset:")
        for entity_type, value in still_exists:
            print(f"  - {entity_type}: '{value}'")
    else:
        print("✅ All outlier values have been successfully removed!")


def complete_duplicate_removal_pipeline(
    original_dataset_path="synthetic_moldova_pii_data.json",
    grouped_entities_path="grouped_values.json",
    outliers_path="simplified_outliers.json",
    cleaned_dataset_path="cleaned_dataset.json",
    threshold=0.5
):
    """
    Complete pipeline for duplicate/outlier removal.
    
    Args:
        original_dataset_path: Path to original BIO2 dataset
        grouped_entities_path: Path to save grouped entities
        outliers_path: Path to save detected outliers
        cleaned_dataset_path: Path to save cleaned dataset
        threshold: Similarity threshold for outlier detection
    """
    print("🚀 Starting Complete Duplicate Removal Pipeline")
    print("=" * 60)
    
    # Step 1: Extract and group entities
    print("\n📊 Step 1: Extracting and grouping entities...")
    extract_and_save_entities_bio2(
        original_dataset_path,
        grouped_entities_path
    )
    
    # Step 2: Detect outliers
    print("\n🔍 Step 2: Detecting outliers...")
    process_all_labels(
        grouped_entities_path,
        outliers_path,
        threshold
    )
    
    # Step 3: Remove outliers from original dataset
    print("\n🧹 Step 3: Removing outliers from dataset...")
    remove_outliers_from_dataset(
        original_dataset_path,
        outliers_path,
        cleaned_dataset_path
    )
    
    # Step 4: Verify removal
    print("\n✅ Step 4: Verifying removal...")
    verify_removal(
        original_dataset_path,
        outliers_path,
        cleaned_dataset_path
    )
    
    print("\n🎉 Pipeline completed successfully!")


if __name__ == "__main__":
    # Run the complete pipeline
    complete_duplicate_removal_pipeline()
    
    # Or run individual steps:
    # 
    # # Extract entities
    # extract_and_save_entities_bio2(
    #     "synthetic_moldova_pii_data.json",
    #     "grouped_values.json"
    # )
    # 
    # # Detect outliers
    # process_all_labels()
    # 
    # # Remove outliers
    # remove_outliers_from_dataset(
    #     "synthetic_moldova_pii_data.json",
    #     "simplified_outliers.json",
    #     "cleaned_dataset.json"
    # )
    # 
    # # Verify removal
    # verify_removal(
    #     "synthetic_moldova_pii_data.json",
    #     "simplified_outliers.json",
    #     "cleaned_dataset.json"
    # )