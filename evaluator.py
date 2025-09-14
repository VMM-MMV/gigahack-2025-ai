"""
Evaluator for anonymization clients.

- Loads a dataset in JSONL format (each line is [text, entities_dict])
- Reconstructs raw text and gold entity spans with character offsets
- Runs an anonymization client to get predicted spans via metadata
- Computes micro precision/recall/F1 (optionally span-only with --ignore-labels)
- Verifies deanonymization fidelity (exact text match to original)
- Times each example and reports total/average time

Usage examples:

CLI (default mock client):
  python evaluator.py --data mock_subset_200.json --limit 200

Programmatic:
  from anonymizer_mock import AnonymizerMock
  from evaluator import Evaluator, load_dataset

  client = AnonymizerMock()
  examples = load_dataset("mock_subset_200.json", limit=200)
  evalr = Evaluator(client, ignore_labels=False)
  metrics = evalr.evaluate(examples)
  print(metrics)
"""
from typing import List, Dict, Tuple, Any
import argparse
import json
import time


def load_dataset(path: str, limit: int | None = None) -> List[Dict[str, Any]]:
    """Load a JSONL dataset where each line is [text, entities_dict] and return a list of examples with
    fields: text, gold_spans (list of {start, end, label, text})
    """
    examples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # Each line should be [text, entities_dict]
            if len(data) < 2:
                continue
            text = data[0]
            entities_dict = data[1]
            if "entities" not in entities_dict:
                continue
            gold_spans = []
            for ent in entities_dict["entities"]:
                if len(ent) != 3:
                    continue
                start, end, label = ent
                span_text = text[start:end]
                gold_spans.append({
                    "start": start,
                    "end": end,
                    "label": label,
                    "text": span_text
                })
            examples.append({
                "text": text,
                "gold_spans": gold_spans
            })
            if limit is not None and len(examples) >= limit:
                break
    return examples


class Evaluator:
    def __init__(self, client: Any, ignore_labels: bool = False):
        self.client = client
        self.ignore_labels = ignore_labels

    def _to_tuple_set(self, spans: List[Dict[str, Any]]):
        if self.ignore_labels:
            return {(s["start"], s["end"]) for s in spans}
        return {(s["start"], s["end"], s["label"]) for s in spans}

    def evaluate(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate micro P/R/F1 and deanonymization fidelity.
        examples: list of {text, gold_spans}
        Returns metrics + timing info
        """
        tp = 0
        fp = 0
        fn = 0
        deanonym_ok = 0
        total = 0

        total_time = 0.0
        anonymize_time = 0.0
        deanonymize_time = 0.0
        per_example_times = []

        for ex in examples:
            total += 1
            text = ex["text"]
            gold = ex["gold_spans"]

            start_total = time.perf_counter()

            # Anonymize
            start_anon = time.perf_counter()
            anon_text, metadata = self.client.anonymize(text)
            end_anon = time.perf_counter()
            anonymize_time += end_anon - start_anon

            # Deanonymize
            start_deanon = time.perf_counter()
            try:
                deanon = self.client.deanonymize(anon_text, metadata)
                if deanon == text:
                    deanonym_ok += 1
            except Exception as e:
                # Log if needed: print(f"Deanonymization failed: {e}")
                pass
            end_deanon = time.perf_counter()
            deanonymize_time += end_deanon - start_deanon

            # Predicted spans
            pred_spans = metadata.get("entities", []) if isinstance(metadata, dict) else []

            gold_set = self._to_tuple_set(gold)
            pred_set = self._to_tuple_set(pred_spans)

            tp += len(gold_set & pred_set)
            fp += len(pred_set - gold_set)
            fn += len(gold_set - pred_set)

            end_total = time.perf_counter()
            example_time = end_total - start_total
            per_example_times.append(example_time)
            total_time += example_time

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        avg_time_per_example = total_time / total if total > 0 else 0.0

        return {
            "samples": total,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "deanonymization_success_rate": deanonym_ok / total if total > 0 else 0.0,

            # Timing metrics
            "total_time_seconds": total_time,
            "avg_time_per_example_seconds": avg_time_per_example,
            "anonymize_time_total_seconds": anonymize_time,
            "deanonymize_time_total_seconds": deanonymize_time,
            "anonymize_time_avg_seconds": anonymize_time / total if total > 0 else 0.0,
            "deanonymize_time_avg_seconds": deanonymize_time / total if total > 0 else 0.0,
            "per_example_times_seconds": per_example_times,  # Optional: for detailed analysis
        }


def main():
    """CLI to run evaluator using a client implementation.
    """
    # Load data from new JSONL format
    ignore_labels = False
    examples = load_dataset(r"C:\Users\Huntrese\Documents\github\gigahack-2025-ai\data\ner_dataset_spacy.jsonl", limit=2000)
    if not examples:
        print("No examples loaded. Check the --data path.")
        return

    # Init client and evaluator
    from anonymizer_template import Anonymizer
    client = Anonymizer(model_path=r"C:\Users\Huntrese\Documents\github\gigahack-2025-ai\models\spacy_metadata_extraction_model2.0\best_model_epoch4")
    evaluator = Evaluator(client, ignore_labels=ignore_labels)

    # Evaluate
    print("Starting evaluation...")
    start_eval = time.perf_counter()
    metrics = evaluator.evaluate(examples)
    end_eval = time.perf_counter()

    print("\n=== RESULTS ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, list):  # Skip printing long list of per-example times
            continue
        else:
            print(f"  {k}: {v}")

    print(f"\n=== TIMING SUMMARY ===")
    print(f"Total evaluation time: {end_eval - start_eval:.4f} seconds")
    print(f"Total examples processed: {metrics['samples']}")
    print(f"Average time per example: {metrics['avg_time_per_example_seconds']:.4f} seconds")
    print(f"Anonymization (avg): {metrics['anonymize_time_avg_seconds']:.4f} s")
    print(f"Deanonymization (avg): {metrics['deanonymize_time_avg_seconds']:.4f} s")
    print(f"Deanonymization success rate: {metrics['deanonymization_success_rate']:.2%}")


if __name__ == "__main__":
    main()