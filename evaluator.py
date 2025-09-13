"""
Evaluator for anonymization clients.

- Loads a dataset in JSONL format (each line is [text, entities_dict])
- Reconstructs raw text and gold entity spans with character offsets
- Runs an anonymization client to get predicted spans via metadata
- Computes micro precision/recall/F1 (optionally span-only with --ignore-labels)
- Verifies deanonymization fidelity (exact text match to original)

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
        """
        tp = 0
        fp = 0
        fn = 0
        deanonym_ok = 0
        total = 0

        for ex in examples:
            total += 1
            text = ex["text"]
            gold = ex["gold_spans"]
            anon_text, metadata = self.client.anonymize(text)

            # Predicted spans expected in metadata["entities"]
            pred_spans = metadata.get("entities", []) if isinstance(metadata, dict) else []

            # Deanonymization check
            try:
                deanon = self.client.deanonymize(anon_text, metadata)
                if deanon == text:
                    deanonym_ok += 1
            except Exception:
                pass

            gold_set = self._to_tuple_set(gold)
            pred_set = self._to_tuple_set(pred_spans)

            tp += len(gold_set & pred_set)
            fp += len(pred_set - gold_set)
            fn += len(gold_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "samples": total,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


def main():
    """CLI to run evaluator using a client implementation.
    """
    # Load data from new JSONL format
    ignore_labels = False
    examples = load_dataset("data/ner_dataset_spacy.jsonl", 500)
    if not examples:
        print("No examples loaded. Check the --data path.")
        return

    # Init client and evaluator
    from anonymizer_template import Anonymizer
    client = Anonymizer(model_path=r"models\spacy_metadata_extraction_model2.0\best_model_epoch4")
    evaluator = Evaluator(client, ignore_labels=ignore_labels)

    # Evaluate
    metrics = evaluator.evaluate(examples)
    print("Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()