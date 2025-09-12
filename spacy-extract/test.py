import spacy
from spacy.training.example import Example
from utils.io_service import load_data


def run_spacy_evaluation(model_path, data_path):
    nlp = spacy.load(model_path)
    all_data = load_data(data_path)
    evaluate_model_on_data(nlp, all_data)


def evaluate_model_on_data(nlp, test_data):
    examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in test_data]

    scores = nlp.evaluate(examples)

    print("Evaluation Scores (percentages):")
    print(f"Token accuracy: {scores['token_acc'] * 100:.2f}%")
    print(f"Entity precision: {scores['ents_p'] * 100:.2f}%")
    print(f"Entity recall: {scores['ents_r'] * 100:.2f}%")
    print(f"Entity F1-score: {scores['ents_f'] * 100:.2f}%")

    print("\nPer entity type breakdown:")
    for label, metrics in scores["ents_per_type"].items():
        print(
            f"  {label}: "
            f"P={metrics['p'] * 100:.2f}% "
            f"R={metrics['r'] * 100:.2f}% "
            f"F1={metrics['f'] * 100:.2f}%"
        )


if __name__ == "__main__":
    # Example usage
    run_spacy_evaluation("path/to/model", "path/to/data.json")
