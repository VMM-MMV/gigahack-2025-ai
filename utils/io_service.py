import json

def load_data(train_data_path):
    all_data = []
    with open(train_data_path, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            all_data.append(example)
    return all_data