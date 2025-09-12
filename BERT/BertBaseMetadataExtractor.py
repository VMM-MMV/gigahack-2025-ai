from MetadataExtractionBase import MetadataExtractionBase
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import spacy


class BertBaseMetadataExtraction(MetadataExtractionBase):
    def __init__(self):
        self.id2label_custom = {
            'LABEL_1': 'PERSON',
            'LABEL_2': 'PERSON',
            'LABEL_3': 'ORG',
            'LABEL_4': 'ORG',
            'LABEL_5': 'LOC',
            'LABEL_7': 'ADDRESS',
            'LABEL_8': 'ADDRESS',
            'LABEL_25': 'NUMBER',
            'LABEL_26': 'NUMBER',
            'LABEL_17': 'DATE',
            'LABEL_18': 'DATE',
        }

        tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-ner")
        model = AutoModelForTokenClassification.from_pretrained("dumitrescustefan/bert-base-romanian-ner")

        self.ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
        self.id2label = self.id2label_custom #self.ner_pipeline.model.config.id2label
        # Load Romanian model
        #self.nlp = spacy.load("ro_core_news_sm")
        custom_nlp = spacy.blank("ro")
        custom_nlp.add_pipe("sentencizer")
        self.nlp = custom_nlp

    def merge_bio_entities(predictions, id2label):
        merged_entities = []
        current = None

        for token in predictions:
            label_id = token['entity']
            label = id2label.get(label_id, label_id)
            word = token['word']
            score = float(token['score'])
            start = token['start']
            end = token['end']

            # Skip non-entity tokens
            if label == 'O':
                if current:
                    # Finalize previous entity
                    current['score'] = sum(current['score_list']) / len(current['score_list'])
                    del current['score_list']
                    merged_entities.append(current)
                    current = None
                continue

            # Extract main label type (e.g., PERSON, ORG)
            label_type = label.split('-')[-1]

            # Subword token
            if word.startswith('##'):
                word = word[2:]
                if current:
                    current['word'] += word
                    current['end'] = end
                    current['score_list'].append(score)
                else:
                    # Rare case: subword without starting B-label
                    current = {
                        'entity': label_type,
                        'word': word,
                        'start': start,
                        'end': end,
                        'score_list': [score],
                    }
                continue

            if label.startswith('B-') or current is None or current['entity'] != label_type:
                # Finalize previous entity
                if current:
                    current['score'] = sum(current['score_list']) / len(current['score_list'])
                    del current['score_list']
                    merged_entities.append(current)
                # Start new entity
                current = {
                    'entity': label_type,
                    'word': word,
                    'start': start,
                    'end': end,
                    'score_list': [score],
                }
            else:
                # I- label of the same type → continue
                current['word'] += ' ' + word
                current['end'] = end
                current['score_list'].append(score)

        if current:
            current['score'] = sum(current['score_list']) / len(current['score_list'])
            del current['score_list']
            merged_entities.append(current)

        return merged_entities

    def merge_subwords(predictions, id2label):
        """
        Merge subword tokens (##) and group by entity label.

        Args:
            predictions: list of dicts from model outputs.
            id2label: dict mapping 'LABEL_x' to readable label names.

        Returns:
            List of merged entities: [{"entity": label, "score": avg_score, "word": entity_text, "start": ..., "end": ...}]
        """
        merged_entities = []
        current_entity = None

        for token in predictions:
            label = id2label.get(token['entity'], token['entity'])
            word = token['word']
            score = float(token['score'])  # Ensure Python float
            start = token['start']
            end = token['end']

            if word.startswith('##'):
                word = word[2:]
                if current_entity:
                    # Merge with previous entity
                    current_entity['word'] += word
                    current_entity['end'] = end
                    current_entity['score_list'].append(score)
                else:
                    # Rare but safe fallback
                    current_entity = {
                        'entity': label,
                        'word': word,
                        'start': start,
                        'end': end,
                        'score_list': [score],
                    }
            else:
                # Save the previous entity
                if current_entity:
                    current_entity['score'] = sum(current_entity['score_list']) / len(current_entity['score_list'])
                    del current_entity['score_list']
                    merged_entities.append(current_entity)

                # Start new entity
                current_entity = {
                    'entity': label,
                    'word': word,
                    'start': start,
                    'end': end,
                    'score_list': [score],
                }

        # Append the last entity
        if current_entity:
            current_entity['score'] = sum(current_entity['score_list']) / len(current_entity['score_list'])
            del current_entity['score_list']
            merged_entities.append(current_entity)

        return merged_entities

    def merge_entities(predictions):
        merged_entities = []
        current_entity = predictions[0]
        number_of_merged_entities = 1

        for next_entity in predictions[1:]:
            if current_entity["entity"] == next_entity["entity"] and (
                    current_entity["end"] + 1 == next_entity["start"] or current_entity["end"] == next_entity["start"]):
                if current_entity["end"] + 1 == next_entity["start"]:
                    current_entity["end"] = next_entity["end"]
                    current_entity["word"] = current_entity["word"] + ' ' + next_entity["word"]
                    current_entity["score"] = (current_entity["score"] * number_of_merged_entities + next_entity[
                        "score"]) / (number_of_merged_entities + 1)
                    number_of_merged_entities += 1
                else:
                    current_entity["end"] = next_entity["end"]
                    current_entity["word"] = current_entity["word"] + next_entity["word"]
                    current_entity["score"] = (current_entity["score"] * number_of_merged_entities + next_entity[
                        "score"]) / (number_of_merged_entities + 1)
                    number_of_merged_entities += 1
            else:
                merged_entities.append(current_entity)
                current_entity = next_entity
                number_of_merged_entities = 1

        # append the last entity
        merged_entities.append(current_entity)
        return merged_entities

    def removeUnrelatedWords(list_of_words: list):
        predictions = []
        for word in list_of_words:
            if word['entity'] == 'LABEL_0':
                continue
            predictions.append(word)
        return predictions

    def extract(self,documentText: str):
        # Use spaCy to split sentences
        doc = self.nlp(documentText)
        sentences = [sent.text for sent in doc.sents]

        all_ner_results = []
        for sentence in sentences:
            ner_results = self.ner_pipeline(sentence)
            all_ner_results.extend(ner_results)

        predictions = self.removeUnrelatedWords(all_ner_results)
        predictions = self.merge_subwords(predictions,self.id2label)
        predictions = self.merge_entities(predictions)
        #predictions = merge_bio_entities(predictions,self.id2label)
        for entity in predictions:
            print(f"{entity['entity']} ({entity['score']:.3f}): {entity['word']}")
        return predictions