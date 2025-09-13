from typing import Tuple, Dict, List, Any, Optional
import spacy
import os

# Replace this with your spaCy model path (local folder path)
MODEL_PATH = "models\model_899"  # Set to your spaCy model path

# Label map EXACTLY matching the Moldova-specific PII labels (do not change)
LABEL_MAP = {
    # Core Identity
    "NUME_PRENUME": "NUME_PRENUME",
    "CNP": "CNP",
    "DATA_NASTERII": "DATA_NASTERII",
    "SEX": "SEX",
    "NATIONALITATE": "NATIONALITATE",
    "LIMBA_VORBITA": "LIMBA_VORBITA",

    # Contact Information
    "ADRESA": "ADRESA",
    "ADRESA_LUCRU": "ADRESA_LUCRU",
    "TELEFON_MOBIL": "TELEFON_MOBIL",
    "TELEFON_FIX": "TELEFON_FIX",
    "EMAIL": "EMAIL",
    "COD_POSTAL": "COD_POSTAL",

    # Location & Origin
    "ORAS_NASTERE": "ORAS_NASTERE",
    "TARA_NASTERE": "TARA_NASTERE",

    # Professional Information
    "PROFESIE": "PROFESIE",
    "ACTIVITATE": "ACTIVITATE",
    "ANGAJATOR": "ANGAJATOR",
    "VENIT": "VENIT",

    # Personal Status
    "STARE_CIVILA": "STARE_CIVILA",
    "EDUCATIE": "EDUCATIE",

    # Financial Information
    "IBAN": "IBAN",
    "CONT_BANCAR": "CONT_BANCAR",
    "CARD_NUMBER": "CARD_NUMBER",

    # Identity Documents
    "PASAPORT": "PASAPORT",
    "BULETIN": "BULETIN",
    "NUMAR_LICENTA": "NUMAR_LICENTA",

    # Medical Information
    "ASIGURARE_MEDICALA": "ASIGURARE_MEDICALA",
    "GRUPA_SANGE": "GRUPA_SANGE",
    "ALERGII": "ALERGII",
    "CONDITII_MEDICALE": "CONDITII_MEDICALE",

    # Digital & Technology
    "IP_ADDRESS": "IP_ADDRESS",
    "USERNAME": "USERNAME",
    "DEVICE_ID": "DEVICE_ID",
    "BIOMETRIC": "BIOMETRIC",

    # Additional Financial & Legal
    "NUMAR_CONTRACT": "NUMAR_CONTRACT",
    "NUMAR_PLACA": "NUMAR_PLACA",
    "CONT_DIGITAL": "CONT_DIGITAL",
    "WALLET_CRYPTO": "WALLET_CRYPTO",
    "NUMAR_CONT_ALT": "NUMAR_CONT_ALT",

    # Other
    "SEGMENT": "SEGMENT",
    "EXPUS_POLITIC": "EXPUS_POLITIC",
    "STATUT_FATCA": "STATUT_FATCA",
}


class Anonymizer:
    def __init__(self, model_path: Optional[str] = None):
        # Load spaCy model directly
        self.nlp = spacy.load(model_path or MODEL_PATH)
        print("[Anonymizer] spaCy model loaded successfully")

    @staticmethod
    def _map_label(entity_label: str) -> str:
        if not entity_label:
            return "MISC"
        key = entity_label.upper()
        return LABEL_MAP.get(key, key)

    def anonymize(self, text: str) -> Tuple[str, Dict]:
        # Process text with spaCy
        doc = self.nlp(text)
        predicted_spans = []
        
        for ent in doc.ents:
            s = ent.start_char
            e = ent.end_char
            label = self._map_label(ent.label_)
            predicted_spans.append({
                "start": s,
                "end": e,
                "label": label,
                "text": ent.text,
            })

        predicted_spans.sort(key=lambda s: s["start"])  # Sort by start position

        # Build anonymized text with placeholders
        parts = []
        cursor = 0
        entities_meta = []
        for idx, span in enumerate(predicted_spans, start=1):
            s, e, label = span["start"], span["end"], span["label"]
            placeholder = f"<{label}_{idx}>"
            parts.append(text[cursor:s])
            parts.append(placeholder)
            cursor = e
            entities_meta.append({
                "start": s,
                "end": e,
                "label": label,
                "text": text[s:e],
                "replacement": placeholder,
            })
        parts.append(text[cursor:])
        anon_text = "".join(parts)
        metadata = {"entities": entities_meta}
        return anon_text, metadata

    def deanonymize(self, text: str, metadata: Dict) -> str:
        # Ensure metadata has entities
        entities = metadata.get("entities", [])
        result = text
        # Process in reverse order to avoid replacement issues
        for ent in reversed(entities):
            result = result.replace(ent["replacement"], ent["text"], 1)
        return result