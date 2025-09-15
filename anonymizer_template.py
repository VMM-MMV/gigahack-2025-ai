from typing import Tuple, Dict, List, Any, Optional
import spacy

# Replace this with your spaCy model path (local folder path)
MODEL_PATH = "./best_model_epoch10"

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
        try:
            # Try to load the custom model
            self.nlp = spacy.load(model_path or MODEL_PATH)
            print(f"[Anonymizer] Custom spaCy model loaded from {self.nlp.path}")
        except Exception as e:
            print(f"[Anonymizer] Failed to load custom model: {e}")
            print("[Anonymizer] Falling back to basic Romanian model")
            # Fallback to basic Romanian model or create blank model
            try:
                self.nlp = spacy.load("ro_core_news_sm")
                print("[Anonymizer] Loaded ro_core_news_sm as fallback")
            except:
                print("[Anonymizer] Creating blank Romanian model")
                self.nlp = spacy.blank("ro")
                # Add basic NER pipeline if not present
                if "ner" not in self.nlp.pipe_names:
                    ner = self.nlp.add_pipe("ner")
                    print("[Anonymizer] Added basic NER pipeline")

    @staticmethod
    def _map_label(entity_label: str) -> str:
        if not entity_label:
            return "MISC"
        key = entity_label.upper()
        return LABEL_MAP.get(key, key)

    def anonymize(self, text: str) -> Tuple[str, Dict]:
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Sort entities by start position to process in order
        entities = sorted(doc.ents, key=lambda ent: ent.start_char)
        
        # Build anonymized text with placeholders
        parts = []
        cursor = 0
        entities_meta = []
        
        for idx, ent in enumerate(entities, start=1):
            s = ent.start_char
            e = ent.end_char
            label = self._map_label(ent.label_)
            placeholder = f"<{label}_{idx}>"
            
            # Add text before entity
            parts.append(text[cursor:s])
            # Add placeholder
            parts.append(placeholder)
            cursor = e
            
            entities_meta.append({
                "start": s,
                "end": e,
                "label": label,
                "text": text[s:e],
                "replacement": placeholder,
            })
        
        # Add remaining text
        parts.append(text[cursor:])
        anon_text = "".join(parts)
        metadata = {"entities": entities_meta}
        
        return anon_text, metadata

    def deanonymize(self, text: str, metadata: Dict) -> str:
        """
        Reverses the anonymization process by replacing placeholders with original text.
        Uses exact string matching for 100% accuracy.
        """
        entities = metadata.get("entities", [])
        result = text
        
        # Replace placeholders in reverse order to handle nested cases correctly
        for ent in reversed(entities):
            replacement = ent.get("replacement")
            original_text = ent.get("text")
            
            if replacement and original_text:
                # Use exact string replacement (case-sensitive)
                result = result.replace(replacement, original_text, 1)
                
        return result