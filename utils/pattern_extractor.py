import json
import re

romanian_months = {
    'ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie',
    'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'
}

# ⚠️ ONLY TWO PATTERNS ALLOWED — NO STANDALONE MONTHS
month_list = '|'.join(re.escape(m) for m in romanian_months)

entity_regexes = {
    "NUME_PRENUME": r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$",
    "CNP": r"\b[1-8]\d{12}\b",
    "DATA_NASTERII": r"\b(0[1-9]|[12][0-9]|3[01])\s?[\/.]\s?(0[1-9]|1[0-2])\s?[\/.]\s?(19|20)\d{2}\b",
    # "SEX": r"^(masculin|feminin)$",
    # "NATIONALITATE": r"^(română|moldoveană|român|moldovean)$", nope
    # "LIMBA_VORBITA": r"^(română|rusă|engleză|ucraineană|găgăuză)(?:,\s*(română|rusă|engleză|ucraineană|găgăuză))*$",
    # "ADRESA": r"^str\.?\s+[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\s+\d+(?:[A-Za-z]?)?,\s+[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*$", nope
    # "ADRESA_LUCRU": r"^str\.?\s+[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\s+\d+(?:[A-Za-z]?)?,\s+[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*$", nope
    "TELEFON_MOBIL": r"\b06[789]\d{6}\b",
    "TELEFON_FIX": r"\b022\d{6}\b",
    "EMAIL": "[a-z0-9._%+-]+(?:\s*\.\s*[a-z0-9._%+-]+)*\s*@\s*[a-z0-9.-]+(?:\s*\.\s*[a-z]{2,})",
    "COD_POSTAL": r"\b(MD\s*[-]\s*\d{4})\b",
    # "ORAS_NASTERE": r"^[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*$", nope
    "TARA_NASTERE": r"\b(?:moldova|republica\s+moldova)\b",
    # "PROFESIE": r"^[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*(?:\s+[\w\-]+)*$", nope
    # "ACTIVITATE": r"^(IT|medicină|finanțe|educație|construcții|energie|transport|altul)$", nope
    # "ANGAJATOR": r"^[A-Z][a-zA-Z0-9\s\.\-\&\(\)]+ SA$", nope
    "VENIT": r"\b\d[\d\s.,]*\s*(?:M\s*DL|lei)\b",
    # "STARE_CIVILA": r"^(căsătorit|necăsătorit|divorțat|văduv|văduvă)$", NOPE
    # "EDUCATIE": r"^(superior|mediu|primar|secundar|post-superior)$",  NOPE
    "IBAN": r"MD\d{2}\s*[A-Z]{2}\s*(?:\d\s*){18}",
    # "CONT_BANCAR": r"^\d{10,12}$",  NOPE
    # "CARD_NUMBER": r"^****\d{4}$",
    # "PASAPORT": r"^MD\d{7}$",
    # "BULETIN": r"^\d{10}$",
    # "NUMAR_LICENTA": r"^[A-Z]{3}\d{6}$",
    # "ASIGURARE_MEDICALA": r"^AM\d{9}$",
    # "GRUPA_SANGE": r"^([AOB][+-])$",
    # "ALERGII": r"^[a-zăîâșțĂÎÂȘȚ]+(?:,\s*[a-zăîâșțĂÎÂȘȚ]+)*$",
    # "CONDITII_MEDICALE": r"^[a-zăîâșțĂÎÂȘȚ]+(?:,\s*[a-zăîâșțĂÎÂȘȚ]+)*$",
    "IP_ADDRESS": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\s*\.\s*){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
    "USERNAME": r"\b(?=[a-zA-Z]*_[a-zA-Z0-9]*)(?=[a-zA-Z0-9]*[a-zA-Z])[a-zA-Z0-9_]{3,30}\b",
    "DEVICE_ID": r"\bDEV\s*\d{3}\s*\d{6}\b",
    # "BIOMETRIC": r"^(amprenta\s+digitală|imagine\s+facială|scanare\s+iris)$",
    # "NUMAR_CONTRACT": r"^CNT-[12]\d{3}-\d{6}$",
    # "NUMAR_PLACA": r"^[A-Z]{3}\s+\d{2,3}\s+[A-Z]{2}$",
    # "CONT_DIGITAL": r"^[a-zA-Z]+:\s*[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "WALLET_CRYPTO": r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}(?![a-zA-Z0-9])\b",
    # "NUMAR_CONT_ALT": r"^[a-zA-Z]+:\s*\d+$",
    # "SEGMENT": r"^(mass|afluent|VIP)$",
    # "EXPUS_POLITIC": r"^(DA|NU)$",
    # "STATUT_FATCA": r"^(activ|inactiv)$"
}

if __name__ == "__main__":
    # Compile without VERBOSE — pure simple
    regex = re.compile(entity_regexes["USERNAME"], re.IGNORECASE)

    with open("mock_subset_200.json", "r", encoding="UTF-8") as f:
        raw_data = json.load(f)

    matches = []

    for entry in raw_data:
        if isinstance(entry, dict) and 'tokens' in entry:
            full_text = ' '.join(entry['tokens'])
            print("Searching in text:", full_text)
            match = regex.search(full_text)
            if match:
                print("  Found match:", match.group(0))
                matches.append(match.group(0))

    with open("matches.json", "w", encoding="UTF-8") as out_f:
        json.dump(matches, out_f, indent=2, ensure_ascii=False)

    print(f"Found {len(matches)}  matches. Written to matches.json")