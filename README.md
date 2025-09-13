## Evaluator
Evaluator.py will be used to evaluate your model.
Make sure the code is compatible with it.

There are some examples
anonymizer_mock.py - a fake anonymizer example, it uses train data to output 95% accuracy.
anonymizer_ronec.py - uses original ronec pretrained model (but different labels)
anonymizer_template.py - use is a template for your code.


## Dataset Characteristics:

- **Language**: Romanian (Moldova dialect)
- **Format**: JSON with RONEC-compatible structure
- **Sentence Length**: 80-165 tokens (complex, multi-clause sentences)
- **Entity Density**: 8-15 PII entities per sentence
- **Domains**: 8 cross-domain scenarios (32 specific contexts)
- **Generation Method**: OpenAI GPT-4 with concurrent processing

PII Entity Types (42 Total)

Core Identity Entities
| Entity | Description | Example | Frequency |
|--------|-------------|---------|-----------|
| `NUME_PRENUME` | Full name (Romanian/Moldovan) | Ion Popescu | 95% |
| `CNP` | Romanian/Moldovan Personal Numeric Code | 2850315123456 | 85% |
| `DATA_NASTERII` | Date of birth | 15.03.1985 | 40% |
| `SEX` | Gender | masculin/feminin | 25% |
| `NATIONALITATE` | Nationality | moldoveană, română | 35% |
| `LIMBA_VORBITA` | Spoken language | română, rusă | 15% |

Contact Information
| Entity | Description | Example | Frequency |
|--------|-------------|---------|-----------|
| `ADRESA` | Home address | str. Ștefan cel Mare 45, Chișinău | 70% |
| `ADRESA_LUCRU` | Work address | bd. Dacia 12, Chișinău | 35% |
| `TELEFON_MOBIL` | Mobile phone number | 069123456 | 80% |
| `TELEFON_FIX` | Landline phone number | 022123456 | 30% |
| `EMAIL` | Email address | ion.popescu@gmail.com | 65% |
| `COD_POSTAL` | Postal code | MD-2001 | 45% |

Location & Origin
| Entity | Description | Example | Frequency |
|--------|-------------|---------|-----------|
| `ORAS_NASTERE` | Place of birth | Chișinău, Bălți | 50% |
| `TARA_NASTERE` | Country of birth | Moldova, România | 35% |

Professional Information
| Entity | Description | Example | Frequency |
|--------|-------------|---------|-----------|
| `PROFESIE` | Specific profession | inginer software, medic cardiolog | 60% |
| `ACTIVITATE` | General activity field | IT, medicină | 50% |
| `ANGAJATOR` | Employer name | Moldtelecom SA | 45% |
| `VENIT` | Monthly income | 15000 MDL | 30% |

Personal Status
| Entity | Description | Example | Frequency |
|--------|-------------|---------|-----------|
| `STARE_CIVILA` | Marital status | căsătorit/necăsătorit/divorțat | 25% |
| `EDUCATIE` | Education level | superior, mediu | 35% |

Financial Information
| Entity | Description | Example | Frequency |
|--------|-------------|---------|-----------|
| `IBAN` | International Bank Account Number | MD24AG000000225100013104 | 55% |
| `CONT_BANCAR` | Local bank account number | 225100013104 | 40% |
| `CARD_NUMBER` | Masked credit/debit card number | ****1234 | 35% |

Identity Documents
| Entity | Description | Example | Frequency |
|--------|-------------|---------|-----------|
| `PASAPORT` | Passport number | MD1234567 | 30% |
| `BULETIN` | Identity card serial and number | 0123456789 | 45% |
| `NUMAR_LICENTA` | License number (driving, professional) | AAA123456 | 25% |

Medical Information
| Entity | Description | Example | Frequency |
|--------|-------------|---------|-----------|
| `ASIGURARE_MEDICALA` | Health insurance policy number | AM1234567890 | 20% |
| `GRUPA_SANGE` | Blood type | A+, B-, O+ | 10% |
| `ALERGII` | Medical allergies | polen, medicamente | 15% |
| `CONDITII_MEDICALE` | Medical conditions | diabet, hipertensiune | 12% |

Digital & Technology
| Entity | Description | Example | Frequency |
|--------|-------------|---------|-----------|
| `IP_ADDRESS` | IP address | 192.168.1.100 | 8% |
| `USERNAME` | Username/handle | ion_popescu | 20% |
| `DEVICE_ID` | Device identifier | DEV123456789 | 5% |
| `BIOMETRIC` | Biometric data reference | amprenta digitală | 3% |

Additional Financial & Legal
| Entity | Description | Example | Frequency |
|--------|-------------|---------|-----------|
| `NUMAR_CONTRACT` | Contract number | CNT-2024-001234 | 25% |
| `NUMAR_PLACA` | License plate number | CHI 123 AB | 15% |
| `CONT_DIGITAL` | Digital wallet account | PayPal: ion.p@gmail.com | 18% |
| `WALLET_CRYPTO` | Cryptocurrency wallet address | 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa | 8% |
| `NUMAR_CONT_ALT` | Other account numbers | util: 123456789 | 20% |

Legacy/Internal Classifications
| Entity | Description | Example | Frequency |
|--------|-------------|---------|-----------|
| `SEGMENT` | Customer segment | mass/afluent/VIP | 30% |
| `EXPUS_POLITIC` | Politically Exposed Person status | DA/NU | 20% |
| `STATUT_FATCA` | FATCA compliance status | activ/inactiv | 15% |

Dataset Structure
Each sample follows the RONEC-compatible format:
```json
{
  "id": 1,
  "tokens": ["Domnul", "Ion", "Popescu", ",", "cu", "CNP", "2850315123456", "..."],
  "ner_tags": ["O", "B-NUME_PRENUME", "I-NUME_PRENUME", "O", "O", "O", "B-CNP", "..."],
  "ner_ids": [0, 1, 2, 0, 0, 0, 3, ...],
  "space_after": [true, true, false, true, true, true, false, ...],
  "generation_method": "concurrent_openai"
}
```

Samples sentences: 100000
Total PII entities: 1211216
Entity distribution:
      "TELEFON_MOBIL": 79504,
      "NUME_PRENUME": 97725,
      "STARE_CIVILA": 23541,
      "ANGAJATOR": 48511,
      "BULETIN": 42072,
      "ADRESA": 73114,
      "ORAS_NASTERE": 54804,
      "STATUT_FATCA": 16257,
      "PROFESIE": 65630,
      "DATA_NASTERII": 48475,
      "CNP": 82130,
      "EMAIL": 63256,
      "NUMAR_CONTRACT": 25175,
      "NATIONALITATE": 32630,
      "VENIT": 34006,
      "ASIGURARE_MEDICALA": 17824,
      "CONDITII_MEDICALE": 7505,
      "ADRESA_LUCRU": 31031,
      "EXPUS_POLITIC": 13141,
      "TARA_NASTERE": 20937,
      "EDUCATIE": 26521,
      "IBAN": 47437,
      "TELEFON_FIX": 26031,
      "CONT_DIGITAL": 12426,
      "SEGMENT": 22091,
      "CARD_NUMBER": 26166,
      "WALLET_CRYPTO": 3812,
      "CONT_BANCAR": 26460,
      "GRUPA_SANGE": 6551,
      "LIMBA_VORBITA": 13802,
      "PASAPORT": 19506,
      "NUMAR_LICENTA": 11429,
      "COD_POSTAL": 23824,
      "IP_ADDRESS": 6085,
      "SEX": 11436,
      "USERNAME": 12950,
      "ACTIVITATE": 11002,
      "BIOMETRIC": 2260,
      "ALERGII": 11425,
      "NUMAR_PLACA": 7339,
      "DEVICE_ID": 3567,
      "NUMAR_CONT_ALT": 1828
