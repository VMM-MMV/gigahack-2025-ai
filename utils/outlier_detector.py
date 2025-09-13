import json
import re
from collections import defaultdict


class RealisticOutlierDetector:
    def __init__(self, grouped_entities_path):
        with open(grouped_entities_path, 'r', encoding='utf-8') as f:
            self.entities = json.load(f)

    
    def detect_encoding_corruption(self, label):
        """Detect clear encoding corruption like ÃAA123456"""
        if label not in self.entities:
            return []

        corruption_patterns = [
            r'Ã[€‚ƒ„…†‡ˆ‰Š‹ŒŽ''""•–—˜™š›œžŸ¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿]',
            r'[ÃÂ][A-Za-z]',
            r'�',
            r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]',
        ]

        return [
            value for value in self.entities[label]
            if any(re.search(pattern, str(value)) for pattern in corruption_patterns)
        ]

    @staticmethod
    def is_valid_moldovan_iban(value_str: str) -> bool:
        """Check if value is a valid Moldovan IBAN."""
        pattern = r'^(?:IBAN\s*)?MD\d{2}[A-Z]{2}\d{20}$'
        return bool(re.fullmatch(pattern, value_str.strip()))

    def _check_car_indicators(self, value_str):
        car_indicators = ['din', 'model', 'an', 'volkswagen', 'audi', 'bmw', 'mercedes', 'ford', 'toyota']
        return any(ind.lower() in value_str.lower() for ind in car_indicators)

    def _check_wallet_address(self, value_str):
        return '@' in value_str and '.' in value_str

    def _check_employer_address(self, value_str):
        address_patterns = [r'^bd\.\s', r'^str\.\s', r'^\d+.*,\s*Chișinău$']
        return any(re.search(pattern, value_str) for pattern in address_patterns)

    def _check_phone_number(self, value_str):
        digits_only = re.sub(r'\D', '', value_str)
        return len(digits_only) < 6

    def _check_iban(self, value_str):
        if not self.is_valid_moldovan_iban(value_str):
            return True
        if re.search(r'M D\d{2}', value_str):
            return True
        if re.search(r'MD\d{2}[A-Z]{2}\d+\s+\d+', value_str):
            return True
        return False

    def _check_cnp(self, value_str):
        digits_only = re.sub(r'\D', '', value_str)
        return len(digits_only) > 13 or re.search(r'\d+\s+\d+', value_str)

    def _check_email(self, value_str):
        return re.search(r'@.*\s.*\.com', value_str) is not None

    def detect_wrong_data_type(self, label):
        """Detect completely wrong data types in specific fields"""
        if label not in self.entities:
            return []

        outliers = []
        for value in self.entities[label]:
            value_str = str(value).strip()

            if label == 'NUMAR_PLACA' and self._check_car_indicators(value_str):
                outliers.append(value)
            elif label == 'WALLET_CRYPTO' and self._check_wallet_address(value_str):
                outliers.append(value)
            elif label == 'ANGAJATOR' and self._check_employer_address(value_str):
                outliers.append(value)
            elif label in ['TELEFON_MOBIL', 'TELEFON_FIX'] and self._check_phone_number(value_str):
                outliers.append(value)
            elif label == 'IBAN' and self._check_iban(value_str):
                outliers.append(value)
            elif label == 'CNP' and self._check_cnp(value_str):
                outliers.append(value)
            elif label == 'EMAIL' and self._check_email(value_str):
                outliers.append(value)

        return outliers

    def detect_obvious_data_leakage(self, label):
        """Detect obvious data leakage between fields"""
        if label not in self.entities:
            return []

        return [
            value for value in self.entities[label]
            if label == 'CONT_BANCAR' and re.match(r'^IBAN\s+MD\d{2}[A-Z]{2}\d{18}', str(value).strip())
        ]

    def detect_technical_errors(self, label):
        """Detect technical errors like escaped characters or incomplete data"""
        if label not in self.entities:
            return []

        outliers = []
        structured_fields = ['IBAN', 'CNP', 'TELEFON_MOBIL', 'TELEFON_FIX', 'PASAPORT']

        for value in self.entities[label]:
            value_str = str(value)

            if any(err in value_str for err in ['\\n', '\\t', '\\r']):
                outliers.append(value)
            elif label in structured_fields and '  ' in value_str:
                outliers.append(value)
            elif label in ['TELEFON_FIX', 'TELEFON_MOBIL'] and re.search(r'022l\d+', value_str):
                outliers.append(value)

        return outliers

    def process_all_labels_realistic(self, output_file="realistic_outliers.json"):
        """Process all labels but only flag genuinely broken data"""
        all_outliers, total_outliers = {}, 0

        print("🔍 Detecting ONLY genuinely broken data...")
        print("=" * 50)

        for label in self.entities.keys():
            label_outliers = set()
            label_outliers.update(self.detect_encoding_corruption(label))
            label_outliers.update(self.detect_wrong_data_type(label))
            label_outliers.update(self.detect_obvious_data_leakage(label))
            label_outliers.update(self.detect_technical_errors(label))

            if label_outliers:
                all_outliers[label] = sorted(label_outliers)
                total_outliers += len(label_outliers)
                print(f"🚨 {label}: {len(label_outliers)} broken entries")
                for outlier in sorted(label_outliers):
                    print(f"   • '{outlier}'")
            else:
                print(f"✅ {label}: No broken data found")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_outliers, f, ensure_ascii=False, indent=2)

        print(f"\n💾 Results saved to {output_file}")
        print(f"📊 Total broken entries found: {total_outliers}")
        print(f"🏷️ Labels with broken data: {len(all_outliers)}")

        return all_outliers

    def show_breakdown(self, outliers):
        """Show breakdown of what types of errors were found"""
        print("\n" + "=" * 60)
        print("🔬 ERROR TYPE BREAKDOWN")
        print("=" * 60)

        error_types = defaultdict(list)

        for label, values in outliers.items():
            for value in values:
                value_str = str(value)
                if re.search(r'Ã[€‚ƒ„…†‡ˆ‰Š‹ŒŽ''""•–—˜™š›œžŸ]|�', value_str):
                    error_types['Encoding Corruption'].append(f"{label}: {value}")
                elif any(word in value_str.lower() for word in ['din', 'model', 'volkswagen', 'bmw', 'audi']):
                    error_types['Wrong Data Type'].append(f"{label}: {value}")
                elif '@' in value_str and label != 'EMAIL':
                    error_types['Data Leakage'].append(f"{label}: {value}")
                elif '\\n' in value_str or 'l' in value_str.replace('gmail', '').replace('email', ''):
                    error_types['Technical Errors'].append(f"{label}: {value}")
                else:
                    error_types['Format Issues'].append(f"{label}: {value}")

        for error_type, examples in error_types.items():
            print(f"\n{error_type}: {len(examples)} cases")
            for example in examples[:5]:
                print(f"   • {example}")
            if len(examples) > 5:
                print(f"   • ... and {len(examples) - 5} more")


def main():
    detector = RealisticOutlierDetector("grouped_entities.json")

    print("🎯 Realistic Outlier Detector")
    print("Only flagging genuinely broken data, not valid variations")

    outliers = detector.process_all_labels_realistic("realistic_outliers.json")
    detector.show_breakdown(outliers)

    if outliers:
        print("\n⚠️  These entries need manual review and likely correction.")
    else:
        print("\n🎉 No genuinely broken data found! Your dataset looks clean.")

    print("\n✨ Check 'realistic_outliers.json' for complete results.")


if __name__ == "__main__":
    main()
