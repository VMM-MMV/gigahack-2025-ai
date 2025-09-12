
class MetadataExtractionBase:
    def __init__(self):
        self.nlp = None

    def extract(self,documentText: str):
        print(documentText)
        return {}