
from fastapi import FastAPI
from pydantic import BaseModel
from anonymizer_template import Anonymizer

app = FastAPI()
anonymizer = Anonymizer()

class AnonymizeRequest(BaseModel):
    text: str

class DeanonymizeRequest(BaseModel):
    text: str
    metadata: dict

@app.post("/anonymize")
async def anonymize(request: AnonymizeRequest):
    anonymized_text, metadata = anonymizer.anonymize(request.text)
    return {"anonymized_text": anonymized_text, "metadata": metadata}

@app.post("/deanonymize")
async def deanonymize(request: DeanonymizeRequest):
    deanonymized_text = anonymizer.deanonymize(request.text, request.metadata)
    return {"deanonymized_text": deanonymized_text}
