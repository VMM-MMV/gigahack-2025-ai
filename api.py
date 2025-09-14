from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from anonymizer_template import Anonymizer

app = FastAPI()
anonymizer = Anonymizer()

# ✅ Allow Angular dev server (http://localhost:4200) to call the API
origins = [
    "http://localhost:4200",
    "http://127.0.0.1:4200"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
