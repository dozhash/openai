from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from textblob import TextBlob
import spacy
import torch
import language_tool_python
import io

# Load everything once
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
nlp = spacy.load("en_core_web_sm")
tool = language_tool_python.LanguageTool('en-US')

app = FastAPI()

@app.post("/analyze/")
async def analyze(file: UploadFile = None, direct_text: str = Form("")):
    if file:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        pixel_values = processor(images=img, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        corrected_spelling = str(TextBlob(text).correct())
    elif direct_text:
        corrected_spelling = direct_text
    else:
        return {"error": "Please upload an image or enter text."}

    # Grammar checking
    doc = nlp(corrected_spelling)
    feedback = []
    for sent in doc.sents:
        sentence = sent.text.strip()
        matches = tool.check(sentence)
        corrected = language_tool_python.utils.correct(sentence, matches)
        feedback.append({
            "original": sentence,
            "corrected": corrected,
            "issues": [
                {"message": m.message, "suggestions": m.replacements}
                for m in matches
            ]
        })

    return JSONResponse(content={"input": corrected_spelling, "feedback": feedback})
