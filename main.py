import os
from fastapi import UploadFile, Form, File
import openai
import io
from fastapi import FastAPI
from PIL import Image


app = FastAPI()


# Set OpenAI key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/smart-correct/")
async def smart_correct(
    file: UploadFile = File(None),
    direct_text: str = Form("")
):
    # Step 1: Extract text from image if image is uploaded
    if file:
        try:
            img = Image.open(io.BytesIO(await file.read())).convert("RGB")
            pixel_values = processor(images=img, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            user_text = str(TextBlob(extracted_text).correct())
        except Exception as e:
            return {"error": f"Failed to process image: {str(e)}"}
    elif direct_text:
        user_text = direct_text
    else:
        return {"error": "Please provide either an image or text input."}

    # Step 2: Send text to OpenAI for correction
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # or "gpt-4"
            messages=[
                {"role": "system", "content": "You are a grammar correction assistant. Return the corrected sentence"},
                {"role": "user", "content": f"Correct :\n\n{user_text}"}
            ],
            temperature=0.4
        )

        reply = response["choices"][0]["message"]["content"]

        return {
            "input": user_text,
            "feedback": reply
        }

    except Exception as e:
        return {"error": f"OpenAI error: {str(e)}"}
