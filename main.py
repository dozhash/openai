import os
from fastapi import UploadFile, Form, File, FastAPI
import openai

app = FastAPI()

# Set OpenAI key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/smart-correct/")
async def smart_correct(
    file: UploadFile = File(None),
    direct_text: str = Form("")
):
    user_text = ""

    # Step 1: Use OpenAI Vision to extract text if image is uploaded
    if file:
        try:
            image_data = await file.read()

            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an assistant that extracts text from images and returns it."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract the text from this image."},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data.decode('latin1').encode('base64').decode()}"
                            }},
                        ]
                    }
                ],
                temperature=0.3
            )

            user_text = response["choices"][0]["message"]["content"]

        except Exception as e:
            return {"error": f"Failed to process image with OpenAI: {str(e)}"}

    elif direct_text:
        user_text = direct_text
    else:
        return {"error": "Please provide either an image or text input."}

    # Step 2: Grammar correction
    try:
        correction = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a grammar correction assistant. Return the corrected sentence."},
                {"role": "user", "content": f"Correct this text:\n\n{user_text}"}
            ],
            temperature=0.4
        )

        corrected_text = correction["choices"][0]["message"]["content"]

        return {
            "input": user_text,
            "feedback": corrected_text
        }

    except Exception as e:
        return {"error": f"OpenAI error during grammar correction: {str(e)}"}
