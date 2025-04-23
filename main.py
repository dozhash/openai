import os
import base64
from fastapi import FastAPI, UploadFile, File
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/smart-correct/")
async def smart_correct(file: UploadFile = File(...)):
    try:
        # Step 1: Read and encode image
        image_data = await file.read()
        base64_image = base64.b64encode(image_data).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{base64_image}"

        # Step 2: Send to GPT-4o for OCR + Grammar correction
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please read the text and check the spelling mistakes and grammar rule. For the misspelled words, give the 3 possible correct words. And, for the grammar rule fedback just be simple and spicific. In general, The feedback should not be perfect for now. Just basic correction as a week trained model."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ]
                }
            ],
        )

        # Step 3: Extract and return corrected text
        reply = response.choices[0].message.content
        return {"correction": reply}

    except Exception as e:
        return {"error": f"Failed to process image with OpenAI: {str(e)}"}
