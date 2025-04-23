import os
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import logging

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/smart-correct/")
async def smart_correct(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image files are supported.")

        # Step 1: Read and encode image
        image_data = await file.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        base64_image = base64.b64encode(image_data).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{base64_image}"

        # Step 2: Send to GPT-4o
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

        # Step 3: Return reply
        reply = response.choices[0].message.content
        return {"correction": reply}

    except HTTPException as he:
        logger.warning(f"Client error: {he.detail}")
        raise he  # Let FastAPI return proper status and message

    except Exception as e:
        logger.error(f"Unexpected server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process image with OpenAI: {str(e)}")
