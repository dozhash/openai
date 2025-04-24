import os
import base64
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_feedback_from_text(text: str) -> str:
    """Generates feedback from raw text using GPT-4o."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"Please read the following text and check the spelling mistakes and grammar rules. For misspelled words, give 3 possible correct options. For grammar feedback, keep it simple and specific. The feedback should be basic and resemble a weakly trained model.\n\n{text}"
            }
        ]
    )
    return response.choices[0].message.content


def generate_feedback_from_image(image_data: bytes) -> str:
    """Generates feedback from an image using GPT-4o with Vision."""
    base64_image = base64.b64encode(image_data).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{base64_image}"

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
        ]
    )
    return response.choices[0].message.content


@app.post("/smart-correct/")
async def smart_correct(
    file: UploadFile = File(None),
    direct_text: str = Form("")
):
    try:
        if file:
            # Validate file type
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Only image files are supported.")

            image_data = await file.read()
            if not image_data:
                raise HTTPException(status_code=400, detail="Empty file uploaded.")

            reply = generate_feedback_from_image(image_data)

        elif direct_text.strip():
            reply = generate_feedback_from_text(direct_text)

        else:
            raise HTTPException(status_code=400, detail="Please provide either an image or text input.")

        return {"correction": reply}

    except HTTPException as he:
        logger.warning(f"Client error: {he.detail}")
        raise he

    except Exception as e:
        logger.error(f"Unexpected server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process input: {str(e)}")
