from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re

from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse


# =========================================================
# APP CONFIGURATION
# =========================================================

app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0"
)


# =========================================================
# LOAD MODEL AND TOKENIZER
# =========================================================

model = T5ForConditionalGeneration.from_pretrained(
    "./saved_summary_model"
)

tokenizer = T5Tokenizer.from_pretrained(
    "./saved_summary_model"
)


# =========================================================
# DEVICE
# =========================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

print(f"Using device: {device}")


# =========================================================
# HTML TEMPLATE
# =========================================================

templates = Jinja2Templates(directory=".")


# =========================================================
# INPUT MODEL
# =========================================================

class DialogueInput(BaseModel):
    dialogue: str


# =========================================================
# DATA CLEANING
# =========================================================

def clean_data(text):

    # Remove line breaks
    text = re.sub(r"[\r\n]+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


# =========================================================
# SUMMARIZATION FUNCTION
# =========================================================

def summarize_dialogue(dialogue: str) -> str:

    # Clean input text
    dialogue = clean_data(dialogue)

    # Tokenize input
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    # Move tensors to device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate summary
    with torch.no_grad():

        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    # Decode generated tokens
    summary = tokenizer.decode(
        targets[0],
        skip_special_tokens=True
    )

    return summary


# =========================================================
# API ENDPOINT
# =========================================================

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):

    summary = summarize_dialogue(
        dialogue_input.dialogue
    )

    return {
        "summary": summary
    }


# =========================================================
# HOME PAGE
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )