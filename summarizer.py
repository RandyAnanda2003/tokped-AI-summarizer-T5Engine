from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


# ===============================
# FASTAPI APP
# ===============================
app = FastAPI(
    title="IndoT5 Summarization API",
    description="API untuk ringkasan teks menggunakan IndoT5",
    version="1.0"
)

# ===============================
# LOAD MODEL (HANYA SEKALI)
# ===============================
model_path = r"siRendy/model-akhir-skripsi-bismillah"

model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.config.use_cache = False   # ⬅️ PENTING
model.eval()

print(f"✅ Model loaded on {device}")

# ===============================
# REQUEST SCHEMA
# ===============================
class SummarizeRequest(BaseModel):
    text: str

# ===============================
# CORE LOGIC
# ===============================
def summarize_text(text):
    inputs = tokenizer(
        f"ringkaslah: {text}",
        return_tensors="pt",
        max_length=4000,
        truncation=True
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=300,
            min_length=60,
            num_beams=4,
            do_sample=False,
            no_repeat_ngram_size=2,
            repetition_penalty=1.1,
            length_penalty=1.0,
            early_stopping=True
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ===============================
# API ENDPOINT
# ===============================
@app.post("/summarize")
def summarize(req: SummarizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text tidak boleh kosong")

    summary = summarize_text(req.text)
    return {
        "summary": summary
    }

# ===============================
# OPTIONAL: HEALTH CHECK
# ===============================
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "device": str(device)
    }


