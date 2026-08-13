"""FastAPI backend for the Phishing Email Detection website.

Run:  python -m uvicorn app:app --reload --port 8002
Then: http://127.0.0.1:8002
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import llm
from features import META_COLUMNS, extract_metadata

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model.pkl"
LABELS = {0: "LEGITIMATE", 1: "PHISHING"}

# Human-readable reasons: (threshold the signal must exceed, message template).
# Thresholds keep ordinary text — one link, a single capitalised word — from being
# reported back as a warning sign.
SIGNAL_TEXT = {
    "has_ip_url": (0, "Link points at a raw IP address instead of a domain"),
    "has_shortener": (0, "Uses a URL shortener that hides the real destination"),
    "n_urgency_terms": (1, "Urgency and credential language ({v} matches)"),
    "uppercase_ratio": (0.15, "Heavy use of capitals ({v:.0%} of letters)"),
    "n_exclamations": (2, "Excessive exclamation marks ({v})"),
    "sender_is_freemail": (0, "Sent from a free mail provider, not an organisation domain"),
    "subject_has_urgency": (0, "Subject line pressures the reader to act"),
    "n_urls": (2, "Contains an unusual number of links ({v})"),
}

app = FastAPI(title="Phishing Email Detection API", version="1.0")

if not MODEL_PATH.exists():
    raise SystemExit("model.pkl not found - run `python train.py` first.")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]


class Email(BaseModel):
    text: str = Field(..., min_length=1)
    subject: str = ""
    sender: str = ""
    # "ml" = trained model only, "llm" = Groq only, "both" = run both and compare.
    engine: str = "both"


@app.get("/api/info")
def info():
    return {
        "model": "TF-IDF + structural features + Logistic Regression",
        "accuracy": round(bundle["accuracy"], 4),
        "false_positive_rate": round(bundle["fpr"], 4),
        "training_samples": bundle["n_samples"],
        "classes": list(LABELS.values()),
        "llm_available": llm.available(),
        "llm_model": llm.MODEL,
    }


@app.post("/api/predict")
def predict(email: Email):
    body = email.text.strip()
    if len(body) < 20:
        raise HTTPException(400, "Please paste at least 20 characters of the email body.")

    engine = email.engine if email.engine in ("ml", "llm", "both") else "both"
    ml_result = classify_with_model(body, email.subject, email.sender) if engine in ("ml", "both") else None
    llm_result = llm.classify(body, email.subject, email.sender) if engine in ("llm", "both") else None

    if engine == "llm" and llm_result and "error" in llm_result:
        raise HTTPException(503, llm_result["error"])

    # The headline verdict comes from whichever engine the caller asked to decide;
    # with "both" the trained model decides and the LLM is shown beside it.
    primary = llm_result if engine == "llm" else ml_result
    agreement = None
    if ml_result and llm_result and "error" not in llm_result:
        agreement = ml_result["label"] == llm_result["label"]

    return {
        "label": primary["label"],
        "confidence": primary["confidence"],
        "engine": engine,
        "ml": ml_result,
        "llm": llm_result,
        "agreement": agreement,
    }


def classify_with_model(body, subject="", sender=""):
    meta = extract_metadata(body, subject, sender)
    row = pd.DataFrame([{"text": body, **meta}])[["text"] + META_COLUMNS]

    label = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]

    reasons = [
        template.format(v=meta[key])
        for key, (threshold, template) in SIGNAL_TEXT.items()
        if meta[key] > threshold
    ]

    return {
        "label": LABELS[label],
        "confidence": round(float(proba[label]), 4),
        "probabilities": {"LEGITIMATE": round(float(proba[0]), 4),
                          "PHISHING": round(float(proba[1]), 4)},
        "signals": meta,
        "reasons": reasons,
    }


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")
