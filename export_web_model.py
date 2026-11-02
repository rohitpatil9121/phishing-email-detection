"""Export model.pkl to static/model.json for the in-browser demo (static/demo.js).

The browser cannot unpickle scikit-learn, so this writes out the numbers the
pipeline actually uses: the TF-IDF vocabulary and IDF weights, the MinMaxScaler
parameters for the structural features, and the logistic-regression weights.

Run after train.py:  python export_web_model.py
"""

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from features import META_COLUMNS

BASE = Path(__file__).parent

bundle = joblib.load(BASE / "model.pkl")
model = bundle["model"]
columns = model.named_steps["features"]
vec = columns.named_transformers_["text"]
scaler = columns.named_transformers_["meta"]
clf = model.named_steps["clf"]

assert list(clf.classes_) == [0, 1]
terms = sorted(vec.vocabulary_, key=vec.vocabulary_.get)
n_text = len(terms)
coef = clf.coef_[0]

out = {
    "terms": terms,
    "idf": [round(float(x), 6) for x in vec.idf_],
    "text_coef": [round(float(x), 6) for x in coef[:n_text]],
    "meta_columns": META_COLUMNS,
    "meta_min": [float(x) for x in scaler.min_],
    "meta_scale": [float(x) for x in scaler.scale_],
    "meta_coef": [float(x) for x in coef[n_text:]],
    "intercept": float(clf.intercept_[0]),
    "stop_words": sorted(ENGLISH_STOP_WORDS),
    "info": {
        "accuracy": round(bundle["accuracy"], 4),
        "false_positive_rate": round(bundle["fpr"], 4),
        "training_samples": bundle["n_samples"],
    },
}
path = BASE / "static" / "model.json"
path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB, {n_text} terms)")
