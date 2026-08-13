"""Download a real phishing corpus into data/dataset.csv.

Source: Hugging Face dataset `zefang-liu/phishing-email-dataset` (~18.6k real
messages: legitimate Enron mail plus collected phishing mail). No account or
API key needed.

Run:  python download_data.py
Then: python train.py
"""

from pathlib import Path

import pandas as pd

URL = ("https://huggingface.co/api/datasets/zefang-liu/phishing-email-dataset"
       "/parquet/default/train/0.parquet")
OUT = Path(__file__).parent / "data" / "dataset.csv"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("downloading…")
    df = pd.read_parquet(URL)

    df = df.rename(columns={"Email Text": "text", "Email Type": "type"})
    df["label"] = (df["type"] == "Phishing Email").astype(int)

    # The corpus carries the body only; the app's subject/sender fields stay empty,
    # so those structural signals simply read as absent during training.
    df["subject"] = ""
    df["sender"] = ""

    df = df[["text", "subject", "sender", "label"]].dropna(subset=["text"])
    df = df[df["text"].str.strip().str.len() > 40].drop_duplicates(subset="text")

    df.to_csv(OUT, index=False)
    print(f"\nsaved {OUT}  ({len(df):,} emails)")
    print(df["label"].value_counts().rename({0: "LEGITIMATE", 1: "PHISHING"}).to_string())


if __name__ == "__main__":
    main()
