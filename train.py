"""Train the phishing-email classifier and save it to model.pkl.

Features = TF-IDF over the message text + structural signals (links, IP URLs,
shorteners, urgency words, shouting, free-mail sender) extracted from the RAW
text before cleaning, because cleaning destroys the characters they depend on.

Data source, in order of preference:
  1. data/dataset.csv   (columns: text,label [, subject, sender] -- 0=LEGITIMATE, 1=PHISHING)
  2. built-in sample corpus

Run:  python train.py
"""

import random
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from features import META_COLUMNS, extract_metadata

DATA = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"


# --------------------------------------------------------------------------- data
def build_sample_corpus(n_per_class=600):
    """Templated corpus used when no real dataset is present."""
    random.seed(42)

    legit_subj = ["Q3 planning notes", "Re: Thursday standup", "Invoice #4471 attached",
                  "Team lunch on Friday", "Draft agenda for review", "Onboarding checklist"]
    legit_body = [
        "Hi team, attaching the agenda for Thursday. Please review the draft budget section "
        "before the meeting and send corrections.",
        "Thanks for the update. I have added my comments to the shared document, let me know "
        "if anything is unclear.",
        "The report you requested is attached. Figures are as of last Friday and the summary "
        "is on page two.",
        "Reminder that the sprint review moves to 3pm tomorrow. The room booking has been "
        "updated in the calendar invite.",
        "Please find the signed copy attached. Let me know if accounts needs anything further "
        "from our side.",
    ]
    legit_sender = ["jane.doe@corp.example.com", "accounts@vendor-example.com",
                    "r.patel@university.edu", "no-reply@internal.example.org"]

    phish_subj = ["URGENT: Your account has been suspended", "Action Required: Verify your password",
                  "Security Alert!!! Unusual sign-in", "FINAL NOTICE: Payment failed",
                  "You have won a prize - claim now"]
    phish_body = [
        "Dear Customer, unusual activity was detected. Your account will be SUSPENDED within "
        "24 hours. Verify your password immediately at http://192.168.4.11/secure/login to "
        "avoid permanent closure!!!",
        "Your payment could not be processed. Confirm your bank details now at "
        "http://bit.ly/2xKq9a or your service will be TERMINATED today. Act immediately!",
        "CONGRATULATIONS! You have been selected for a cash prize. Click "
        "http://claim-reward-now.example/login and confirm your identity within 24 hours!!!",
        "Dear user, we could not verify your credentials. Update your password urgently at "
        "http://203.0.113.9/account/update or access will be permanently revoked.",
        "IMPORTANT: your mailbox is full and will be DEACTIVATED. Login here "
        "http://tinyurl.com/mailfix immediately to restore access. Failure to act is final.",
    ]
    phish_sender = ["security-alert@gmail.com", "support@yahoo.com", "billing@hotmail.com",
                    "admin@secure-verify.example"]

    rows = []
    for _ in range(n_per_class):
        rows.append({"subject": random.choice(legit_subj), "sender": random.choice(legit_sender),
                     "text": random.choice(legit_body), "label": 0})
        rows.append({"subject": random.choice(phish_subj), "sender": random.choice(phish_sender),
                     "text": random.choice(phish_body), "label": 1})
    return pd.DataFrame(rows)


def load_dataset():
    single = DATA / "dataset.csv"
    if single.exists():
        df = pd.read_csv(single)
        print(f"loaded data/dataset.csv: {len(df):,} emails")
    else:
        df = build_sample_corpus()
        print(f"no dataset found - using built-in sample corpus: {len(df):,} emails")

    for column in ("subject", "sender"):
        if column not in df.columns:
            df[column] = ""
    df = df[["text", "subject", "sender", "label"]].fillna("")
    df["label"] = df["label"].astype(int)
    # Duplicates land on both sides of the split and turn the test score into a
    # memorisation score, so they go before anything else happens.
    df = df.drop_duplicates(subset=["text", "subject", "sender"])
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def add_metadata(df):
    meta = pd.DataFrame(
        [extract_metadata(r.text, r.subject, r.sender) for r in df.itertuples()],
        index=df.index,
    )
    return pd.concat([df, meta], axis=1)


# --------------------------------------------------------------------------- train
def main():
    df = add_metadata(load_dataset())
    features = ["text"] + META_COLUMNS

    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    model = Pipeline([
        ("features", ColumnTransformer([
            ("text", TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                     stop_words="english", sublinear_tf=True), "text"),
            # Scaled to [0,1] rather than standardised: raw counts would dominate
            # the TF-IDF block numerically.
            ("meta", MinMaxScaler(), META_COLUMNS),
        ])),
        ("clf", LogisticRegression(max_iter=1000, C=5.0)),
    ])
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print(f"\naccuracy: {accuracy:.4f}   false-positive rate: {fpr:.4f}")
    print(classification_report(y_test, preds, target_names=["LEGITIMATE", "PHISHING"]))

    joblib.dump({"model": model, "accuracy": float(accuracy), "fpr": float(fpr),
                 "n_samples": int(len(df))}, MODEL_PATH)
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
