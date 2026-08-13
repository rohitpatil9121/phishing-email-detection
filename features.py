"""Structural features of an email, measured on the RAW text.

TF-IDF describes what an email says; these describe what it does — how many
links it carries, whether they point at raw IPs or shorteners, who sent it, and
how hard it is shouting. Shared by train.py and app.py so the training-time and
serving-time feature construction can never drift apart.
"""

import re

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
IP_URL_RE = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}", re.I)
SHORTENERS = ("bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly")
FREEMAIL = ("gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "mail.ru")
URGENCY = ("urgent", "immediately", "verify", "suspended", "expire", "expires", "act now",
           "confirm", "password", "account", "click here", "final notice", "within 24 hours",
           "limited time", "security alert", "deactivated", "terminated", "winner", "prize")

META_COLUMNS = [
    "n_urls", "has_ip_url", "has_shortener", "n_urgency_terms", "uppercase_ratio",
    "n_exclamations", "sender_is_freemail", "subject_has_urgency", "text_length",
]


def extract_metadata(text, subject="", sender=""):
    text = str(text or "")
    subject = str(subject or "")
    sender = str(sender or "").lower()
    lowered = (text + " " + subject).lower()
    letters = [c for c in text if c.isalpha()]

    return {
        "n_urls": len(URL_RE.findall(text)),
        "has_ip_url": int(bool(IP_URL_RE.search(text))),
        "has_shortener": int(any(s in lowered for s in SHORTENERS)),
        "n_urgency_terms": sum(lowered.count(term) for term in URGENCY),
        "uppercase_ratio": round(sum(c.isupper() for c in letters) / len(letters), 4) if letters else 0.0,
        "n_exclamations": text.count("!"),
        "sender_is_freemail": int(any(sender.endswith("@" + d) or d in sender for d in FREEMAIL)),
        "subject_has_urgency": int(any(term in subject.lower() for term in URGENCY)),
        "text_length": len(text),
    }
