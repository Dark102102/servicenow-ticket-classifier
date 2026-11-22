"""Text cleaning that reproduces the dataset author's pre-processing, so that new tickets
(demo input, hand-written test sets) look like the training data before prediction.

How the rules were derived (see notebooks/02_cleaning_and_splits.ipynb):
- The corpus contains only lowercase a-z and spaces (plus 64 stray '.' / ',').
- Single-letter tokens never occur except "a"; contraction fragments ("don", "isn") never occur.
- 41 scikit-learn stop words never occur (e.g. "i", "my", "in", "will") -> they were removed.
- Some common words were scrubbed during anonymisation ("email", "wifi", "login", ...);
  a few are mapped to in-vocabulary equivalents instead of being lost.
"""
import re

REMOVED_STOPWORDS = {
    "amoungst", "an", "bill", "couldnt", "cry", "eg", "eleven", "else", "fifteen", "fifty",
    "first", "forty", "hasnt", "hereafter", "hereupon", "i", "in", "inc", "ltd", "many", "may",
    "my", "noone", "see", "seeming", "six", "sixty", "so", "thence", "thereafter", "thereby",
    "thereupon", "thick", "three", "un", "whence", "whereafter", "wherein", "whereupon",
    "whither", "will",
    # contraction fragments left after stripping apostrophes
    "don", "isn", "wasn", "didn", "doesn", "hasn", "wouldn", "couldn", "shouldn", "aren",
    "weren", "ll", "ve", "im", "ive", "dont",
}

# Words absent from the training vocabulary -> closest word the models have seen.
SYNONYMS = {
    "email": "mail", "emails": "mail", "wifi": "wireless", "login": "log",
    "logon": "log", "username": "user", "internet": "network",
}

_NON_LETTERS = re.compile(r"[^a-z]+")


def clean_text(text: str) -> str:
    """Lowercase, keep letters only, map unseen synonyms, drop removed stop words and 1-letter tokens."""
    text = str(text).lower().replace("e-mail", "email").replace("wi-fi", "wifi")
    tokens = _NON_LETTERS.sub(" ", text).split()
    tokens = [SYNONYMS.get(t, t) for t in tokens]
    return " ".join(t for t in tokens if t not in REMOVED_STOPWORDS and (len(t) > 1 or t == "a"))
