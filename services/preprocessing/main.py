from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import re
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ──────────────────────────────────────────────
# Download NLTK data (once)
# ──────────────────────────────────────────────
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

app = FastAPI(
    title="Preprocessing Service",
    description="IR Text Processing Pipeline",
    version="1.0.0",
)

# ──────────────────────────────────────────────
# INIT (FIX #1 - STOPWORDS CACHED ONCE)
# ──────────────────────────────────────────────
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = set()

# ──────────────────────────────────────────────
# REQUEST MODELS
# ──────────────────────────────────────────────
class PreprocessRequest(BaseModel):
    text: str
    use_stemming: bool = False
    use_lemmatization: bool = True
    remove_stopwords: bool = True
    normalize: bool = True


class PreprocessResponse(BaseModel):
    original: str
    tokens: List[str]
    processed_tokens: List[str]
    processed_text: str


# ──────────────────────────────────────────────
# CORE FIXES
# ──────────────────────────────────────────────

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ──────────────────────────────────────────────
# FIX #2 (CRITICAL): FAST TOKENIZER بدل word_tokenize
# ──────────────────────────────────────────────
def tokenize(text: str) -> List[str]:
    return text.split()
    # أو بديل أدق:
    # return re.findall(r"\b\w+\b", text)


# ──────────────────────────────────────────────
# FIX #3: NO RE-CREATION OF STOPWORDS INSIDE LOOP
# ──────────────────────────────────────────────
def remove_stop_words(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t.lower() not in STOP_WORDS and len(t) > 1]


def apply_stemming(tokens: List[str]) -> List[str]:
    return [stemmer.stem(t) for t in tokens]


# ──────────────────────────────────────────────
# FIX #4 (PERFORMANCE): DISABLE DEFAULT LEMMATIZATION OPTION
# ──────────────────────────────────────────────
def apply_lemmatization(tokens: List[str]) -> List[str]:
    return [lemmatizer.lemmatize(t) for t in tokens]


# ──────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────
def preprocess_single(
    text: str,
    use_stemming: bool = False,
    use_lemmatization: bool = False,  # ⬅️ IMPORTANT FIX (OFF BY DEFAULT)
    remove_stopwords: bool = True,
    normalize: bool = True,
) -> PreprocessResponse:

    original = text

    if normalize:
        text = normalize_text(text)

    tokens = tokenize(text)
    processed = tokens

    if remove_stopwords:
        processed = remove_stop_words(processed)

    if use_stemming:
        processed = apply_stemming(processed)
    elif use_lemmatization:
        processed = apply_lemmatization(processed)

    return PreprocessResponse(
        original=original,
        tokens=tokens,
        processed_tokens=processed,
        processed_text=" ".join(processed),
    )


# ──────────────────────────────────────────────
# API
# ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/preprocess", response_model=PreprocessResponse)
def preprocess(req: PreprocessRequest):
    return preprocess_single(
        text=req.text,
        use_stemming=req.use_stemming,
        use_lemmatization=req.use_lemmatization,
        remove_stopwords=req.remove_stopwords,
        normalize=req.normalize,
    )

@app.post("/preprocess/batch", response_model=dict)
def preprocess_batch(req: dict):
    results = [
        preprocess_single(
            t,
            use_lemmatization=req.get("use_lemmatization", False),
            remove_stopwords=req.get("remove_stopwords", True),
        )
        for t in req["texts"]
    ]
    return {"results": results}
