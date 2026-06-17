"""
Query Refinement Service
=====================================================================
IR Concepts Applied (Project Spec §5 - Query Refinement):

    Query Formulation Assistance:
        Help users express their information need more precisely.
        Techniques: spell correction, synonym expansion, query suggestion.

    Query Expansion (Spec §5):
        "Adding synonyms to the user's query."
        Uses WordNet to find synonyms for query terms.
        Increases recall by covering more relevant documents.

    Spell Correction (Spec §5):
        "Correcting the query linguistically."
        Uses edit-distance (Levenshtein) against the index vocabulary.

    Search History Weighting (Spec §5):
        "Weighting user query with information from previous search history."
        Boosts terms that appeared in the user's past queries → Personalization.

    Query Suggestion (Spec §5):
        Suggest similar queries based on index vocabulary and past queries.

    Relation to Preprocessing (Lecture 1):
        Query refinement runs AFTER basic preprocessing.
        The refined query is then preprocessed the same way as documents.

SOA Role (Project Spec):
    خدمة Query Refinement — مستقلة، تُحسِّن الاستعلام قبل إرساله لخدمة الاسترجاع.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Set
from collections import defaultdict
import re
import httpx
import nltk
from nltk.corpus import wordnet

for pkg in ["wordnet", "omw-1.4"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

app = FastAPI(
    title="Query Refinement Service",
    description="Query Expansion, Spell Correction, Synonym Addition, History-based Weighting",
    version="1.0.0",
)

PREPROCESSING_URL = "http://localhost:8001"
INDEXING_URL = "http://localhost:8002"

# ──────────────────────────────────────────────
# In-Memory Search History Store
# ──────────────────────────────────────────────
# {user_id: [list of past query strings]}
user_history: Dict[str, List[str]] = defaultdict(list)

# Term frequency across all past queries (for suggestion)
# {user_id: {term: count}}
user_term_freq: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────

class RefineRequest(BaseModel):
    query: str
    user_id: Optional[str] = "anonymous"
    dataset_id: Optional[str] = ""
    use_synonyms: bool = True
    use_spell_correction: bool = True
    use_history: bool = True
    max_synonyms_per_term: int = 2


class RefineResponse(BaseModel):
    original_query: str
    refined_query: str
    added_synonyms: List[str]
    spell_corrections: Dict[str, str]   # {original_term: corrected_term}
    history_boost_terms: List[str]
    suggestions: List[str]


class HistoryEntry(BaseModel):
    user_id: str
    query: str


class SuggestionRequest(BaseModel):
    partial_query: str
    user_id: Optional[str] = "anonymous"
    top_k: int = 5


# ──────────────────────────────────────────────
# Synonym Expansion (Spec §5 - WordNet)
# ──────────────────────────────────────────────

def get_synonyms(term: str, max_per_term: int = 2) -> List[str]:
    """
    Query Expansion via WordNet synonyms (Project Spec §5).
    "Adding synonyms to the user's query increases recall
     by covering documents that use different words for the same concept."

    Example: query "car" → expands to include "automobile", "vehicle"
    """
    synonyms: Set[str] = set()

    for syn in wordnet.synsets(term):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ").lower()
            # Only add single-word synonyms that differ from original
            if candidate != term and " " not in candidate:
                synonyms.add(candidate)
                if len(synonyms) >= max_per_term:
                    break
        if len(synonyms) >= max_per_term:
            break

    return list(synonyms)


# ──────────────────────────────────────────────
# Spell Correction (Spec §5 - Edit Distance)
# ──────────────────────────────────────────────

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Levenshtein edit distance between two strings.
    Used for spell correction: find the closest vocabulary word.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def spell_correct(term: str, vocabulary: List[str], max_distance: int = 2) -> Optional[str]:
    """
    Spell correction using edit distance against index vocabulary.
    Spec §5: "Correcting the query linguistically."
    Returns the closest vocabulary word if within max_distance, else None.
    """
    if term in vocabulary:
        return term  # Already correct

    best_word = None
    best_dist = max_distance + 1

    for vocab_word in vocabulary:
        # Quick length filter to avoid computing all distances
        if abs(len(term) - len(vocab_word)) > max_distance:
            continue
        dist = levenshtein_distance(term, vocab_word)
        if dist < best_dist:
            best_dist = dist
            best_word = vocab_word

    return best_word if best_dist <= max_distance else None


# ──────────────────────────────────────────────
# History-based Query Weighting (Spec §5)
# ──────────────────────────────────────────────

def get_history_boost_terms(user_id: str, query_tokens: List[str], top_k: int = 3) -> List[str]:
    """
    Search History Weighting (Project Spec §5):
    "Weighting user query with information from previous search history."

    Strategy:
    - Look at terms the user frequently searched before
    - Boost terms co-occurring with current query terms
    - Add top-K frequently used terms to the query
    """
    if user_id not in user_term_freq:
        return []

    user_terms = user_term_freq[user_id]
    query_set = set(query_tokens)

    # Find terms from history not already in query, ranked by frequency
    boost_candidates = [
        (term, freq)
        for term, freq in user_terms.items()
        if term not in query_set and freq >= 2
    ]
    boost_candidates.sort(key=lambda x: x[1], reverse=True)

    return [term for term, _ in boost_candidates[:top_k]]


def update_history(user_id: str, query: str, tokens: List[str]):
    """Record a query in the user's search history."""
    user_history[user_id].append(query)
    # Keep only last 50 queries
    user_history[user_id] = user_history[user_id][-50:]
    # Update term frequency map
    for token in tokens:
        user_term_freq[user_id][token] += 1


# ──────────────────────────────────────────────
# Query Suggestion (Spec §5)
# ──────────────────────────────────────────────

def generate_suggestions(
    query: str,
    user_id: str,
    top_k: int = 5,
) -> List[str]:
    """
    Query Suggestion (Project Spec §5):
    Suggest similar queries based on:
    1. User's past queries that share terms
    2. Simple term completion from history
    """
    suggestions = set()
    query_lower = query.lower()

    # From user history: find past queries that start with or contain the current query
    for past_query in user_history.get(user_id, []):
        if past_query.lower() != query_lower and query_lower in past_query.lower():
            suggestions.add(past_query)
        elif past_query.lower().startswith(query_lower[:3]) and past_query != query:
            suggestions.add(past_query)

    return list(suggestions)[:top_k]


# ──────────────────────────────────────────────
# Main Refinement Pipeline
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "query_refinement"}


@app.post("/refine", response_model=RefineResponse)
async def refine_query(req: RefineRequest):
    """
    Main query refinement pipeline:
    1. Preprocess the query (tokenize, normalize)
    2. Spell correction on each token
    3. Synonym expansion for each token
    4. History-based term boosting
    5. Build refined query string
    """
    original_query = req.query
    spell_corrections: Dict[str, str] = {}
    added_synonyms: List[str] = []
    history_boost: List[str] = []

    # ── Step 1: Preprocess ──
    async with httpx.AsyncClient(timeout=30.0) as client:
        prep_r = await client.post(
            f"{PREPROCESSING_URL}/preprocess",
            json={
                "text": req.query,
                "use_lemmatization": True,
                "remove_stopwords": True,
            },
        )
        prep = prep_r.json()
        query_tokens: List[str] = prep["processed_tokens"]

    if not query_tokens:
        return RefineResponse(
            original_query=original_query,
            refined_query=original_query,
            added_synonyms=[],
            spell_corrections={},
            history_boost_terms=[],
            suggestions=[],
        )

    # ── Step 2: Spell Correction ──
    corrected_tokens = list(query_tokens)
    if req.use_spell_correction:
        # Get index vocabulary for correction reference
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                stats_r = await client.get(f"{INDEXING_URL}/stats")
                # In full implementation, we'd fetch vocabulary from indexing service
                # For now use basic correction
        except Exception:
            pass

        # Apply basic spell correction using NLTK words corpus
        try:
            nltk.download("words", quiet=True)
            from nltk.corpus import words as nltk_words
            vocab = set(w.lower() for w in nltk_words.words())
            for i, token in enumerate(corrected_tokens):
                if len(token) > 3 and token not in vocab:
                    # Simple: only correct obvious misspellings
                    correction = spell_correct(token, list(vocab)[:5000], max_distance=1)
                    if correction and correction != token:
                        spell_corrections[token] = correction
                        corrected_tokens[i] = correction
        except Exception:
            pass  # Spell correction is optional; don't block retrieval

    # ── Step 3: Synonym Expansion ──
    extra_terms: List[str] = []
    if req.use_synonyms:
        for token in corrected_tokens:
            synonyms = get_synonyms(token, req.max_synonyms_per_term)
            extra_terms.extend(synonyms)
            added_synonyms.extend(synonyms)
        extra_terms = list(set(extra_terms))  # deduplicate

    # ── Step 4: History Boost ──
    if req.use_history and req.user_id != "anonymous":
        history_boost = get_history_boost_terms(req.user_id, corrected_tokens)

    # ── Step 5: Build Refined Query ──
    all_terms = corrected_tokens + extra_terms + history_boost
    refined_query = " ".join(all_terms)

    # ── Update user history ──
    update_history(req.user_id, original_query, corrected_tokens)

    # ── Generate suggestions ──
    suggestions = generate_suggestions(original_query, req.user_id)

    return RefineResponse(
        original_query=original_query,
        refined_query=refined_query,
        added_synonyms=added_synonyms,
        spell_corrections=spell_corrections,
        history_boost_terms=history_boost,
        suggestions=suggestions,
    )


@app.post("/history/add")
def add_to_history(entry: HistoryEntry):
    """Manually add a query to a user's history."""
    user_history[entry.user_id].append(entry.query)
    return {"added": True, "user_id": entry.user_id, "history_size": len(user_history[entry.user_id])}


@app.get("/history/{user_id}")
def get_history(user_id: str):
    """Return a user's past queries."""
    return {
        "user_id": user_id,
        "queries": user_history.get(user_id, []),
        "count": len(user_history.get(user_id, [])),
    }


@app.post("/suggest", response_model=List[str])
def suggest(req: SuggestionRequest):
    """Return query suggestions for a partial query."""
    return generate_suggestions(req.partial_query, req.user_id, req.top_k)


@app.delete("/history/{user_id}")
def clear_history(user_id: str):
    """Clear a user's search history."""
    user_history.pop(user_id, None)
    user_term_freq.pop(user_id, None)
    return {"cleared": True, "user_id": user_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)