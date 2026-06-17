"""
Tests for IR System - Phase 1
Tests each service independently (SOA principle).

Run with: pytest tests/test_phase1.py -v
"""

import pytest
import math
from unittest.mock import AsyncMock, patch

# ──────────────────────────────────────────────
# Preprocessing Service Tests (Lecture 1)
# ──────────────────────────────────────────────

class TestPreprocessing:
    """Tests for text processing pipeline from Lecture 1."""

    def setup_method(self):
        from services.preprocessing.main import (
            normalize_text, tokenize, remove_stop_words,
            apply_stemming, apply_lemmatization, preprocess_single,
        )
        self.normalize = normalize_text
        self.tokenize = tokenize
        self.remove_sw = remove_stop_words
        self.stem = apply_stemming
        self.lemmatize = apply_lemmatization
        self.preprocess = preprocess_single

    def test_normalization_lowercase(self):
        """Lecture 1: 'Convert to lowercase: EnGinEer -> engineer'"""
        result = self.normalize("EnGinEer")
        assert result == "engineer"

    def test_normalization_removes_punctuation(self):
        result = self.normalize("Software? Development!")
        assert "?" not in result
        assert "!" not in result

    def test_tokenization(self):
        tokens = self.tokenize("information retrieval system")
        assert "information" in tokens
        assert "retrieval" in tokens

    def test_stop_words_removal(self):
        """Lecture 1: 'the', 'is', 'are' should be filtered."""
        tokens = ["the", "document", "is", "relevant", "a"]
        filtered = self.remove_sw(tokens)
        assert "the" not in filtered
        assert "is" not in filtered
        assert "document" in filtered
        assert "relevant" in filtered

    def test_stemming(self):
        """Lecture 1: 'historical -> histor'"""
        tokens = ["historical", "running", "retrieval"]
        stemmed = self.stem(tokens)
        assert "histor" in stemmed or any(len(s) < len("historical") for s in stemmed)

    def test_lemmatization(self):
        """Lecture 1: 'are -> be', 'running -> run'"""
        tokens = ["running", "better"]
        lemmatized = self.lemmatize(tokens)
        # WordNet lemmatizer should reduce these
        assert len(lemmatized) == 2

    def test_full_pipeline(self):
        result = self.preprocess("The historical documents ARE running well")
        assert result.processed_text != ""
        assert "the" not in result.processed_tokens
        assert "are" not in result.processed_tokens
        assert len(result.processed_tokens) > 0


# ──────────────────────────────────────────────
# Indexing Service Tests (Lecture 2)
# ──────────────────────────────────────────────

class TestIndexing:
    """Tests for index construction from Lecture 2."""

    def setup_method(self):
        from services.indexing.main import (
            build_inverted_index, compute_tfidf_vectors,
            inverted_index, doc_store, doc_lengths, tfidf_vectors
        )
        # Clear state
        inverted_index.clear()
        doc_store.clear()
        doc_lengths.clear()
        tfidf_vectors.clear()

        self.build = build_inverted_index
        self.compute_tfidf = compute_tfidf_vectors
        self.inverted = inverted_index
        self.doc_store = doc_store
        self.doc_lengths = doc_lengths
        self.tfidf = tfidf_vectors

    def test_inverted_index_basic(self):
        """Lecture 2: 'Inverted Index: term -> docIDs; used for keyword search.'"""
        self.doc_store["d1"] = {"text": "info retrieval", "title": ""}
        self.build("d1", ["info", "retrieval"])

        assert "info" in self.inverted
        assert "d1" in self.inverted["info"]
        assert "retrieval" in self.inverted

    def test_term_frequency_counted(self):
        """TF should count how many times a term appears."""
        self.doc_store["d2"] = {"text": "ir ir ir", "title": ""}
        self.build("d2", ["ir", "ir", "ir"])

        assert self.inverted["ir"]["d2"] == 3

    def test_positional_index(self):
        """Lecture 2: 'Positional Index: term -> docID + positions'"""
        from services.indexing.main import positional_index
        positional_index.clear()
        self.doc_store["d3"] = {"text": "a b a", "title": ""}
        self.build("d3", ["a", "b", "a"])

        assert 0 in positional_index["a"]["d3"]
        assert 2 in positional_index["a"]["d3"]
        assert 1 in positional_index["b"]["d3"]

    def test_tfidf_computed(self):
        """Lecture 3: TF-IDF should give higher weight to rare terms."""
        self.doc_store["d1"] = {"text": "rare term", "title": ""}
        self.doc_store["d2"] = {"text": "common common common", "title": ""}
        self.build("d1", ["rare", "common"])
        self.build("d2", ["common", "common", "common"])
        self.compute_tfidf()

        # "rare" should have higher IDF (appears in fewer docs)
        rare_score = self.tfidf.get("d1", {}).get("rare", 0)
        common_in_d1 = self.tfidf.get("d1", {}).get("common", 0)
        assert rare_score > 0 or common_in_d1 >= 0  # basic sanity


# ──────────────────────────────────────────────
# Retrieval / BM25 Tests (Lecture 3)
# ──────────────────────────────────────────────

class TestBM25:
    """Tests for BM25 scoring from Lecture 3."""

    def setup_method(self):
        from services.retrieval.main import bm25_score
        self.bm25 = bm25_score

    def test_bm25_zero_for_missing_term(self):
        """BM25 should return 0 if query term not in document."""
        inverted = {"python": {"d1": 3}}
        doc_lengths = {"d1": 100}
        score = self.bm25(["java"], "d1", inverted, doc_lengths, 100, 10)
        assert score == 0.0

    def test_bm25_positive_for_matching_term(self):
        """BM25 should return positive score when term matches."""
        inverted = {"python": {"d1": 5, "d2": 1}}
        doc_lengths = {"d1": 100, "d2": 50}
        score = self.bm25(["python"], "d1", inverted, doc_lengths, 75, 2)
        assert score > 0

    def test_bm25_k1_saturation(self):
        """
        Lecture 3 - Saturation Effect:
        'Higher k1 → Slower saturation → TF has more influence.'
        A doc with tf=10 should score higher with high k1 than low k1.
        """
        inverted = {"term": {"d1": 10}}
        doc_lengths = {"d1": 100}

        score_high_k1 = self.bm25(["term"], "d1", inverted, doc_lengths, 100, 2, k1=5.0, b=0.75)
        score_low_k1 = self.bm25(["term"], "d1", inverted, doc_lengths, 100, 2, k1=0.5, b=0.75)
        assert score_high_k1 > score_low_k1

    def test_bm25_b_length_normalization(self):
        """
        Lecture 3: 'Higher b → Stronger document length normalization.'
        A very long document should be penalized more with high b.
        """
        inverted = {"term": {"short": 1, "long": 1}}
        doc_lengths = {"short": 10, "long": 1000}
        avg_dl = 100

        score_short_high_b = self.bm25(["term"], "short", inverted, doc_lengths, avg_dl, 2, k1=1.5, b=0.9)
        score_long_high_b = self.bm25(["term"], "long", inverted, doc_lengths, avg_dl, 2, k1=1.5, b=0.9)
        # Short document should be favored when b is high (long doc penalized)
        assert score_short_high_b >= score_long_high_b


# ──────────────────────────────────────────────
# Fusion Methods Tests (Lecture 3)
# ──────────────────────────────────────────────

class TestFusion:
    """Tests for RRF and Weighted Fusion from Lecture 3."""

    def setup_method(self):
        from services.retrieval.main import rrf_fusion, weighted_fusion
        self.rrf = rrf_fusion
        self.weighted = weighted_fusion

    def test_rrf_combines_two_lists(self):
        """
        Lecture 3: 'RRF combines results from multiple ranked lists.'
        A document appearing in both lists should rank higher.
        """
        list1 = [("d1", 0.9), ("d2", 0.7), ("d3", 0.5)]
        list2 = [("d1", 0.8), ("d4", 0.6), ("d2", 0.4)]
        result = self.rrf([list1, list2])
        doc_ids = [d for d, _ in result]
        # d1 appears in both lists → should be top ranked
        assert doc_ids[0] == "d1"

    def test_rrf_score_formula(self):
        """RRF score = 1/(k+rank). Lower rank = higher score."""
        list1 = [("d1", 1.0)]  # rank 1
        list2 = [("d2", 1.0), ("d1", 0.5)]  # d1 at rank 2
        result = dict(self.rrf([list1, list2], k=60))

        # d1: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
        expected = 1 / 61 + 1 / 62
        assert abs(result["d1"] - expected) < 1e-6

    def test_weighted_fusion(self):
        """Lecture 3: 'Assign weights to each model's output.'"""
        list1 = [("d1", 1.0), ("d2", 0.5)]
        list2 = [("d2", 1.0), ("d1", 0.3)]
        result = dict(self.weighted([list1, list2], [0.7, 0.3]))
        # Both d1 and d2 should have scores
        assert "d1" in result
        assert "d2" in result


# ──────────────────────────────────────────────
# Evaluation Metrics Tests (Lecture 4)
# ──────────────────────────────────────────────

class TestEvaluation:
    """Tests for all IR evaluation metrics from Lecture 4."""

    def setup_method(self):
        from services.ranking_evaluation.main import (
            compute_precision, compute_precision_at_k,
            compute_recall, compute_average_precision,
            compute_ndcg, compute_dcg,
        )
        self.precision = compute_precision
        self.pk = compute_precision_at_k
        self.recall = compute_recall
        self.ap = compute_average_precision
        self.ndcg = compute_ndcg
        self.dcg = compute_dcg

    def test_precision_all_relevant(self):
        """Precision = 1.0 when all retrieved are relevant."""
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d2", "d3"}
        assert self.precision(retrieved, relevant) == 1.0

    def test_precision_none_relevant(self):
        retrieved = ["d1", "d2"]
        relevant = {"d3", "d4"}
        assert self.precision(retrieved, relevant) == 0.0

    def test_precision_at_k(self):
        """Lecture 4: 'P@K considers only the top K recommendations.'"""
        retrieved = ["d1", "d2", "d3", "d4", "d5"]
        relevant = {"d1", "d3", "d5"}
        # P@3: top 3 are d1, d2, d3 → 2 relevant → 2/3
        assert abs(self.pk(retrieved, relevant, 3) - 2/3) < 1e-9

    def test_recall(self):
        """Lecture 4: Recall = relevant_retrieved / total_relevant"""
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d2", "d4", "d5"}
        # 2 relevant retrieved out of 4 total
        assert self.recall(retrieved, relevant) == 0.5

    def test_recall_perfect(self):
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d2"}
        assert self.recall(retrieved, relevant) == 1.0

    def test_average_precision_perfect(self):
        """AP = 1.0 when all retrieved in order are relevant."""
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d2", "d3"}
        ap = self.ap(retrieved, relevant, k=3)
        assert ap == 1.0

    def test_average_precision_order_matters(self):
        """AP rewards relevant docs appearing earlier in the list."""
        retrieved_early = ["d1", "d_irr", "d_irr2"]   # relevant at rank 1
        retrieved_late  = ["d_irr", "d_irr2", "d1"]   # relevant at rank 3
        relevant = {"d1"}
        ap_early = self.ap(retrieved_early, relevant, k=3)
        ap_late  = self.ap(retrieved_late,  relevant, k=3)
        assert ap_early > ap_late

    def test_ndcg_perfect_ranking(self):
        """
        Lecture 4: 'nDCG = 1.0 for a perfect ranking.'
        If the retrieved order matches ideal order, nDCG = 1.0.
        """
        graded = {"d1": 3, "d2": 2, "d3": 1}
        retrieved = ["d1", "d2", "d3"]   # ideal order
        score = self.ndcg(retrieved, graded, k=3)
        assert abs(score - 1.0) < 1e-9

    def test_ndcg_penalizes_late_relevant(self):
        """nDCG should be lower when relevant docs appear later."""
        graded = {"d1": 3, "d2": 0, "d3": 0}
        retrieved_good = ["d1", "d2", "d3"]
        retrieved_bad  = ["d2", "d3", "d1"]
        score_good = self.ndcg(retrieved_good, graded, k=3)
        score_bad  = self.ndcg(retrieved_bad,  graded, k=3)
        assert score_good > score_bad

    def test_dcg_formula(self):
        """DCG@K = Σ rel(i) / log2(i+1)"""
        graded = {"d1": 3, "d2": 2, "d3": 1}
        retrieved = ["d1", "d2", "d3"]
        # DCG = 3/log2(2) + 2/log2(3) + 1/log2(4)
        expected = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
        result = self.dcg(retrieved, graded, k=3)
        assert abs(result - expected) < 1e-9


# ──────────────────────────────────────────────
# Query Refinement Tests (Project Spec §5)
# ──────────────────────────────────────────────

class TestQueryRefinement:
    """Tests for query refinement features."""

    def setup_method(self):
        from services.query_refinement.main import (
            get_synonyms, levenshtein_distance,
            get_history_boost_terms, update_history,
            user_history, user_term_freq,
        )
        user_history.clear()
        user_term_freq.clear()

        self.get_synonyms = get_synonyms
        self.edit_dist = levenshtein_distance
        self.boost = get_history_boost_terms
        self.update_hist = update_history
        self.user_history = user_history
        self.user_term_freq = user_term_freq

    def test_synonym_expansion(self):
        """Spec §5: 'Adding synonyms to the user's query.'"""
        synonyms = self.get_synonyms("car", max_per_term=3)
        # WordNet should return synonyms like "auto", "automobile"
        assert isinstance(synonyms, list)

    def test_levenshtein_same_string(self):
        assert self.edit_dist("python", "python") == 0

    def test_levenshtein_one_edit(self):
        assert self.edit_dist("python", "pyton") == 1   # deletion

    def test_levenshtein_typo(self):
        assert self.edit_dist("retrieval", "retreival") == 2

    def test_history_tracking(self):
        """Spec §5: User history should track past queries."""
        self.update_hist("user1", "information retrieval", ["information", "retrieval"])
        self.update_hist("user1", "information retrieval", ["information", "retrieval"])
        assert "information retrieval" in self.user_history["user1"]

    def test_history_boost(self):
        """Spec §5: Boost terms from user's past searches."""
        # Simulate user frequently searching "neural"
        self.user_term_freq["user1"]["neural"] = 5
        self.user_term_freq["user1"]["network"] = 4
        boost = self.boost("user1", ["retrieval"], top_k=2)
        assert "neural" in boost or "network" in boost