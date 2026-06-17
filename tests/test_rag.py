"""
اختبارات RAG Service
======================================================
تُغطي كل مكونات الـ RAG Pipeline:
1. بناء الـ Prompt
2. الاسترجاع الاستخراجي (بدون API)
3. تنسيق الوثائق
"""

import pytest
import sys
import re
sys.path.insert(0, '.')

from services.rag.main import (
    build_rag_prompt,
    generate_answer_local,
    RetrievedContext,
)


# ── بيانات الاختبار ──
def make_context(rank=1, snippet="Information retrieval is the process of finding relevant documents."):
    return RetrievedContext(
        doc_id=f"doc_{rank}",
        rank=rank,
        score=0.95 - rank * 0.1,
        snippet=snippet,
        title=f"Document {rank}",
    )


class TestPromptBuilding:
    """اختبار بناء الـ Prompt — المرحلة الثانية من RAG."""

    def test_arabic_prompt_contains_question(self):
        """يجب أن يحتوي الـ Prompt على سؤال المستخدم."""
        ctx = [make_context(1)]
        prompt = build_rag_prompt("ما هو BM25؟", ctx, language="arabic")
        assert "ما هو BM25؟" in prompt

    def test_arabic_prompt_contains_context(self):
        """يجب أن يحتوي الـ Prompt على مقتطف الوثيقة."""
        ctx = [make_context(1, "BM25 is a ranking function.")]
        prompt = build_rag_prompt("what is BM25", ctx, language="arabic")
        assert "BM25 is a ranking function." in prompt

    def test_english_prompt_structure(self):
        """Prompt الإنجليزي يجب أن يتضمن تعليمات الاستشهاد."""
        ctx = [make_context(1), make_context(2)]
        prompt = build_rag_prompt("what is IR?", ctx, language="english")
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "hallucinate" in prompt.lower()

    def test_prompt_includes_no_hallucination_instruction(self):
        """
        لمنع الـ Hallucination يجب أن يكون في الـ Prompt تعليمات
        بعدم اختراع معلومات.
        """
        ctx = [make_context(1)]
        prompt_ar = build_rag_prompt("سؤال", ctx, language="arabic")
        prompt_en = build_rag_prompt("question", ctx, language="english")
        # أحد التعليمتين يجب أن يكون موجوداً
        assert "لا تخترع" in prompt_ar or "NOT hallucinate" in prompt_en

    def test_multiple_contexts_all_included(self):
        """جميع الوثائق يجب أن تُضمَّن في الـ Prompt."""
        contexts = [make_context(i, f"snippet number {i}") for i in range(1, 6)]
        prompt = build_rag_prompt("test", contexts, language="english")
        for i in range(1, 6):
            assert f"snippet number {i}" in prompt

    def test_document_rank_labels(self):
        """كل وثيقة يجب أن تُرقَّم [1], [2], [3]..."""
        contexts = [make_context(i) for i in range(1, 4)]
        prompt = build_rag_prompt("test", contexts, language="english")
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[3]" in prompt


class TestExtractiveGeneration:
    """
    اختبار التوليد الاستخراجي — يعمل بدون API Key.
    يختار الجمل الأكثر صلة من الوثائق.
    """

    def test_returns_string(self):
        """يجب أن يُعيد نصاً."""
        ctx = [make_context(1, "Information retrieval systems find relevant documents.")]
        result = generate_answer_local("what is IR?", ctx, language="english")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_arabic_mode_has_arabic_header(self):
        """وضع العربية يجب أن يُعيد header بالعربية."""
        ctx = [make_context(1)]
        result = generate_answer_local("ما هو IR؟", ctx, language="arabic")
        assert "بناءً على" in result

    def test_english_mode_has_english_header(self):
        """وضع الإنجليزية يجب أن يُعيد header بالإنجليزية."""
        ctx = [make_context(1)]
        result = generate_answer_local("what is IR?", ctx, language="english")
        assert "Based on" in result

    def test_uses_max_3_contexts(self):
        """يجب ألا يتجاوز 3 وثائق في الإجابة الاستخراجية."""
        contexts = [make_context(i, f"sentence about topic {i}. More text here.") for i in range(1, 8)]
        result = generate_answer_local("topic", contexts, language="english")
        # يجب أن يحتوي على [1] و [2] و [3] لكن ليس [4]
        assert "[1]" in result
        assert "[4]" not in result

    def test_relevant_sentence_selected(self):
        """
        يجب أن تختار الجملة التي تحتوي على كلمات السؤال.
        السؤال عن BM25 → اختار الجملة التي تحتوي BM25.
        """
        snippet = "The weather is nice today. BM25 is a probabilistic ranking model. I like coffee."
        ctx = [make_context(1, snippet)]
        result = generate_answer_local("BM25 ranking model", ctx, language="english")
        assert "BM25" in result

    def test_fallback_note_present(self):
        """يجب أن تُشير الإجابة إلى أنها استخراجية (ليست من LLM)."""
        ctx = [make_context(1)]
        result = generate_answer_local("test", ctx, language="english")
        assert "Extractive" in result or "API Key" in result


class TestRAGContextFormatting:
    """اختبار تنسيق الوثائق المُسترجَعة."""

    def test_context_fields(self):
        """كل RetrievedContext يجب أن تحتوي الحقول المطلوبة."""
        ctx = RetrievedContext(
            doc_id="d123",
            rank=1,
            score=0.95,
            snippet="test snippet",
            title="Test Title",
        )
        assert ctx.doc_id == "d123"
        assert ctx.rank == 1
        assert 0 <= ctx.score <= 1
        assert len(ctx.snippet) > 0

    def test_score_range(self):
        """درجة التشابه يجب أن تكون بين 0 و 1 (cosine similarity)."""
        ctx = make_context(1)
        assert 0.0 <= ctx.score <= 1.0

    def test_snippet_not_empty(self):
        """مقتطف الوثيقة يجب ألا يكون فارغاً."""
        ctx = make_context(1, "Some text here.")
        assert len(ctx.snippet) > 0


class TestRAGPipelineLogic:
    """اختبار منطق الـ Pipeline."""

    def test_prompt_length_reasonable(self):
        """
        الـ Prompt يجب ألا يكون طويلاً جداً (يتجاوز context window).
        حد عملي: أقل من 8000 حرف لـ top_k=5.
        """
        contexts = [
            make_context(i, "a" * 600)  # كل وثيقة 600 حرف
            for i in range(1, 6)
        ]
        prompt = build_rag_prompt("question", contexts, language="english")
        assert len(prompt) < 8000, f"Prompt too long: {len(prompt)} chars"

    def test_empty_contexts_handled(self):
        """إذا كانت الوثائق فارغة، الإجابة يجب أن تكون معقولة."""
        result = generate_answer_local("test question", [], language="arabic")
        assert isinstance(result, str)

    def test_arabic_question_in_english_prompt(self):
        """سؤال عربي مع prompt إنجليزي يجب أن يعمل بدون خطأ."""
        ctx = [make_context(1)]
        prompt = build_rag_prompt("ما هو التعلم الآلي؟", ctx, language="english")
        assert "ما هو التعلم الآلي؟" in prompt