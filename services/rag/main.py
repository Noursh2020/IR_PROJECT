"""
RAG Service — Retrieval-Augmented Generation
======================================================================
ما هو RAG؟ (الميزة الإضافية المطلوبة)
----------------------------------------------------------------------
RAG = Retrieval-Augmented Generation

    الفكرة الأساسية:
        بدلاً من أن يجيب نموذج اللغة (LLM) من ذاكرته فقط،
        نُغذّيه أولاً بالوثائق الأكثر صلة بالسؤال، فيُجيب
        بناءً على مصادر حقيقية من مجموعة البيانات.

    Pipeline RAG الكامل:
        ┌─────────────┐
        │  سؤال المستخدم │
        └──────┬──────┘
               │
               ▼
        ┌─────────────────────────────┐
        │  1. Retrieval (الاسترجاع)   │  ← BM25 + Embedding (FAISS)
        │     أفضل K وثيقة ذات صلة    │
        └──────┬──────────────────────┘
               │
               ▼
        ┌─────────────────────────────┐
        │  2. Context Building        │  ← دمج الوثائق في سياق
        │     بناء الـ Prompt         │
        └──────┬──────────────────────┘
               │
               ▼
        ┌─────────────────────────────┐
        │  3. Generation (التوليد)    │  ← Claude API (claude-sonnet)
        │     إجابة مدعومة بالمصادر  │
        └─────────────────────────────┘

    لماذا RAG أفضل من LLM وحده؟
        - LLM وحده: يُجيب من ذاكرته → قد يهلوس (Hallucination)
        - RAG: يُجيب من وثائق حقيقية → إجابات موثوقة وقابلة للتحقق
        - يُوثّق المصادر التي استند إليها

    لماذا RAG أفضل من IR وحده؟
        - IR وحده: يُعيد وثائق خام → المستخدم يقرأ ويستخلص
        - RAG: يُولّد إجابة مباشرة مدعومة بالمصادر

    الفرق قبل/بعد RAG في التقييم:
        قبل: نظام استرجاع فقط — يُرتّب الوثائق
        بعد: نظام استرجاع + توليد — يُجيب على الأسئلة

    Chunk Strategy (تقسيم الوثائق):
        الوثائق الطويلة تُقسَّم إلى chunks صغيرة
        لأن LLM له حد للـ context window.
        chunk_size = 512 token (افتراضي)
        chunk_overlap = 50 token (تداخل لحفظ السياق)

    Re-ranking in RAG (Lecture 3 — Hybrid Serial):
        Stage 1: BM25 يسترجع top-50 candidates
        Stage 2: Cross-encoder يُعيد ترتيبها
        Stage 3: أفضل K وثائق تذهب للـ LLM

SOA Role (Project Spec):
    خدمة RAG مستقلة على Port 8008.
    تستدعي: Retrieval Service → Embedding Service → Claude API
    Gateway يُوجّه طلبات /rag إليها.

مصدر: Lewis et al. 2020 "Retrieval-Augmented Generation for
        Knowledge-Intensive NLP Tasks" — Facebook AI Research
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import httpx
import re

app = FastAPI(
    title="RAG Service",
    description="Retrieval-Augmented Generation — يسترجع الوثائق ثم يولّد إجابة بالـ LLM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── عناوين الخدمات الداخلية ──
RETRIEVAL_URL  = "http://localhost:8003"
EMBEDDING_URL  = "http://localhost:8006"

# ── Anthropic API ──
# يُستخدم claude-sonnet-4-20250514 حسب المتطلبات
ANTHROPIC_API_URL   = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL        = "claude-sonnet-4-20250514"
ANTHROPIC_API_KEY   = ""  # يُضاف من الـ environment أو يُمرَّر في الطلب


# ══════════════════════════════════════════════
# Schemas
# ══════════════════════════════════════════════

class RAGRequest(BaseModel):
    """
    طلب RAG من المستخدم.

    question:       سؤال المستخدم بالنص الطبيعي
    dataset_id:     مجموعة البيانات للبحث فيها (msmarco أو nq)
    retrieval_model: نموذج الاسترجاع (bm25, hybrid_serial, hybrid_parallel)
    top_k:          عدد الوثائق التي تُمرَّر للـ LLM
    max_tokens:     الحد الأقصى لطول الإجابة
    language:       لغة الإجابة (arabic أو english)
    api_key:        مفتاح Anthropic API (اختياري إن كان في البيئة)
    """
    question: str
    dataset_id: str = "msmarco"
    retrieval_model: str = "hybrid_serial"  # أفضل للـ RAG (دقة عالية)
    top_k: int = 5                           # عدد الوثائق للـ context
    max_tokens: int = 800
    language: str = "arabic"                # "arabic" | "english"
    api_key: Optional[str] = None
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


class RetrievedContext(BaseModel):
    """وثيقة مُسترجَعة ضمن الـ context."""
    doc_id: str
    rank: int
    score: float
    snippet: str
    title: str


class RAGResponse(BaseModel):
    """
    استجابة RAG الكاملة:
    - answer:    الإجابة المولَّدة من Claude
    - contexts:  الوثائق التي استُند إليها
    - pipeline:  تفاصيل كل مرحلة (للتوثيق والشفافية)
    """
    question: str
    answer: str
    contexts: List[RetrievedContext]
    pipeline: Dict[str, Any]
    retrieval_model: str
    dataset_id: str
    tokens_used: Optional[int] = None


# ══════════════════════════════════════════════
# المرحلة 1: الاسترجاع (Retrieval)
# ══════════════════════════════════════════════

async def retrieve_contexts(
    question: str,
    dataset_id: str,
    model: str,
    top_k: int,
    bm25_k1: float,
    bm25_b: float,
) -> List[RetrievedContext]:
    """
    المرحلة الأولى من RAG: استرجاع الوثائق ذات الصلة.

    لماذا نسترجع قبل التوليد؟
        اللغة النموذج لا يعرف محتوى مجموعة بياناتك المحددة.
        الاسترجاع يُحضر المعلومات الصحيحة من الـ Dataset.

    لماذا hybrid_serial الأفضل للـ RAG؟
        - BM25 يضمن تغطية المصطلحات الدقيقة (keyword match)
        - Embedding يضمن الفهم الدلالي (semantic understanding)
        - معاً: أعلى دقة في اختيار الوثائق → إجابة أفضل

    top_k = 5 افتراضياً:
        أقل من 3: سياق ضيق → إجابة منقوصة
        أكثر من 8: يتجاوز حد الـ context window → تكلفة أعلى
        5 هو التوازن المثالي للمشروع.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{RETRIEVAL_URL}/retrieve",
            json={
                "query": question,
                "dataset_id": dataset_id,
                "model": model,
                "top_k": top_k,
                "bm25_k1": bm25_k1,
                "bm25_b": bm25_b,
                "embedding_model": "sbert",
                "hybrid_candidates": 100,
            },
        )

    if r.status_code != 200:
        raise HTTPException(502, f"Retrieval service error: {r.text[:300]}")

    data = r.json()
    results = data.get("results", [])

    contexts = []
    for item in results:
        contexts.append(RetrievedContext(
            doc_id=item["doc_id"],
            rank=item["rank"],
            score=round(item["score"], 6),
            snippet=item.get("snippet", "")[:600],
            title=item.get("title", item["doc_id"]),
        ))

    return contexts


# ══════════════════════════════════════════════
# المرحلة 2: بناء الـ Prompt
# ══════════════════════════════════════════════

def build_rag_prompt(
    question: str,
    contexts: List[RetrievedContext],
    language: str = "arabic",
) -> str:
    """
    المرحلة الثانية من RAG: بناء الـ Prompt للـ LLM.

    ما هو الـ Prompt Engineering؟
        الطريقة التي نُخبر بها الـ LLM بما نريده.
        الـ Prompt الجيد = إجابة جيدة.

    مكونات الـ Prompt في RAG:
        1. System instruction: دور الـ LLM + قيوده
        2. Retrieved contexts: الوثائق مرقّمة
        3. Question: السؤال الأصلي
        4. Answer format: تعليمات الإجابة

    لماذا نضع "استند فقط إلى الوثائق المذكورة"؟
        لمنع الـ Hallucination — اللغة النموذج قد يخترع معلومات.
        بهذا القيد، يُجيب فقط مما وجدناه في الـ Dataset.

    Grounding Strategy:
        كل وثيقة تُرقَّم [1], [2], [3]...
        الـ LLM يُشير إليها في الإجابة كـ [1], [2]...
        هذا يُتيح التحقق من المصادر (Verifiability).
    """
    # بناء قسم الوثائق
    context_blocks = []
    for ctx in contexts:
        title_line = f"العنوان: {ctx.title}" if ctx.title and ctx.title != ctx.doc_id else ""
        block = f"[{ctx.rank}] {title_line}\n{ctx.snippet}"
        context_blocks.append(block)

    context_text = "\n\n".join(context_blocks)

    if language == "arabic":
        prompt = f"""أنت مساعد بحثي متخصص في نظام استرجاع المعلومات.

لديك الوثائق التالية المُسترجَعة من مجموعة البيانات:

{context_text}

السؤال: {question}

التعليمات:
- أجب بالعربية بشكل واضح ومنظم.
- استند فقط إلى المعلومات الموجودة في الوثائق أعلاه.
- أشر إلى رقم الوثيقة [1] أو [2] عند الاستشهاد.
- إذا لم تكفِ الوثائق للإجابة، وضّح ذلك صراحةً.
- لا تخترع معلومات غير موجودة في الوثائق.

الإجابة:"""

    else:
        prompt = f"""You are a research assistant specialized in information retrieval.

You have the following documents retrieved from the dataset:

{context_text}

Question: {question}

Instructions:
- Answer clearly and in an organized manner.
- Base your answer ONLY on the information in the documents above.
- Cite the document number [1] or [2] when referencing.
- If the documents are insufficient, state that explicitly.
- Do NOT hallucinate information not present in the documents.

Answer:"""

    return prompt


# ══════════════════════════════════════════════
# المرحلة 3: التوليد (Generation)
# ══════════════════════════════════════════════

async def generate_answer(
    prompt: str,
    max_tokens: int,
    api_key: str,
) -> Dict[str, Any]:
    """
    المرحلة الثالثة من RAG: توليد الإجابة بواسطة Claude.

    كيف يعمل Claude API؟
        - نُرسل الـ prompt كـ user message
        - Claude يقرأ الوثائق + السؤال
        - يُولّد إجابة مبنية على الوثائق

    النموذج المستخدم: claude-sonnet-4-20250514
        - متوازن بين السرعة والجودة
        - يدعم العربية والإنجليزية
        - context window كبير كافٍ للوثائق

    معامل Temperature:
        temperature = 0.3 (منخفض)
        في RAG نريد إجابات دقيقة لا إبداعية.
        درجة حرارة منخفضة → إجابات أكثر تحديداً وموثوقية.

    معامل max_tokens:
        يتحكم بطول الإجابة.
        800 token ≈ 600 كلمة عربية — كافٍ لإجابة شاملة.
    """
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "temperature": 0.3,   # منخفض للدقة في RAG
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(ANTHROPIC_API_URL, headers=headers, json=payload)

    if r.status_code != 200:
        raise HTTPException(502, f"Claude API error {r.status_code}: {r.text[:400]}")

    data = r.json()

    # استخراج النص من الاستجابة
    content_blocks = data.get("content", [])
    answer = " ".join(
        block["text"] for block in content_blocks
        if block.get("type") == "text"
    ).strip()

    tokens_used = data.get("usage", {}).get("output_tokens", 0)

    return {"answer": answer, "tokens_used": tokens_used}


# ══════════════════════════════════════════════
# Fallback: RAG بدون API Key
# ══════════════════════════════════════════════

def generate_answer_local(
    question: str,
    contexts: List[RetrievedContext],
    language: str,
) -> str:
    """
    توليد إجابة محلية بدون API Key.
    يُستخدم عند غياب مفتاح Anthropic.

    الاستراتيجية:
        - استخرج الجمل الأكثر صلة بالسؤال من الوثائق
        - رتّبها وقدّمها كإجابة منظمة
        - هذا أبسط من LLM لكن لا يحتاج API

    هذا يُسمى Extractive QA (بدلاً من Abstractive QA مع LLM).
    Extractive: يُعيد جمل موجودة فعلاً في الوثائق
    Abstractive: يُولّد جمل جديدة مبنية على الوثائق
    """
    question_words = set(re.sub(r'[^\w\s]', '', question.lower()).split())

    # اختر الجملة الأكثر صلة من كل وثيقة
    best_sentences = []
    for ctx in contexts[:3]:
        sentences = [s.strip() for s in re.split(r'[.!?؟]', ctx.snippet) if len(s.strip()) > 30]
        if not sentences:
            best_sentences.append(ctx.snippet[:200])
            continue

        # احسب عدد كلمات السؤال في كل جملة
        best = max(
            sentences,
            key=lambda s: len(set(s.lower().split()) & question_words)
        )
        best_sentences.append(best)

    if language == "arabic":
        header = f"بناءً على الوثائق المُسترجَعة حول: **{question}**\n\n"
        body = "\n\n".join(
            f"• [{i+1}] {sent}"
            for i, sent in enumerate(best_sentences) if sent
        )
        footer = "\n\n*(هذه إجابة استخراجية — أضف Anthropic API Key للحصول على إجابة توليدية أشمل)*"
    else:
        header = f"Based on retrieved documents about: **{question}**\n\n"
        body = "\n\n".join(
            f"• [{i+1}] {sent}"
            for i, sent in enumerate(best_sentences) if sent
        )
        footer = "\n\n*(Extractive answer — add Anthropic API Key for a full generative response)*"

    return header + body + footer


# ══════════════════════════════════════════════
# API Endpoints
# ══════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "rag",
        "claude_model": CLAUDE_MODEL,
        "pipeline": ["retrieval", "context_building", "generation"],
    }


@app.post("/ask", response_model=RAGResponse)
async def ask(req: RAGRequest):
    """
    ══════════════════════════════════════════
    الـ Endpoint الرئيسي للـ RAG
    ══════════════════════════════════════════

    Pipeline كامل في طلب واحد:

    1. Retrieval:
       يُرسل السؤال لـ Retrieval Service
       يُعيد top_k وثائق ذات صلة

    2. Context Building:
       يبني Prompt يتضمن الوثائق + السؤال

    3. Generation:
       يُرسل Prompt لـ Claude API
       يُعيد إجابة مولَّدة مدعومة بمصادر

    المعاملات الرئيسية:
        retrieval_model: اختر النموذج المناسب
            - hybrid_serial: الأفضل للدقة (BM25 → SBERT)
            - bm25: الأسرع
            - sbert: للأسئلة الدلالية

        top_k: عدد الوثائق للـ context (3-7 مثالي)

        language: arabic أو english
    """
    pipeline_log = {}

    # ── المرحلة 1: الاسترجاع ──
    pipeline_log["step_1"] = "🔍 Retrieval — جارٍ استرجاع الوثائق..."
    try:
        contexts = await retrieve_contexts(
            question=req.question,
            dataset_id=req.dataset_id,
            model=req.retrieval_model,
            top_k=req.top_k,
            bm25_k1=req.bm25_k1,
            bm25_b=req.bm25_b,
        )
        pipeline_log["step_1"] = f"✅ Retrieval — استُرجع {len(contexts)} وثيقة"
        pipeline_log["retrieved_doc_ids"] = [c.doc_id for c in contexts]
    except Exception as e:
        pipeline_log["step_1"] = f"❌ Retrieval فشل: {str(e)}"
        raise HTTPException(502, f"Retrieval failed: {e}")

    if not contexts:
        return RAGResponse(
            question=req.question,
            answer="لم يُعثر على وثائق ذات صلة في مجموعة البيانات للإجابة على هذا السؤال.",
            contexts=[],
            pipeline=pipeline_log,
            retrieval_model=req.retrieval_model,
            dataset_id=req.dataset_id,
        )

    # ── المرحلة 2: بناء الـ Prompt ──
    pipeline_log["step_2"] = "📝 Context Building — جارٍ بناء الـ Prompt..."
    prompt = build_rag_prompt(req.question, contexts, req.language)
    pipeline_log["step_2"] = f"✅ Context Building — Prompt بُني من {len(contexts)} وثيقة"
    pipeline_log["prompt_length_chars"] = len(prompt)

    # ── المرحلة 3: التوليد ──
    api_key = req.api_key or ANTHROPIC_API_KEY
    tokens_used = None

    if api_key:
        # توليد حقيقي بـ Claude API
        pipeline_log["step_3"] = f"🤖 Generation — جارٍ الإرسال لـ {CLAUDE_MODEL}..."
        try:
            result = await generate_answer(prompt, req.max_tokens, api_key)
            answer = result["answer"]
            tokens_used = result["tokens_used"]
            pipeline_log["step_3"] = f"✅ Generation — {tokens_used} token مُولَّدة"
            pipeline_log["generation_mode"] = "claude_api"
        except Exception as e:
            pipeline_log["step_3"] = f"⚠️ Claude API فشل، تحويل للـ Extractive: {str(e)}"
            answer = generate_answer_local(req.question, contexts, req.language)
            pipeline_log["generation_mode"] = "extractive_fallback"
    else:
        # توليد استخراجي بدون API
        pipeline_log["step_3"] = "📋 Generation — وضع استخراجي (بدون API Key)"
        answer = generate_answer_local(req.question, contexts, req.language)
        pipeline_log["generation_mode"] = "extractive_local"

    return RAGResponse(
        question=req.question,
        answer=answer,
        contexts=contexts,
        pipeline=pipeline_log,
        retrieval_model=req.retrieval_model,
        dataset_id=req.dataset_id,
        tokens_used=tokens_used,
    )


@app.post("/ask/batch")
async def ask_batch(questions: List[str], dataset_id: str = "msmarco", api_key: Optional[str] = None):
    """
    تشغيل RAG على مجموعة أسئلة دفعةً واحدة.
    مفيد لتقييم النظام على مجموعة الـ queries من ir-datasets.
    """
    results = []
    for question in questions[:20]:  # حد 20 سؤالاً لتجنب timeout
        try:
            req = RAGRequest(
                question=question,
                dataset_id=dataset_id,
                api_key=api_key,
            )
            result = await ask(req)
            results.append({
                "question": question,
                "answer": result.answer[:300],  # اختصار للـ batch
                "contexts_count": len(result.contexts),
                "status": "ok",
            })
        except Exception as e:
            results.append({
                "question": question,
                "answer": None,
                "status": "error",
                "error": str(e),
            })
    return {"results": results, "total": len(results)}


@app.get("/explain")
def explain_rag():
    """
    شرح مبسط لـ RAG للتقرير والمقابلة.
    """
    return {
        "rag_definition": "Retrieval-Augmented Generation — نموذج يجمع الاسترجاع والتوليد",
        "pipeline_steps": [
            {
                "step": 1,
                "name": "Retrieval",
                "description": "البحث في مجموعة البيانات عن أكثر الوثائق صلة بالسؤال",
                "tool": "BM25 + SBERT + FAISS",
                "output": "top-K وثائق مُرتَّبة حسب الصلة",
            },
            {
                "step": 2,
                "name": "Context Building",
                "description": "دمج الوثائق المُسترجَعة في Prompt منظم للـ LLM",
                "tool": "Prompt Engineering",
                "output": "Prompt يتضمن السؤال + الوثائق + التعليمات",
            },
            {
                "step": 3,
                "name": "Generation",
                "description": "Claude يقرأ الـ Prompt ويُولّد إجابة مدعومة بالمصادر",
                "tool": f"Claude API ({CLAUDE_MODEL})",
                "output": "إجابة نصية مع مراجع للوثائق",
            },
        ],
        "advantages_over_ir_only": [
            "يُولّد إجابة مباشرة بدلاً من قائمة وثائق",
            "يدعم الأسئلة التحليلية والمقارنة",
            "يوثّق مصادره — قابلية التحقق",
        ],
        "advantages_over_llm_only": [
            "يستند إلى بيانات Dataset حقيقية",
            "يمنع الـ Hallucination",
            "يمكن تحديث المعرفة بتحديث الـ Dataset فقط",
        ],
        "reference": "Lewis et al. 2020, 'RAG for Knowledge-Intensive NLP Tasks', Facebook AI",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)