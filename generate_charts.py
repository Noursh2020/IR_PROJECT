# generate_charts.py
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

with open("data/touche/evaluation_results.json", encoding="utf-8") as f:
    data = json.load(f)

MODELS = ["tfidf", "bm25", "sbert", "word2vec", "hybrid_serial", "hybrid_parallel"]
LABELS = ["TF-IDF", "BM25", "SBERT", "Word2Vec", "Hybrid Serial", "Hybrid Parallel"]

# ── شارت 1: MAP و nDCG قبل refinement لكل النماذج ──
map_vals  = [data[f"{m}_before_refinement"]["MAP"] for m in MODELS]
ndcg_vals = [data[f"{m}_before_refinement"]["nDCG"] for m in MODELS]

x = range(len(MODELS))
fig, ax = plt.subplots(figsize=(10, 5))
width = 0.35
ax.bar([i - width/2 for i in x], map_vals, width, label="MAP", color="#6c63ff")
ax.bar([i + width/2 for i in x], ndcg_vals, width, label="nDCG@10", color="#60a5fa")
ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=15)
ax.set_ylabel("Score"); ax.set_title("MAP & nDCG@10 across Models — Touché 2020 (Before Refinement)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("chart_map_ndcg_per_model.png", dpi=150)
print("✅ chart_map_ndcg_per_model.png")

# ── شارت 2: قبل/بعد query refinement (MAP) ──
before = [data[f"{m}_before_refinement"]["MAP"] for m in MODELS]
after  = [data[f"{m}_after_refinement"]["MAP"] for m in MODELS]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar([i - width/2 for i in x], before, width, label="قبل Query Refinement", color="#34d399")
ax.bar([i + width/2 for i in x], after, width, label="بعد Query Refinement", color="#f87171")
ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=15)
ax.set_ylabel("MAP"); ax.set_title("تأثير Query Refinement على MAP لكل نموذج")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("chart_before_after_refinement.png", dpi=150)
print("✅ chart_before_after_refinement.png")

# ── شارت 3: كل المقاييس الأربعة لأفضل نموذج (hybrid_serial) ──
metrics = ["MAP", "Recall", "P@10", "nDCG"]
vals = [data["hybrid_serial_before_refinement"][m] for m in metrics]
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(metrics, vals, color=["#a78bfa", "#34d399", "#fbbf24", "#60a5fa"])
ax.set_title("Hybrid Serial — كل المقاييس (49 استعلام)")
ax.set_ylim(0, 1)
for i, v in enumerate(vals):
    ax.text(i, v + 0.01, f"{v:.4f}", ha="center")
plt.tight_layout()
plt.savefig("chart_hybrid_serial_metrics.png", dpi=150)
print("✅ chart_hybrid_serial_metrics.png")

print("\n📁 كل الصور جاهزة باللصق المباشر بتقرير Word/PDF.")
