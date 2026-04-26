from main import BOOKS_DIR, DB_FOLDER
from rag import RAGService

import inspect

print("ФАЙЛ RAG:", inspect.getfile(RAGService))

rag = RAGService(BOOKS_DIR, DB_FOLDER)

test_cases = [
    {"query": "Що таке психосоматика?", "relevant_docs": [0, 2]},
    {"query": "Причини тривожності", "relevant_docs": [1, 3]},
    {"query": "Як стрес впливає на тіло?", "relevant_docs": [2, 4]},
    {"query": "Симптоми тривожності", "relevant_docs": [1, 3]},
    {"query": "Методи боротьби зі стресом", "relevant_docs": [2, 4]},
    {"query": "Як проявляється психосоматика?", "relevant_docs": [0, 2]},
    {"query": "Що викликає панічні атаки?", "relevant_docs": [1, 3]},
    {"query": "Вплив емоцій на фізичне здоров’я", "relevant_docs": [0, 2]},
    {"query": "Як знизити рівень тривоги?", "relevant_docs": [1, 3]},

    # 🔻 out-of-domain кейси
    {"query": "Як варити борщ", "relevant_docs": []},
    {"query": "Причини несправності літака", "relevant_docs": []},
    {"query": "Найкращі міста для подорожей", "relevant_docs": []},
    {"query": "Як створити книжку", "relevant_docs": []},
    {"query": "Чому птахи відлітають у теплі краї", "relevant_docs": []}
]

K = 3
all_results = []

for case in test_cases:
    result = rag.evaluate_retrieval(
        query=case["query"],
        relevant_indices=case["relevant_docs"],
        k=K
    )

    retrieved_top_k = result["retrieved_order"][:K]
    similarities_top_k = result["similarities"][:K]

    relevant = set(case["relevant_docs"])

    # 🔥 КОРЕКТНА ЛОГІКА
    if len(relevant) == 0:
        recall_k = None   # ❗ правильніше ніж 0
        mrr = None
        is_valid = False
    else:
        found_relevant = len([doc for doc in retrieved_top_k if doc in relevant])
        recall_k = found_relevant / len(relevant)
        mrr = result["mrr"]
        is_valid = True

    avg_similarity = (
        sum(similarities_top_k) / len(similarities_top_k)
        if similarities_top_k else 0
    )

    print(f"\nЗапит: {case['query']}")
    print(f"Recall@{K}: {recall_k if recall_k is not None else 'N/A'}")
    print(f"MRR: {mrr if mrr is not None else 'N/A'}")
    print(f"Avg Similarity@{K}: {avg_similarity:.3f}")

    all_results.append({
        "query": case["query"],
        f"recall@{K}": recall_k,
        "mrr": mrr,
        "avg_similarity": avg_similarity,
        "is_valid": is_valid
    })

# 🔹 середні значення (тільки по валідних запитах)
valid_results = [r for r in all_results if r["is_valid"]]

avg_recall = sum(r[f"recall@{K}"] for r in valid_results) / len(valid_results)
avg_mrr = sum(r["mrr"] for r in valid_results) / len(valid_results)
avg_similarity = sum(r["avg_similarity"] for r in valid_results) / len(valid_results)

print("\n===== СЕРЕДНІ МЕТРИКИ (тільки релевантні запити) =====")
print(f"Середній Recall@{K}: {avg_recall:.3f}")
print(f"Середній MRR: {avg_mrr:.3f}")
print(f"Середня Similarity: {avg_similarity:.3f}")