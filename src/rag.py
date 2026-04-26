import os
import logging
import numpy as np

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 🔹 Метрики
# =========================

def compute_cosine_similarity(query_vector, doc_vectors):
    similarities = cosine_similarity([query_vector], doc_vectors)
    return similarities.flatten()


def recall_at_k(retrieved_indices, relevant_indices, k):
    if len(relevant_indices) == 0:
        return 0.0
    retrieved_k = retrieved_indices[:k]
    hits = len(set(retrieved_k) & set(relevant_indices))
    return hits / len(relevant_indices)


def mean_reciprocal_rank(retrieved_indices, relevant_indices):
    for rank, idx in enumerate(retrieved_indices, start=1):
        if idx in relevant_indices:
            return 1 / rank
    return 0.0


# =========================
# 🔹 Допоміжна функція
# =========================

def extract_preview_text(pdf_path, pages=3):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()[:pages]
    return "\n".join(d.page_content for d in docs)


# =========================
# 🔹 RAG Service
# =========================

class RAGService:

    def __init__(self, books_dir, db_dir):
        self.books_dir = books_dir
        self.db_dir = db_dir
        # Використання OpenAIEmbeddings для створення векторних представлень
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._load_or_create_db()

    # -------------------------

    def _load_or_create_db(self):
        """Завантажує існуючу базу Chroma або створює нову, якщо папка порожня."""
        if not os.path.exists(self.db_dir) or not os.listdir(self.db_dir):
            self.rebuild_from_folder()
        else:
            self.db = Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings
            )
            self._build_chain()

    # -------------------------

    def rebuild_from_folder(self):
        """Завантажує PDF, розбиває на частини та індексує в Chroma."""
        logging.info("🔄 Оновлення бази знань...")
        docs = []

        if not os.path.exists(self.books_dir):
            os.makedirs(self.books_dir)

        for file in os.listdir(self.books_dir):
            if file.endswith(".pdf"):
                docs.extend(
                    PyPDFLoader(os.path.join(self.books_dir, file)).load()
                )

        if not docs:
            logging.warning("⚠️ Не знайдено файлів для індексації.")
            return

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        ).split_documents(docs)

        self.db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )

        self._build_chain()

    # -------------------------

    def _build_chain(self):
        """Створює ланцюжок RAG для генерації відповідей."""
        retriever = self.db.as_retriever(search_kwargs={"k": 5})

        prompt = PromptTemplate.from_template("""
Ви спеціаліст із психосоматики. Надайте відповідь, спираючись на контекст.

КОНТЕКСТ:
{context}

ПИТАННЯ:
{question}
""")

        self.chain = (
                {
                    "context": retriever | (lambda d: "\n\n".join(x.page_content for x in d)),
                    "question": RunnablePassthrough()
                }
                | prompt
                | ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
                | StrOutputParser()
        )

    # -------------------------
    # 🔥 ОБМЕЖЕННЯ ТА БЕЗПЕКА (КОД-РІВЕНЬ)
    # -------------------------

    def ask(self, query: str) -> str:
        """
        Метод із вбудованим порогом релевантності 0.3 та автоматичним дисклеймером.
        """
        # 1. Пошук документів із розрахунком оцінки релевантності
        docs_with_scores = self.db.similarity_search_with_relevance_scores(query, k=5)

        # 2. Hard Constraint: Поріг 0.3
        # Якщо схожість знайдених документів менша за 0.3, бот відмовляє за шаблоном.
        threshold = 0.3
        if not docs_with_scores or docs_with_scores[0][1] < threshold:
            current_score = docs_with_scores[0][1] if docs_with_scores else 0
            logging.info(f"Відмова: низька релевантність {current_score:.4f} для запиту: {query}")

            return (
                "На жаль, у моїх ресурсах недостатньо інформації для точної відповіді на це питання. "
                "Я спеціалізуюся на психосоматиці. Рекомендую звернутися до профільного фахівця."
            )

        # 3. Генерація відповіді через LLM, якщо поріг пройдено
        response = self.chain.invoke(query)

        # 4. Програмне додавання дисклеймера (неможливо обійти через промт)
        disclaimer = (
            "\n\n---\n"
            "⚠️ *Важливо: Ця інформація носить ознайомлювальний характер і базується на літературі з психосоматики. "
            "Вона не є медичним діагнозом. У разі проблем зі здоров'ям зверніться до лікаря.*"
        )

        return f"{response}{disclaimer}"

    # -------------------------
    # 🔥 МЕТРИКИ
    # -------------------------
    def evaluate_retrieval(self, query, relevant_indices, k=5):
        """Оцінка якості пошуку за метриками Recall та MRR."""

        docs = self.db.similarity_search(query, k=k)
        query_vector = self.embeddings.embed_query(query)

        doc_texts = [doc.page_content for doc in docs]
        doc_vectors = np.array(self.embeddings.embed_documents(doc_texts))

        similarities = compute_cosine_similarity(query_vector, doc_vectors)

        # 🔥 порядок документів за релевантністю
        retrieved_indices = np.argsort(similarities)[::-1]

        return {
            "recall@k": float(recall_at_k(retrieved_indices, relevant_indices, k)),
            "mrr": float(mean_reciprocal_rank(retrieved_indices, relevant_indices)),
            "similarities": similarities.tolist(),
            "retrieved_order": retrieved_indices.tolist()
        }