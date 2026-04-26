# PsychoGen-RAG: Controlled Neural Generation for Mental Health Support

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/framework-LangChain-green.svg)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[cite_start]This repository contains the official implementation of the **Method of Controlled Neural Network Generation** for psychoeducational chatbot responses[cite: 729]. [cite_start]The system utilizes a **Retrieval-Augmented Generation (RAG)** architecture to provide accurate, safe, and context-aware information support in the field of mental health and psychosomatics[cite: 815, 820].

> [cite_start]**Note:** This project was developed as a Bachelor's Thesis at Khmelnytskyi National University (2026)[cite: 695].

---

## 📋 Overview

In mental health applications, generic AI responses can pose risks. [cite_start]**PsychoGen-RAG** addresses this by implementing a controlled generation pipeline[cite: 780, 908]. [cite_start]Instead of relying solely on pre-trained knowledge, the system retrieves verified data and applies a semantic safety threshold to ensure responses are grounded in expert literature[cite: 820, 1094].

## ✨ Key Features

* [cite_start]**RAG-Enhanced Accuracy:** Uses a vector database (ChromaDB) to ground responses in expert literature, such as the works of Liz Burbo[cite: 1012, 1085].
* [cite_start]**Multi-Stage Safety Control:** Implements a validation protocol ($S(y) \geq \theta$) to filter out irrelevant or harmful content[cite: 914, 969].
* [cite_start]**Dual-Interface System:** Features a Telegram-based bot for users and an administrative module for real-time updates[cite: 1112, 1176].
* [cite_start]**Automated Evaluation:** Includes a specialized module that scores responses based on relevance and quality[cite: 1104, 1141].

---

## 🧠 Our Methodology

[cite_start]The framework decomposes the complex task of generating safe psychoeducational content into three specialized stages[cite: 933]:

### 1. Semantic Retrieval
[cite_start]When a user submits a query, the system converts it into a vector embedding[cite: 935, 953]. [cite_start]It then performs a semantic search across a specialized vector store using **Cosine Similarity** to find the most relevant fragments[cite: 938, 958].

### 2. Controlled Prompt Engineering
[cite_start]The retrieved context is combined with the original query and passed through a specialized **Prompt Template**[cite: 941, 1088]. [cite_start]This enforces the model's persona as a psychosomatic expert and defines strict boundaries for the response structure[cite: 1173, 1174].

### 3. Verification & Safety Threshold
[cite_start]To ensure "controlled" generation, the output is evaluated by a safety function $S(y)$[cite: 909, 1005]. The final quality is treated as an optimization problem:
$$F = \alpha R + \beta C + \gamma S$$
[cite_start]Where $R$ is relevance, $C$ is consistency, and $S$ is the safety score[cite: 920, 1007].

---

## 🏗 System Architecture

[cite_start]The project is structured into three core functional modules[cite: 1103]:
1.  [cite_start]**RAGService:** The central logic hub connecting the vector database and the LLM[cite: 1108].
2.  [cite_start]**EvaluationService:** The verification module that provides technical assessments of response quality[cite: 1104].
3.  [cite_start]**PsychoBot:** The interface controller for the Telegram API built on the `aiogram` framework[cite: 1078, 1112].

---

## 📊 Performance Highlights

Technical testing on the psychosomatic domain yielded the following results:
* [cite_start]**Recall@3:** Achieved an average score of **0.7**, indicating high efficiency in retrieving relevant context[cite: 1206].
* [cite_start]**Semantic Similarity:** Average cosine similarity for top results reached **0.474**[cite: 1218].
* [cite_start]**Domain Filtering:** Successfully identified and rejected **100%** of out-of-domain queries[cite: 1212, 1215].

---

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Lepestochek1/rag-controlled-psycho.git](https://github.com/Lepestochek1/rag-controlled-psycho.git)
    cd rag-controlled-psycho
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Setup:**
    Create a `.env` file in the root directory:
    ```env
    TELEGRAM_BOT_TOKEN=your_token
    OPENAI_API_KEY=your_key
    ```

---

## 🤝 Acknowledgements

* **Thesis Supervisor:** Ph.D., Assoc. [cite_start]Prof. Oleksandr Mazurets[cite: 731].
* [cite_start]**Institution:** Khmelnytskyi National University, Department of Computer Science[cite: 695, 698].
* [cite_start]**Data Source:** Inspired by the psychosomatic research of Liz Burbo[cite: 1012].

---

### Contact
**Tetiana Kashperuk** — [GitHub](https://github.com/Lepestochek1)
