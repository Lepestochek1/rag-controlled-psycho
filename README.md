# PsychoGen-RAG: Controlled Neural Generation for Mental Health Support

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/framework-LangChain-green.svg)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the **Method of Controlled Neural Network Generation** for psychoeducational chatbot responses. The system utilizes a **Retrieval-Augmented Generation (RAG)** architecture to provide accurate, safe, and context-aware information support in the field of mental health and psychosomatics.

---

## Overview

In mental health applications, generic AI responses can pose risks. **PsychoGen-RAG** addresses this by implementing a controlled generation pipeline. Instead of relying solely on pre-trained knowledge, the system retrieves verified data and applies a semantic safety threshold to ensure responses are grounded in expert literature.

## Key Features

* **RAG-Enhanced Accuracy:** Uses a vector database (ChromaDB) to ground responses in expert literature, such as the works of Liz Burbo.
* **Multi-Stage Safety Control:** Implements a validation protocol ($S(y) \geq \theta$) to filter out irrelevant or harmful content.
* **Dual-Interface System:** Features a Telegram-based bot for users and an administrative module for real-time updates.
* **Automated Evaluation:** Includes a specialized module that scores responses based on relevance and quality.

---

## Our Methodology

The framework decomposes the complex task of generating safe psychoeducational content into three specialized stages:

### 1. Semantic Retrieval
When a user submits a query, the system converts it into a vector embedding. It then performs a semantic search across a specialized vector store using **Cosine Similarity** to find the most relevant fragments.

### 2. Controlled Prompt Engineering
The retrieved context is combined with the original query and passed through a specialized **Prompt Template**. This enforces the model's persona as a psychosomatic expert and defines strict boundaries for the response structure.

### 3. Verification & Safety Threshold
To ensure "controlled" generation, the output is evaluated by a safety function $S(y)$. The final quality is treated as an optimization problem:
$$F = \alpha R + \beta C + \gamma S$$
Where $R$ is relevance, $C$ is consistency, and $S$ is the safety score.

---

## System Architecture

The project is structured into three core functional modules:
1. **RAGService:** The central logic hub connecting the vector database and the LLM.
2. **EvaluationService:** The verification module that provides technical assessments of response quality.
3. **PsychoBot:** The interface controller for the Telegram API built on the `aiogram` framework.

---

## Performance Highlights

Technical testing on the psychosomatic domain yielded the following results:
* **Recall@3:** Achieved an average score of **0.7**, indicating high efficiency in retrieving relevant context.
* **Semantic Similarity:** Average cosine similarity for top results reached **0.474**.
* **Domain Filtering:** Successfully identified and rejected **100%** of out-of-domain queries.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Lepestochek1/rag-controlled-psycho.git](https://github.com/Lepestochek1/rag-controlled-psycho.git)
   cd rag-controlled-psycho
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
2. **Environment Setup:**

   Create a .env file in the root directory:
   ```bash
   TELEGRAM_BOT_TOKEN=your_token
   OPENAI_API_KEY=your_key
