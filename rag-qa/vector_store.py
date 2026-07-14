"""
向量存储模块 — 支持语义向量 (sentence-transformers) + TF-IDF 双模式
"""

import os
import math
import re
import numpy as np


def tokenize(text: str) -> list:
    """简单中英文分词"""
    text = text.lower()
    return re.findall(r'[\u4e00-\u9fa5]|[a-zA-Z]+|[0-9]+', text)


# ==================== TF-IDF 模式 ====================

def compute_tfidf(documents: list) -> dict:
    N = len(documents)
    doc_freq = {}
    tokenized_docs = []
    for doc in documents:
        tokens = tokenize(doc)
        tokenized_docs.append(tokens)
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    vocab = {word: idx for idx, word in enumerate(sorted(doc_freq.keys()))}
    idf = {word: math.log((N + 1) / (df + 1)) + 1 for word, df in doc_freq.items()}
    vectors = []
    for tokens in tokenized_docs:
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vec = [0.0] * len(vocab)
        for word, count in tf.items():
            if word in vocab:
                vec[vocab[word]] = (count / len(tokens)) * idf.get(word, 0)
        vectors.append(vec)
    return {"idf": idf, "vectors": vectors, "vocab": vocab}


def cosine_similarity(vec_a, vec_b):
    a = np.array(vec_a); b = np.array(vec_b)
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(dot / (na * nb)) if na and nb else 0.0


# ==================== 语义向量模式 ====================

class EmbeddingModel:
    """sentence-transformers 封装（懒加载）"""
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            from sentence_transformers import SentenceTransformer
            print("  加载语义模型: all-MiniLM-L6-v2 ...")
            cls._instance = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._instance


# ==================== VectorStore ====================

class VectorStore:
    """
    RAG 向量存储，支持两种模式：
    - semantic: sentence-transformers 语义向量（默认，精度高）
    - tfidf:    TF-IDF 词频向量（零依赖，轻量）
    """

    def __init__(self, persist_dir: str = None, mode: str = None):
        self.mode = mode or os.getenv("VECTOR_MODE", "semantic")
        self.documents = []
        self.metadata = []
        self.tfidf_data = None
        self.embeddings = None
        self.model = None

    def create_collection(self, name: str = "rag_docs"):
        print(f"向量集合 '{name}' 已创建（模式: {self.mode}）")

    def add_documents(self, chunks: list, metadata: list = None):
        self.documents = list(chunks)
        self.metadata = list(metadata) if metadata else [{}] * len(chunks)

        if self.mode == "semantic":
            self.model = EmbeddingModel.get()
            self.embeddings = self.model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
            print(f"已添加 {len(chunks)} 个文本块（语义向量, {self.embeddings.shape[1]} 维）")
        else:
            self.tfidf_data = compute_tfidf(chunks)
            print(f"已添加 {len(chunks)} 个文本块（TF-IDF, {len(self.tfidf_data['vocab'])} 词）")

    def search(self, query: str, top_k: int = 3) -> list:
        if not self.documents:
            return []

        if self.mode == "semantic" and self.embeddings is not None:
            return self._search_semantic(query, top_k)
        else:
            return self._search_tfidf(query, top_k)

    def _search_semantic(self, query: str, top_k: int) -> list:
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        scores = []
        for i, doc_vec in enumerate(self.embeddings):
            sim = float(np.dot(query_vec, doc_vec))
            scores.append((i, sim))
        scores.sort(key=lambda x: x[1], reverse=True)

        output = []
        for idx, sim in scores[:top_k]:
            if sim > 0:
                output.append((
                    self.documents[idx],
                    round(sim, 4),
                    self.metadata[idx] if idx < len(self.metadata) else {},
                ))
        return output

    def _search_tfidf(self, query: str, top_k: int) -> list:
        words = tokenize(query)
        vocab = self.tfidf_data["vocab"]
        idf = self.tfidf_data["idf"]
        query_vec = [0.0] * len(vocab)
        tf = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1
        for word, count in tf.items():
            if word in vocab:
                query_vec[vocab[word]] = (count / len(words)) * idf.get(word, 0)
        scores = [(i, cosine_similarity(query_vec, dv)) for i, dv in enumerate(self.tfidf_data["vectors"])]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            (self.documents[idx], round(sim, 4), self.metadata[idx] if idx < len(self.metadata) else {})
            for idx, sim in scores[:top_k] if sim > 0
        ]

    def get_stats(self) -> dict:
        return {
            "count": len(self.documents),
            "mode": self.mode,
            "dim": int(self.embeddings.shape[1]) if self.embeddings is not None else
                   len(self.tfidf_data["vocab"]) if self.tfidf_data else 0,
        }
