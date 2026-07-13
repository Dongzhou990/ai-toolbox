"""
向量存储模块 — TF-IDF + 余弦相似度
零外部依赖，纯 Python 实现，核心原理与真正的 Embedding 一致
"""

import math
import re


def tokenize(text: str) -> list:
    """简单中文/英文分词"""
    # 中文按字 + 英文按词
    text = text.lower()
    # 保留中文、英文、数字
    tokens = re.findall(r'[\u4e00-\u9fa5]|[a-zA-Z]+|[0-9]+', text)
    return tokens


def compute_tfidf(documents: list) -> dict:
    """
    计算 TF-IDF 矩阵

    返回: {
        "idf": {词: idf值},
        "vectors": [[doc1_tfidf_vec], [doc2_tfidf_vec], ...],
        "vocab": {词: 索引}
    }
    """
    N = len(documents)
    # 统计每个词的文档频率
    doc_freq = {}
    tokenized_docs = []

    for doc in documents:
        tokens = tokenize(doc)
        tokenized_docs.append(tokens)
        unique = set(tokens)
        for token in unique:
            doc_freq[token] = doc_freq.get(token, 0) + 1

    # 构建词汇表
    vocab = {word: idx for idx, word in enumerate(sorted(doc_freq.keys()))}

    # 计算 IDF
    idf = {}
    for word, df in doc_freq.items():
        idf[word] = math.log((N + 1) / (df + 1)) + 1

    # 计算每个文档的 TF-IDF 向量
    vectors = []
    for tokens in tokenized_docs:
        # TF
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        # TF-IDF
        vec = [0.0] * len(vocab)
        for word, count in tf.items():
            if word in vocab:
                vec[vocab[word]] = (count / len(tokens)) * idf.get(word, 0)
        vectors.append(vec)

    return {
        "idf": idf,
        "vectors": vectors,
        "vocab": vocab,
    }


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """余弦相似度"""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore:
    """
    RAG 向量存储

    使用 TF-IDF 将文本转为向量，用余弦相似度做检索。
    原理和真实 Embedding 一样，只是用传统方法代替神经网络。
    """

    def __init__(self, persist_dir: str = None):
        self.documents = []
        self.metadata = []
        self.tfidf_data = None
        self.query_tokens = None

    def create_collection(self, name: str = "rag_docs"):
        print("向量集合 '" + name + "' 已创建（TF-IDF 模式）")

    def add_documents(self, chunks: list, metadata: list = None):
        """添加文档并建立索引"""
        self.documents = list(chunks)
        self.metadata = list(metadata) if metadata else [{}] * len(chunks)
        self.tfidf_data = compute_tfidf(chunks)
        print("已添加 " + str(len(chunks)) + " 个文本块到向量库（TF-IDF 索引）")
        print("  词汇量: " + str(len(self.tfidf_data["vocab"])))

    def search(self, query: str, top_k: int = 3) -> list:
        """
        搜索最相关的文本块

        返回: [(文本块, 相似度分数, 元数据), ...]
        """
        if not self.documents:
            return []

        # 将查询转为 TF-IDF 向量
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

        # 计算与所有文档的相似度
        scores = []
        for i, doc_vec in enumerate(self.tfidf_data["vectors"]):
            sim = cosine_similarity(query_vec, doc_vec)
            scores.append((i, sim))

        # 排序取 top_k
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

    def get_stats(self) -> dict:
        return {
            "count": len(self.documents),
            "vocab_size": len(self.tfidf_data["vocab"]) if self.tfidf_data else 0,
        }
