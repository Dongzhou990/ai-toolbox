"""
RAG 问答 Chain — 检索 + 生成
"""

RAG_PROMPT = """你是一个专业的问答助手。请根据以下参考资料回答用户的问题。

## 规则
1. 只根据提供的参考资料回答，不要使用外部知识
2. 如果参考资料不足以回答问题，请明确告知
3. 在回答末尾列出引用的来源编号
4. 用中文回答

## 参考资料
{context}

## 用户问题
{question}

## 你的回答"""


class RAGChain:
    """RAG 问答链"""

    def __init__(self, vector_store, api_key=None, model="deepseek-v4-flash"):
        self.store = vector_store
        self.model = model

        from openai import OpenAI
        self.llm = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )

    def ask(self, question, top_k=3, verbose=True):
        """
        提问

        返回: {"answer": str, "sources": [...], "retrieved_chunks": [...]}
        """
        # 1. 检索
        results = self.store.search(question, top_k=top_k)

        if not results:
            return {
                "answer": "未找到相关文档。请先导入文档。",
                "sources": [],
                "retrieved_chunks": [],
            }

        # 2. 构建上下文
        context_parts = []
        sources = []
        for i, (doc, sim, meta) in enumerate(results):
            src = meta.get("source", "未知")
            context_parts.append("[来源" + str(i + 1) + " | " + src + "]\n" + doc)
            sources.append({"index": i + 1, "source": src, "similarity": sim, "text": doc[:200]})

        context = "\n\n---\n\n".join(context_parts)

        if verbose:
            print("\n检索到 " + str(len(results)) + " 个相关片段：")
            for i, (doc, sim, meta) in enumerate(results):
                src = meta.get("source", "?")
                print("  [" + str(i + 1) + "] " + src + " | 相似度: " + str(sim))
                print("     " + doc[:100] + "...")

        # 3. 生成
        prompt = RAG_PROMPT.format(context=context, question=question)

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": results,
        }
