"""
RAG 文档问答系统入口
"""

import os
import sys
from loader import load_file, load_folder, chunk_text
from vector_store import VectorStore
from qa_chain import RAGChain


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG 文档问答系统")
    parser.add_argument("--docs", "-d", help="文档路径（文件或文件夹）", default="sample.txt")
    parser.add_argument("--api-key", help="LLM API key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--chunk-size", type=int, default=500, help="分块大小")
    parser.add_argument("--top-k", type=int, default=3, help="检索数量")

    args = parser.parse_args()

    # 1. 加载文档
    print("=" * 60)
    print("RAG 文档问答系统")
    print("=" * 60)

    if os.path.isdir(args.docs):
        print("\n[1/4] 加载文件夹: " + args.docs)
        docs = load_folder(args.docs)
        all_text = "\n\n".join([t for _, t in docs])
    else:
        print("\n[1/4] 加载文档: " + args.docs)
        all_text = load_file(args.docs)

    print("  文档长度: " + str(len(all_text)) + " 字符")

    # 2. 分块
    print("\n[2/4] 文本分块 (size=" + str(args.chunk_size) + ")")
    chunks = chunk_text(all_text, chunk_size=args.chunk_size)
    print("  生成 " + str(len(chunks)) + " 个文本块")

    # 添加元数据
    metadata = []
    for i, chunk in enumerate(chunks):
        metadata.append({
            "source": os.path.basename(args.docs),
            "chunk_index": i,
            "length": len(chunk),
        })

    # 3. 向量化 & 存储
    print("\n[3/4] 向量化并存储到 ChromaDB ...")
    store = VectorStore(persist_dir="./chroma_db")
    store.create_collection("rag_docs")
    store.add_documents(chunks, metadata)
    print("  向量库统计: " + str(store.get_stats()))

    # 4. 交互问答
    print("\n[4/4] 初始化问答链 ...\n")

    if not args.api_key:
        print("未设置 API key。使用本地检索模式（只展示检索结果，不生成回答）\n")
        chain = None
    else:
        chain = RAGChain(store, api_key=args.api_key)

    print("=" * 60)
    print("开始提问！输入 'quit' 退出，'stats' 查看统计\n")

    while True:
        try:
            question = input("❓ 你的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() == "quit":
            print("再见！")
            break
        if question.lower() == "stats":
            print(store.get_stats())
            continue

        if chain:
            result = chain.ask(question, top_k=args.top_k)
            print("\n" + "=" * 40)
            print("📝 回答:")
            print("=" * 40)
            print(result["answer"])
            print()
        else:
            # 无 API key：只展示检索结果
            results = store.search(question, top_k=args.top_k)
            print("\n检索结果（本地模式，无 LLM 回答）:")
            for i, (doc, dist, meta) in enumerate(results):
                print("\n  [" + str(i + 1) + "] " + meta.get("source", "?") + ":")
                print("  " + doc[:200] + "...")
            print()


if __name__ == "__main__":
    main()
