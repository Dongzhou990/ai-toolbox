FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt sentence-transformers numpy

# 复制代码
COPY . .
COPY ../research-agent/agent.py ./research-agent/
COPY ../research-agent/tools.py ./research-agent/
COPY ../rag-qa/loader.py ./rag-qa/
COPY ../rag-qa/vector_store.py ./rag-qa/
COPY ../rag-qa/qa_chain.py ./rag-qa/

# 创建目录
RUN mkdir -p /app/data /app/uploads

# 预下载模型（可选，加速首次启动）
RUN python3 -c "import os; os.environ['HF_ENDPOINT']='https://hf-mirror.com'; \
    from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8080

CMD ["python3", "server.py"]
