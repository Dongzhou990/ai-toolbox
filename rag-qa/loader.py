"""
文档加载和分块模块
支持: .txt, .pdf, .md
"""

import os
import re


def load_file(filepath: str) -> str:
    """加载单个文件，返回文本内容"""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".txt" or ext == ".md":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    else:
        raise ValueError("不支持的文件格式: " + ext)


def load_folder(folder: str) -> list:
    """加载整个文件夹，返回 [(文件名, 文本), ...]"""
    docs = []
    for root, dirs, files in os.walk(folder):
        for fname in files:
            fpath = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext in (".txt", ".md", ".pdf"):
                try:
                    text = load_file(fpath)
                    docs.append((fpath, text))
                except Exception as e:
                    print("跳过 " + fname + ": " + str(e))
    return docs


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """
    将文本切分为块，带重叠

    参数:
        text: 原文
        chunk_size: 每块最大字符数
        overlap: 相邻块之间的重叠字符数

    返回:
        文本块列表
    """
    # 先按段落切分，避免在段落中间切断
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) <= chunk_size:
            current += ("\n\n" + para) if current else para
        else:
            if current:
                chunks.append(current)

            # 如果段落本身超过 chunk_size，进一步切分
            if len(para) > chunk_size:
                sub_chunks = _split_long_text(para, chunk_size, overlap)
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list:
    """按字符切分超长文本"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
