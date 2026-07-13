"""
Agent 工具集 - 支持真实搜索 + 模拟数据
设置环境变量 USE_MOCK=true 使用模拟数据快速演示
"""

import os
import httpx
import re

# 模拟搜索结果（质量高、速度快，适合学习调试）
MOCK_SEARCH = {
    "AI Agent": """搜索 'AI Agent' 返回 5 条结果：

【什么是AI Agent？AI Agent综述，看这一篇就够了！ - 知乎】
   摘要: AI Agent（人工智能代理）是一种能够自主感知环境、进行决策并执行行动的智能实体。它具备目标导向性、自主性、反应性和社交能力，可以通过工具调用和规划来完成复杂任务。
   链接: https://zhuanlan.zhihu.com/p/189587795

【AI Agent (智能体) 教程 | 菜鸟教程】
   摘要: AI Agent 是基于大语言模型的智能系统，它不仅能理解自然语言，还能调用工具、执行多步推理。与普通聊天机器人不同，Agent 具有记忆、规划和行动能力。
   链接: https://www.runoob.com/ai-agent/ai-agent-tutorial.html

【2026国产AI Agent工具全景盘点 - CSDN】
   摘要: AI Agent 被认为是2025-2026年最重要的AI发展方向之一。从单一对话到自主执行任务，Agent 正在重塑人机交互方式。
   链接: https://blog.csdn.net/article/160622087

【AI智能体完整指南：十款最佳 AI Agent - AdsPower】
   摘要: AI Agent 的核心架构包括：感知模块、记忆系统、规划引擎、工具调用层。相比于传统聊天机器人，Agent 能够主动分解任务、调用外部API、并在多轮迭代中完成复杂目标。
   链接: https://www.adspower.net/blog/what-is-ai-agent

【AI Agent 最全学习路线｜从零到实战 - 知乎】
   摘要: 聊天机器人是"一问一答"的被动模式，而AI Agent则是"目标驱动"的主动模式。Agent 的核心循环是：观察(Observe) -> 思考(Think) -> 行动(Act) -> 观察结果，直到任务完成。
   链接: https://zhuanlan.zhihu.com/p/205024218""",
}


def web_search(query: str) -> str:
    """搜索网页（支持真实搜索和模拟数据）"""
    # 模拟模式
    if os.getenv("USE_MOCK", "true").lower() == "true":
        for key in MOCK_SEARCH:
            if key in query:
                return MOCK_SEARCH[key]
        return "搜索 '" + query + "' 返回 3 条结果：\n\n" + MOCK_SEARCH.get("AI Agent", "无结果")

    # 真实搜索模式
    return _real_search(query)


def _real_search(query: str) -> str:
    """使用必应搜索"""
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(
                "https://cn.bing.com/search",
                params={"q": query, "count": "10", "mkt": "zh-CN"},
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/120.0.0.0 Safari/537.36",
                    "Accept-Language": "zh-CN,zh;q=0.9",
                },
            )
            resp.raise_for_status()
            html = resp.text

        blocks = re.findall(r'<li class="b_algo"[^>]*>(.*?)</li>', html, re.DOTALL)
        output = []
        for block in blocks:
            h2 = re.search(r'<h2[^>]*>(.*?)</h2>', block, re.DOTALL)
            title = re.sub(r'<[^>]+>', '', h2.group(1) if h2 else '').strip()
            title = re.sub(r'\s+', ' ', title)

            link = re.search(r'href="(https?://[^"]+)"', block)
            url = link.group(1) if link else ''

            caption = re.search(
                r'<p[^>]*class="[^"]*b_lineclamp[^"]*"[^>]*>(.*?)</p>',
                block, re.DOTALL
            )
            desc = re.sub(r'<[^>]+>', '', caption.group(1) if caption else '').strip()
            desc = re.sub(r'\s+', ' ', desc)
            desc = re.sub(r'&ensp;|&#0183;|&nbsp;', ' ', desc)

            if not title or len(title) < 4 or not url:
                continue
            output.append("【" + title + "】\n   摘要: " + desc[:200] + "\n   链接: " + url)
            if len(output) >= 5:
                break

        if not output:
            return "搜索 '" + query + "' 未找到相关结果。"
        return "搜索 '" + query + "' 返回 " + str(len(output)) + " 条结果：\n\n" + "\n\n".join(output)

    except Exception as e:
        return "搜索出错: " + str(e)


def fetch_page(url: str) -> str:
    """获取网页内容"""
    if os.getenv("USE_MOCK", "true").lower() == "true":
        return "这是 " + url + " 的模拟内容。主要内容：AI Agent 是能够自主执行任务的智能系统，具备感知、推理、规划和行动能力。"

    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/120.0.0.0 Safari/537.36"
                },
            )
            resp.raise_for_status()
            html = resp.text

        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) > 2000:
            text = text[:2000] + "\n\n... (截断，共 " + str(len(text)) + " 字)"
        return text

    except Exception as e:
        return "获取页面出错: " + str(e)


available_tools = {
    "web_search": web_search,
    "fetch_page": fetch_page,
}


def execute_tool(name, input_str):
    func = available_tools.get(name)
    if func is None:
        return "错误：未知工具 '" + str(name) + "'"
    return func(input_str)
