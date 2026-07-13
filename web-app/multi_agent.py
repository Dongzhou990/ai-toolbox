"""
多 Agent 协作：研究员 + 报告写手

流程：
  研究员: 搜索资料 → 提取关键信息 → 输出研究摘要
  写手:   接收研究摘要 → 组织语言 → 生成结构化报告
"""

import json
import os
import sys
import re

# 复用 research-agent 的工具
sys.path.insert(0, str(__file__).rsplit("/", 2)[0] + "/research-agent")
from tools import execute_tool, available_tools

RESEARCHER_PROMPT = """你是一个专业的研究员。你的任务是为给定的主题搜索资料，并提取关键信息。

格式：只输出 JSON

搜索:
{"action": "tool", "tool": "web_search", "input": "关键词"}

阅读网页:
{"action": "tool", "tool": "fetch_page", "input": "https://..."}

完成研究（输出研究摘要）:
{"action": "done", "findings": "你的研究发现（含关键数据和引用来源）"}

规则：做 3 次搜索后必须输出 done。
"""

WRITER_PROMPT = """你是一个资深科技报告写手。请根据研究员提供的研究摘要，生成一篇专业、结构清晰、有深度的报告。

格式：直接输出 Markdown 报告，不要 JSON。

要求：
- 报告标题用 #
- 分段清晰，使用 ## 和 ###
- 引用研究员提供的数据和来源
- 语言专业但不枯燥
- 结尾有一段总结或展望
"""


class MultiAgentSystem:
    """多 Agent 协作系统"""

    def __init__(self, api_key=None, model="deepseek-v4-flash"):
        from openai import OpenAI
        self.llm = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        self.model = model

    def run(self, topic: str):
        """执行多 Agent 协作流程，返回 (日志, 最终报告)"""
        log_lines = []

        # ============ Phase 1: 研究员 ============
        log_lines.append("=" * 50)
        log_lines.append("Phase 1: 研究员 Agent 开始搜索...")
        log_lines.append("=" * 50)

        findings = self._run_researcher(topic, log_lines)

        if not findings:
            log_lines.append("研究员未能获取有效信息")
            return "\n".join(log_lines), "研究未能完成"

        log_lines.append("\n研究员完成，输出研究摘要:")
        log_lines.append(findings[:500] + "..." if len(findings) > 500 else findings)

        # ============ Phase 2: 写手 ============
        log_lines.append("\n" + "=" * 50)
        log_lines.append("Phase 2: 写手 Agent 开始撰写报告...")
        log_lines.append("=" * 50)

        report = self._run_writer(findings, topic, log_lines)

        return "\n".join(log_lines), report

    def _run_researcher(self, topic: str, log_lines: list) -> str:
        messages = [
            {"role": "system", "content": RESEARCHER_PROMPT},
            {"role": "user", "content": "研究主题：" + topic + "\n请搜索资料并整理研究摘要。"},
        ]

        for step in range(1, 5):
            log_lines.append("\n研究员 第" + str(step) + "步")

            result = self._call_llm(messages)
            if not result:
                break

            action = result.get("action")

            if action == "done":
                return result.get("findings", "")

            elif action == "tool":
                tool_name = result.get("tool")
                tool_input = result.get("input", "")
                log_lines.append("  → 调用: " + str(tool_name) + "('" + str(tool_input) + "')")

                observation = execute_tool(tool_name, tool_input)
                log_lines.append("  → 结果: " + observation[:100] + "...")

                messages.append({
                    "role": "user",
                    "content": "[工具结果]\n" + observation + "\n\n请继续。如果信息足够，请用 action: done 输出研究发现。",
                })

            elif action in available_tools:
                tool_input = result.get("input", "")
                log_lines.append("  → 调用: " + str(action) + "('" + str(tool_input) + "')")
                observation = execute_tool(action, tool_input)
                messages.append({
                    "role": "user",
                    "content": "[工具结果]\n" + observation + "\n\n请继续或输出 done。",
                })

        # 强制输出
        messages.append({"role": "user", "content": "已达到最大步数。请立即用 action: done 输出你的研究发现。"})
        result = self._call_llm(messages)
        if result and result.get("action") == "done":
            return result.get("findings", "")
        return ""

    def _run_writer(self, findings: str, topic: str, log_lines: list) -> str:
        messages = [
            {"role": "system", "content": WRITER_PROMPT},
            {"role": "user", "content": "研究主题：" + topic + "\n\n研究员的研究摘要：\n" + findings + "\n\n请基于以上摘要，生成一份专业的 Markdown 报告。"},
        ]

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.4,
            max_tokens=2000,
        )
        report = response.choices[0].message.content.strip()
        log_lines.append("\n写手完成，报告长度: " + str(len(report)) + " 字符")
        return report

    def _call_llm(self, messages):
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1500,
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                if text.endswith("```"):
                    text = text[:-3]

            # 尝试 JSON
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                cleaned = re.sub(r'[\x00-\x1f\x7f]', '', text)
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    m = re.search(r'\{.*\}', text, re.DOTALL)
                    if m:
                        try:
                            return json.loads(re.sub(r'[\x00-\x1f\x7f]', '', m.group(0)))
                        except json.JSONDecodeError:
                            pass
                return None
        except Exception as e:
            return None
