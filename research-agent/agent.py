"""
ReAct Agent —— 自动化研究员
"""

import json
import re
from tools import available_tools, execute_tool

SYSTEM_PROMPT = """你是一个自动化研究 Agent。你需要根据用户的研究主题，搜索网页收集资料，然后生成研究报告。

## 重要：每次回复必须使用以下 JSON 格式（只输出 JSON，不要有其他内容）

当你需要搜索信息时：
{"action": "tool", "tool": "web_search", "input": "搜索关键词"}

当你需要查看某个网页的详细内容时：
{"action": "tool", "tool": "fetch_page", "input": "https://..."}

当你收集到足够信息，可以写报告时（content 内用 \\n 表示换行，不要有真正的换行）：
{"action": "answer", "content": "你的研究报告内容"}

## 规则
1. 每次只输出一个 JSON，不要有多余文字
2. 最多做 3 次搜索，然后必须给出答案
3. answer 的 content 中不要包含真正的换行符，用 \\n 代替
"""


class ResearchAgent:
    def __init__(self, api_key=None, model="deepseek-v4-flash"):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        self.model = model
        self.messages = []
        self.max_steps = 5
        self.api_key = api_key

    def run(self, topic):
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "研究主题：" + topic + "\n\n请先搜索收集资料，然后生成一份研究报告。"},
        ]

        print("\n" + "=" * 60)
        print("研究主题: " + topic)
        print("Agent 模型: " + self.model)
        print("=" * 60 + "\n")

        for step in range(1, self.max_steps + 1):
            print("--- 第 " + str(step) + "/" + str(self.max_steps) + " 步 ---")

            if step == self.max_steps:
                msg = "已达到最大步数。请基于已有信息，立即用 action: answer 输出最终报告。"
                self.messages.append({"role": "user", "content": msg})

            result = self._call_llm()
            if result is None:
                print("LLM 异常")
                return "研究失败：LLM 调用出错"

            action = result.get("action")

            if action == "answer":
                print("研究完成！\n")
                content = result.get("content", "")
                content = content.replace("\\n", "\n")
                return content

            elif action == "tool":
                tool_name = result.get("tool")
                tool_input = result.get("input", "")
                print("调用: " + str(tool_name) + "('" + str(tool_input) + "')")
                observation = execute_tool(tool_name, tool_input)
                print("结果(" + str(len(observation)) + "字): " + observation[:150] + "...\n")

                feedback = "[工具 " + str(tool_name) + " 结果]\n" + observation
                feedback += "\n\n继续下一步。最后请用 action: answer 输出报告。"
                self.messages.append({"role": "user", "content": feedback})

            elif action in available_tools:
                tool_input = result.get("input", result.get("query", ""))
                print("(兼容) 调用: " + str(action) + "('" + str(tool_input) + "')")
                observation = execute_tool(action, tool_input)
                print("结果(" + str(len(observation)) + "字): " + observation[:150] + "...\n")

                feedback = "[工具 " + str(action) + " 结果]\n" + observation
                feedback += "\n\n继续。最后请用 action: answer 输出报告。"
                self.messages.append({"role": "user", "content": feedback})

            else:
                print("未知行动: " + json.dumps(result, ensure_ascii=False)[:200])
                self.messages.append({
                    "role": "user",
                    "content": "格式错误。请用 action: tool 或 action: answer"
                })

        return "研究超时。"

    def _call_llm(self):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=0.3,
                max_tokens=2000,
            )
            text = response.choices[0].message.content.strip()

            # 去掉 code block
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                if text.endswith("```"):
                    text = text[:-3]
            text = text.strip()

            # 尝试标准解析
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            # 清理控制字符后重试
            cleaned = re.sub(r'[\x00-\x1f\x7f]', '', text)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

            # 提取 {...} 再试
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                block = match.group(0)
                block = re.sub(r'[\x00-\x1f\x7f]', '', block)
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    pass

            print("JSON 解析失败，原始返回(前300): " + text[:300])
            return None

        except Exception as e:
            print("LLM 调用失败: " + str(e))
            return None
