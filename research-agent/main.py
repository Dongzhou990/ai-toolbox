"""研究 Agent 入口"""

import os
import sys


def main():
    # 从命令行参数获取研究主题，或交互式输入
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        topic = input("请输入研究主题: ").strip()
        if not topic:
            print("请输入有效的研究主题！")
            sys.exit(1)

    # 初始化 Agent
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("⚠️  未设置 API key，进入演示模式\n")
        demo_flow(topic)
        return

    # 有 API key 才导入，避免依赖问题
    from agent import ResearchAgent

    agent = ResearchAgent(api_key=api_key)
    report = agent.run(topic)

    print("\n" + "=" * 60)
    print("📄 研究报告")
    print("=" * 60 + "\n")
    print(report)

    # 保存报告
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"research_{topic[:20].replace(' ', '_')}.md"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# 研究报告：{topic}\n\n")
        f.write(report)

    print(f"\n💾 报告已保存到: {filepath}")


def demo_flow(topic: str):
    """无 API key 时的流程演示"""
    print(f"  研究主题: {topic}\n")
    print("""
┌─────────────────────────────────────────────────────────────┐
│  Agent 工作流程演示（ReAct 模式）                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  第 1 步: LLM 思考 "我需要搜索关于这个主题的最新资料"             │
│           → 调用工具 web_search("主题关键词")                   │
│           → 拿到 5 条搜索结果                                   │
│                                                             │
│  第 2 步: LLM 分析结果 "第一条和第三条看起来有用"                  │
│           → 调用工具 fetch_page("url")                         │
│           → 获取网页全文（截取前 2000 字）                       │
│                                                             │
│  第 3 步: LLM 思考 "信息还不够全，换个角度再搜"                   │
│           → 调用工具 web_search("另一个角度关键词")               │
│           → 拿到新的搜索结果                                    │
│                                                             │
│  第 4 步: LLM 判断 "已经收集足够信息，可以生成报告"               │
│           → 输出 action: "answer"                             │
│           → 生成带引用的 Markdown 研究报告                      │
│                                                             │
│  循环结束 ✅                                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  关键区别                                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  普通聊天:  用户问 → AI 答 → 结束                              │
│                                                             │
│  Agent:    用户给目标 → AI 规划 → 调工具 → 看结果               │
│            → 再规划 → 再调工具 → ... 循环直到完成               │
│                                                             │
│  这就是 ReAct 模式的精髓：                                     │
│  Reasoning（推理）+ Acting（行动）= 自主完成复杂任务              │
│                                                             │
└─────────────────────────────────────────────────────────────┘

启动真实 Agent：
  export OPENAI_API_KEY=你的key
  python main.py "你想研究的话题"

依赖安装:
  pip install -r requirements.txt
""")


if __name__ == "__main__":
    main()
