"""
口碑助手 — AI 评价回复生成
支持 8 种风格，适用于大众点评 / 美团 / 小红书等平台
"""

import os

STYLES = {
    "sincere": {
        "name": "真诚道歉",
        "icon": "🙏",
        "prompt": "以真诚道歉的口吻回复，承认问题所在，表达改正的决心，语气诚恳但不卑微。",
    },
    "compensate": {
        "name": "补偿安抚",
        "icon": "🎁",
        "prompt": "以安抚为主，主动提出补偿方案（如赠送服务、折扣等），让客人感到被重视。语气温暖，承诺具体行动。",
    },
    "improve": {
        "name": "改进承诺",
        "icon": "💪",
        "prompt": "重点表达会认真改进，列出具体的整改措施。让其他客人看到门店在进步。语气专业有担当。",
    },
    "professional": {
        "name": "专业得体",
        "icon": "✨",
        "prompt": "以专业的美业从业者身份回复，客观说明原因，感谢反馈，展现门店的专业素养。语气冷静有礼。",
    },
    "gentle": {
        "name": "温和亲切",
        "icon": "🌸",
        "prompt": "像朋友一样亲切地回复，用温暖的语言化解不满。语气自然不做作，让人感到被理解。",
    },
    "thanks": {
        "name": "热情感谢",
        "icon": "❤️",
        "prompt": "热情感谢客人的好评，表达被认可的开心，同时强调会继续保持。适合回复好评。",
    },
    "friendly": {
        "name": "朋友互动",
        "icon": "💬",
        "prompt": "像朋友聊天一样轻松回复好评，提到客人说的具体细节，让回复有个性不模板化。适合回复好评。",
    },
    "revisit": {
        "name": "邀请复购",
        "icon": "🔄",
        "prompt": "在感谢好评的同时，巧妙邀请客人再次光临，可以提到新品或优惠活动。适合回复好评。",
    },
}

SYSTEM_PROMPT = """你是一个专业的美业（美容/美甲/美发）门店口碑管理专家。

你的任务是为门店生成对客人评价的回复。要求：
1. 回复长度适中（80-200字），不要太长
2. 称呼客人时用"亲"或"您"
3. 如果是差评，不要辩解或推卸责任
4. 提到具体的服务细节会加分
5. 语气自然，像真人写的，不要像模板
6. 用中文回复
7. 只输出回复内容，不要有任何前缀说明"""


def generate_reply(api_key: str, review_text: str, style: str, rating: int = 3) -> str:
    """
    生成评价回复

    参数:
        api_key: LLM API key
        review_text: 客人评价原文
        style: 风格 key（如 sincere, compensate 等）
        rating: 评分 1-5（1-2=差评, 3=中评, 4-5=好评）

    返回:
        生成的回复文本
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    style_info = STYLES.get(style, STYLES["sincere"])
    rating_label = {1: "很差", 2: "较差", 3: "一般", 4: "好评", 5: "非常好"}.get(rating, "一般")

    user_prompt = f"""客人评分: {rating}/5 ({rating_label})
客人评价:
「{review_text}」

回复风格: {style_info['name']}
风格要求: {style_info['prompt']}

请生成回复："""

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=500,
    )

    return response.choices[0].message.content.strip()
