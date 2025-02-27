import time
from openai import OpenAI


def call_with_messages(model, api_key, instruction, max_retries=5, delay=2):
    """
    使用OpenAI客户端调用API，并实现重试机制

    Args:
        api_key: API密钥
        model: 模型名称
        instruction: 用户指令/问题
        max_retries: 最大重试次数
        delay: 重试间隔时间(秒)

    Returns:
        成功时返回模型回复的内容，失败返回None
    """
    # 创建OpenAI客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 构建消息
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction},
    ]

    # 实现重试机制
    retries = 0
    while retries < max_retries:
        try:
            # 调用API
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                seed=2024,
                temperature=0.1,
            )

            # 提取回复内容
            if not completion or not completion.choices:
                raise ValueError("Response is empty or missing choices")

            return completion.choices[0].message.content

        except Exception as e:
            retries += 1
            print(f"Error: {e}")

            if retries < max_retries:
                print(f"Retrying... ({retries}/{max_retries})")
                time.sleep(delay)  # 等待一段时间再重试
            else:
                print("Max retries reached. Skipping this instruction.")
                return None
