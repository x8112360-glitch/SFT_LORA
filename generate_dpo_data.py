import json
import time
from openai import OpenAI

client = OpenAI(
    api_key="sk-2eb3da4fc82d40309d232d64f0c58dee",
    base_url="https://api.deepseek.com"
)

# 直接复用 SFT 数据集里的问题作为 prompt
with open("./datasets/automotive_sft.json", "r", encoding="utf-8") as f:
    sft_data = json.load(f)

# 只取前120条的问题来生成 DPO 数据
prompts = [item["instruction_zh"] for item in sft_data[:120]]

SYSTEM_PROMPT = """你是一个汽车制造领域的专家。对于给定的问题，请生成两个质量差异明显的回答：
- chosen：准确、专业、结构清晰、分点说明、使用正确术语
- rejected：有技术错误、表述模糊、结构混乱、遗漏重要内容

要求差别要足够明显，让人一眼就能看出好坏。

严格按照以下JSON格式输出，不要输出任何其他内容：
{
  "chosen": "好的回答内容",
  "rejected": "差的回答内容"
}"""

def generate_dpo_pair(prompt: str) -> dict | None:
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"问题：{prompt}"}
            ],
            temperature=0.8,
        )

        content = response.choices[0].message.content.strip()

        # 清理 markdown 代码块
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        pair = json.loads(content)

        # 确保两个字段都有值
        if not pair.get("chosen") or not pair.get("rejected"):
            print(f"  ⚠️ 字段缺失，跳过")
            return None

        return {
            "prompt": prompt,
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
        }

    except json.JSONDecodeError:
        print(f"  ⚠️ JSON解析失败，跳过")
        return None
    except Exception as e:
        print(f"  ❌ 生成失败：{e}")
        return None

def main():
    # 读取已有进度
    output_path = "./datasets/automotive_dpo.json"
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            all_pairs = json.load(f)
        print(f"读取到已有进度 {len(all_pairs)} 条，继续生成...")
    except FileNotFoundError:
        all_pairs = []

    # 跳过已经生成过的
    done_prompts = {item["prompt"] for item in all_pairs}
    remaining = [p for p in prompts if p not in done_prompts]
    print(f"剩余待生成：{len(remaining)} 条")

    for i, prompt in enumerate(remaining):
        print(f"[{i+1}/{len(remaining)}] {prompt[:30]}...")

        pair = generate_dpo_pair(prompt)
        if pair:
            all_pairs.append(pair)
            # 每生成一条就保存一次，防止中途崩了全丢
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_pairs, f, ensure_ascii=False, indent=2)
            print(f"  ✅ 已保存，当前共 {len(all_pairs)} 条")

        time.sleep(1)

    print(f"生成完成！共 {len(all_pairs)} 条 DPO 数据")

if __name__ == "__main__":
    main()