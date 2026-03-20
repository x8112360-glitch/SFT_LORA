import json
import time
from openai import OpenAI

# DeepSeek API 配置
client = OpenAI(
    api_key="sk-2eb3da4fc82d40309d232d64f0c58dee",
    base_url="https://api.deepseek.com"
)

# 汽车制造领域的主题列表
TOPICS = [
    "发动机结构与原理",
    "变速箱工作原理",
    "车身焊接工艺",
    "汽车底盘系统",
    "制动系统原理",
    "汽车电气系统",
    "涡轮增压原理",
    "汽车质量检测标准",
    "冲压工艺流程",
    "整车装配流程",
    "发动机故障诊断",
    "新能源汽车电池系统",
    "汽车悬挂系统",
    "转向系统结构",
    "汽车涂装工艺",
]

# 让 DeepSeek 生成问答对的 prompt 模板
SYSTEM_PROMPT = """你是一个汽车制造领域的专家，请根据给定主题生成高质量的中文问答对。

要求：
1. 生成5个不同角度的问题，覆盖基础概念、工作原理、故障分析、工艺流程等角度
2. 每个回答要专业、准确、结构清晰，100-300字左右
3. 严格按照以下JSON格式输出，不要输出任何其他内容：

[
  {
    "instruction_zh": "问题内容",
    "input_zh": "",
    "output_zh": "回答内容"
  }
]"""

def generate_qa_for_topic(topic: str) -> list:
    """针对一个主题生成问答对"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"请围绕「{topic}」生成10个问答对"}
            ],
            temperature=0.8,
        )
        
        content = response.choices[0].message.content.strip()
        
        # 清理可能的 markdown 代码块标记
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        qa_list = json.loads(content)
        print(f"  ✅ 「{topic}」生成了 {len(qa_list)} 条")
        return qa_list
        
    except json.JSONDecodeError:
        print(f"  ⚠️ 「{topic}」JSON解析失败，跳过")
        return []
    except Exception as e:
        print(f"  ❌ 「{topic}」生成失败：{e}")
        return []

def main():
    all_data = []
    
    print(f"开始生成数据集，共 {len(TOPICS)} 个主题...")
    print("=" * 50)
    
    for i, topic in enumerate(TOPICS):
        print(f"[{i+1}/{len(TOPICS)}] 正在生成：{topic}")
        
        qa_list = generate_qa_for_topic(topic)
        all_data.extend(qa_list)
        
        # 每个主题之间暂停1秒，避免触发频率限制
        time.sleep(1)
    
    print("=" * 50)
    print(f"生成完成！共 {len(all_data)} 条数据")
    
    #生成新数据
    output_path = "./datasets/automotive_sft.json"

    # 读取已有数据
    existing_data = []
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        print(f"读取到已有数据 {len(existing_data)} 条")
    except FileNotFoundError:
        print("没有已有数据，从头开始")

    # 合并新旧数据
    all_data = existing_data + all_data

    # 保存合并后的数据
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"合并完成！共 {len(all_data)} 条数据")
    
    # 打印一条样本看看
    if all_data:
        print("\n样本预览：")
        print(json.dumps(all_data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()