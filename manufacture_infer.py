import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_path = "./models/Qwen2.5-0.5B-Instruct"
adapter_path = "./adapter_automotive"   # 换成新的adapter

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def generate(model, prompt, max_new_tokens=300):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

# 用训练数据里见过的领域来测
questions = [
    "请解释涡轮增压发动机的工作原理",
    "汽车冲压工艺流程包括哪些步骤",
    "新能源汽车电池管理系统的主要功能是什么",
]

# 加载原始模型
print("加载模型中...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# 加载微调后的模型
sft_model = PeftModel.from_pretrained(base_model, adapter_path)
sft_model.eval()

for q in questions:
    prompt = f"### 指令:\n{q}\n\n### 回答:\n"
    print("=" * 50)
    print(f"问题：{q}")
    print(f"\n【微调后回答】")
    print(generate(sft_model, prompt))
    print()