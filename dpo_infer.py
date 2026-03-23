import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_path = "./models/Qwen2.5-0.5B-Instruct"
sft_adapter = "./adapter_automotive"
dpo_adapter = "./adapter_dpo"

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

# 测试问题，选几个有代表性的
questions = [
    "发动机气缸体铸造的关键步骤是什么？",
    "汽车制动系统出现刹车跑偏应该怎么排查？",
    "简单解释一下什么是涡轮增压",
]

print("加载模型中...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

sft_model = PeftModel.from_pretrained(base_model, sft_adapter)
sft_model.eval()

# 加载 DPO 模型（在 SFT 基础上再套一层）
dpo_model = PeftModel.from_pretrained(base_model, dpo_adapter)
dpo_model.eval()

for q in questions:
    prompt = f"### 指令:\n{q}\n\n### 回答:\n"
    print("\n" + "=" * 60)
    print(f"问题：{q}")
    print("\n【SFT 模型回答】")
    print(generate(sft_model, prompt))
    print("\n【DPO 模型回答】")
    print(generate(dpo_model, prompt))