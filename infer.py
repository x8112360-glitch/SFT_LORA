import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_path = "./models/Qwen2.5-0.5B-Instruct"
adapter_path = "./adapter"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def generate(model, prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # 只返回新生成的部分，不含输入
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

# 测试的问题
prompt = "### 指令:\n请根据以下背景知识回答问题\n\n### 输入:\n活塞环的作用是密封气缸，防止燃气泄漏\n\n### 回答:\n"

# 先测原始模型
print("=" * 40)
print("【原始模型】")
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(generate(base_model, prompt))

# 再测加了 adapter 的模型
print("=" * 40)
print("【SFT 微调后】")
sft_model = PeftModel.from_pretrained(base_model, adapter_path)
sft_model.eval()
print(generate(sft_model, prompt))