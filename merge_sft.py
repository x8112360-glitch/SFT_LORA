import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_path = "./models/Qwen2.5-0.5B-Instruct"
sft_adapter = "./adapter_automotive"

print("加载模型和 SFT adapter...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,
    device_map="cpu",       # 合并操作在 CPU 上做更稳
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(model, sft_adapter)

print("合并 SFT adapter 进基础模型...")
# 合并完之后这样保存，强制忽略 peft_config
merged_model = model.merge_and_unload()

# 删掉残留的 peft_config，保存干净的模型
if hasattr(merged_model, 'peft_config'):
    del merged_model.peft_config

merged_model.save_pretrained("./models/Qwen2.5-0.5B-SFT-merged")
tokenizer.save_pretrained("./models/Qwen2.5-0.5B-SFT-merged")
print("干净的合并模型保存完成！")