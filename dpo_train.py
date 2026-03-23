import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import DPOTrainer, DPOConfig

# 加载 DPO 数据集
with open("./datasets/automotive_dpo.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# DPO 数据格式：prompt / chosen / rejected 三个字段
dataset = Dataset.from_list(raw_data)
print(f"DPO 数据集加载完成，共 {len(dataset)} 条")

# 加载基础模型
model_path = "./models/Qwen2.5-0.5B-Instruct"
adapter_path = "./adapter_automotive"   # 在 SFT 的基础上继续训

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# 加载 SFT 训练好的 adapter，在它基础上继续 DPO
model = prepare_model_for_kbit_training(model)
model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
print("SFT adapter 加载完成，开始 DPO 训练!")

# ── DPO 训练配置 ──
dpo_config = DPOConfig(
    output_dir="./checkpoints_dpo",
    num_train_epochs=3,
    per_device_train_batch_size=1,      # DPO 每条数据有 chosen+rejected 两条，显存压力更大
    gradient_accumulation_steps=8,      # 等效 batch_size=8
    learning_rate=5e-5,                 # DPO 学习率要比 SFT 小
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=5,
    save_steps=50,
    bf16=True,
    fp16=False,
    beta=0.1,                           # DPO 专有参数，控制偏好强度，0.1 是常用默认值
    max_length=512,
    # max_prompt_length=256,
    report_to="none",
)

trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# 开始训练
trainer.train()

# 保存 DPO adapter
trainer.model.save_pretrained("./adapter_dpo")
tokenizer.save_pretrained("./adapter_dpo")
print(" DPO adapter 保存完成！")