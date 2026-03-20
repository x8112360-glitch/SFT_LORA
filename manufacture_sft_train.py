import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ── 加载本地 JSON 数据集 ──
with open("./datasets/automotive_sft.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

def format_prompt(example):
    instruction = example["instruction_zh"]
    input_text  = example.get("input_zh", "")
    output_text = example["output_zh"]
    if input_text and input_text.strip():
        text = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 回答:\n{output_text}"
    else:
        text = f"### 指令:\n{instruction}\n\n### 回答:\n{output_text}"
    return {"text": text}

# 转成 HuggingFace Dataset 格式
dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_prompt)
print(f"数据集加载完成，共 {len(dataset)} 条")

# ── 加载模型 ──
model_name = "./models/Qwen2.5-0.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
print(f"模型加载完成!")

# ── 配置 LoRA ──
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── 训练配置 ──
training_args = SFTConfig(
    output_dir="./checkpoints",
    num_train_epochs=5,              # 数据少就多跑几轮
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=5,                 # 数据少，步数少，5步打一次log
    save_steps=50,
    bf16=True,
    fp16=False,
    max_length=512,
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# ── 开始训练 ──
trainer.train()

# ── 保存新的 adapter ──
trainer.model.save_pretrained("./adapter_automotive")
tokenizer.save_pretrained("./adapter_automotive")
print("汽车领域 LoRA adapter 保存完成！")