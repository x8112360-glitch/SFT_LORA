import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

## 加载数据集
dataset = load_dataset("silk-road/alpaca-data-gpt4-chinese", 
                       split="train", 
                       cache_dir="./datasets")

def format_prompt(example):
    instruction = example["instruction_zh"]
    input_text  = example.get("input_zh", "")
    output_text = example["output_zh"]
    if input_text and input_text.strip():
        text = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 回答:\n{output_text}"
    else:
        text = f"### 指令:\n{instruction}\n\n### 回答:\n{output_text}"
    return {"text": text}

dataset = dataset.map(format_prompt)
dataset = dataset.select(range(2000))


## 加载模型
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

print("模型加载完成~")
# print(f"显存占用：{torch.cuda.memory_allocated() / 1024**3:.2f} GB")


## 配置Lora
# 1. 让量化后的模型可以接受训练
model = prepare_model_for_kbit_training(model)

# 2. 配置 LoRA
lora_config = LoraConfig(
    r=8,                  # rank，小显存用 8 就够
    lora_alpha=16,        # 缩放系数，一般是 r 的 2 倍
    target_modules=["q_proj", "v_proj"],  # 插入哪几层
    lora_dropout=0.05,    # 防过拟合
    bias="none",
    task_type="CAUSAL_LM",
)

# 3. 把 LoRA 插入模型
model = get_peft_model(model, lora_config)

# 实际训练了多少参数
# model.print_trainable_parameters() #trainable params: 540,672 || all params: 494,573,440 || trainable%: 0.1093


## 开始训练
training_args = SFTConfig(
    output_dir="./checkpoints",          # 训练过程中保存checkpoint的地方
    num_train_epochs=2,                  # 跑 2 轮
    per_device_train_batch_size=2,       # 每次喂 2 条，4060 安全值
    gradient_accumulation_steps=4,       # 积累 4 步再更新，等效 batch_size=8
    learning_rate=2e-4,                  # 学习率
    warmup_ratio=0.03,                   # 前 3% 的步数做热身
    lr_scheduler_type="cosine",          # 学习率按余弦曲线衰减
    logging_steps=10,                    # 每 10 步打印一次 loss
    save_steps=200,                      # 每 200 步保存一次 checkpoint
    bf16=True,   
    fp16=False,
    max_length=512,                  # 每条数据最长 512 个 token
    dataset_text_field="text",           # 告诉 trainer 用哪个字段
    report_to="none",                    # 不上传 wandb
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# 开始训练！
trainer.train()

# 训练完保存 adapter
trainer.model.save_pretrained("./adapter")
tokenizer.save_pretrained("./adapter")
print("LoRA adapter 保存完成！")



