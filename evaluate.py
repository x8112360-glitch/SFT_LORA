from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_chinese import Rouge
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import jieba

model_path = "./models/Qwen2.5-0.5B-Instruct"
sft_adapter = "./adapter_automotive"
dpo_adapter = "./adapter_dpo"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 测试集：问题 + 参考答案（用 DPO 数据里的 chosen 作为参考答案）
import json
with open("./datasets/automotive_dpo.json", "r", encoding="utf-8") as f:
    dpo_data = json.load(f)

# 取前20条作为测试集
test_data = dpo_data[:20]
test_prompts    = [d["prompt"] for d in test_data]
reference_texts = [d["chosen"] for d in test_data]

def generate(model, prompt, max_new_tokens=300):
    text = f"### 指令:\n{prompt}\n\n### 回答:\n"
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
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

def compute_scores(predictions, references):
    # BLEU（用 jieba 分词）
    refs_tokenized  = [[list(jieba.cut(r))] for r in references]
    preds_tokenized = [list(jieba.cut(p)) for p in predictions]
    bleu = corpus_bleu(
        refs_tokenized, 
        preds_tokenized,
        smoothing_function=SmoothingFunction().method1
    )

    # ROUGE
    rouge = Rouge()
    refs_str  = [" ".join(jieba.cut(r)) for r in references]
    preds_str = [" ".join(jieba.cut(p)) for p in predictions]
    scores = rouge.get_scores(preds_str, refs_str, avg=True)

    return {
        "BLEU":     round(bleu * 100, 2),
        "ROUGE-1":  round(scores["rouge-1"]["f"] * 100, 2),
        "ROUGE-2":  round(scores["rouge-2"]["f"] * 100, 2),
        "ROUGE-L":  round(scores["rouge-l"]["f"] * 100, 2),
    }

print("加载模型中...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# 评估 SFT 模型
print("\n评估 SFT 模型中（共20条）...")
sft_model = PeftModel.from_pretrained(base_model, sft_adapter)
sft_model.eval()
sft_preds = [generate(sft_model, p) for p in test_prompts]
sft_scores = compute_scores(sft_preds, reference_texts)

# 评估 DPO 模型
print("评估 DPO 模型中（共20条）...")
dpo_model = PeftModel.from_pretrained(base_model, dpo_adapter)
dpo_model.eval()
dpo_preds = [generate(dpo_model, p) for p in test_prompts]
dpo_scores = compute_scores(dpo_preds, reference_texts)

# 打印对比结果
print("\n" + "=" * 40)
print(f"{'指标':<12} {'SFT':>10} {'DPO':>10} {'变化':>10}")
print("-" * 40)
for key in ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L"]:
    diff = dpo_scores[key] - sft_scores[key]
    arrow = "↑" if diff > 0 else "↓"
    print(f"{key:<12} {sft_scores[key]:>10} {dpo_scores[key]:>10} {arrow}{abs(diff):>8.2f}")
print("=" * 40)