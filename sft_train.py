from datasets import load_dataset

dataset = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train")
print(dataset[0])  # 先看看数据长什么样