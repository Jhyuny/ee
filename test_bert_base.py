from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.functional import softmax
import pandas as pd
from tqdm import tqdm
import os

# Fine-tuned SST-2 모델과 토크나이저 로드
model_name = "JeremiahZ/bert-base-uncased-sst2"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# classification module
print("-"*20)
print(model.classifier)
print("-"*20)
model.eval()

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# SST-2 validation 데이터 로드
dataset = load_dataset("glue", "sst2", split="validation")

# 결과 저장 리스트
results = []

for item in tqdm(dataset, desc="Running SST-2 predictions"):
    text = item["sentence"]
    label = item["label"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()

    correct = int(pred == label)

    results.append({
        "Sentence": text,
        "ModelOutput": pred,
        "Label": label,
        "Correct": correct
    })

# CSV로 저장
df = pd.DataFrame(results)
os.makedirs("results", exist_ok=True)
df.to_csv("results/sst2_finetuned_predictions.csv", index=False)

# 정확도 출력
accuracy = df["Correct"].mean() * 100
print(f"\n✅ SST-2 Accuracy (fine-tuned BERT): {accuracy:.2f}%")
print("📁 'results/sst2_finetuned_predictions.csv'에 저장 완료!")