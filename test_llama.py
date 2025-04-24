from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. 모델과 토크나이저 로드
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    output_hidden_states=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# 2. 입력 문장
text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# 3. 모델 실행
with torch.no_grad():
    outputs = model(**inputs)
    hidden_states = outputs.hidden_states  # len: 33 (embedding + 32 layers)
    lm_head = model.lm_head  # 마지막 linear layer (hidden_size → vocab_size)

# 4. 각 레이어에서 마지막 토큰 벡터로 다음 단어 예측
print(f"\n입력 문장: '{text}'")
print(f"next prediction:\n")

for layer_idx in range(1, len(hidden_states)):  # 1~32번 레이어 (0번은 embedding)
    layer_output = hidden_states[layer_idx]  # shape: [1, seq_len, hidden_size]
    last_token_vector = layer_output[:, -1, :]  # [1, hidden_size]

    # vocab logits 예측 (linear projection)
    # print(f"last token's representation vector: {last_token_vector}")
    # print(f"last token shape: {last_token_vector.shape}")
    logits = lm_head(last_token_vector)  # shape: [1, vocab_size]
    probs = torch.softmax(logits, dim=-1)
    top_token_id = torch.argmax(probs, dim=-1).item()
    confidence = probs[0, top_token_id].item()
    top_token = tokenizer.decode([top_token_id])

    print(f"Layer {layer_idx:>2}: {top_token.strip()}  ({confidence * 100:.2f}%)")