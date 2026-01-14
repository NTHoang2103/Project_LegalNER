from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from entity_utils import group_entities, normalize_entities
from summary_utils import summarize_clause
# LOAD MODEL & TOKENIZER
MODEL_DIR = "./models/ner_phobert_full"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()

id2label = model.config.id2label
# CLEAN TOKEN
def clean_token(token: str) -> str:
    token = token.replace("▁", "")
    token = token.replace("@@", "")
    return token.strip()


# ======================
# PREDICT NER
# ======================
def predict_ner(text: str):
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**encoding)

    predictions = torch.argmax(outputs.logits, dim=-1)[0]
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    labels = [id2label[p.item()] for p in predictions]

    results = []

    for tok, label in zip(tokens, labels):
        if tok in tokenizer.all_special_tokens:
            continue

        tok = clean_token(tok)
        if tok == "":
            continue

        results.append((tok, label))

    return results

# DEMO
if __name__ == "__main__":
    demo_text = """
    4.2. Nếu Bên B đơn phương chấm dứt hợp đồng mà không thực hiện nghĩa vụ báo trước tới Bên A thì Bên A sẽ không phải hoàn trả lại Bên B số tiền đặt cọc này.
    Nếu Bên A đơn phương chấm dứt hợp đồng mà không thực hiện nghĩa vụ báo trước tới bên B thì bên A sẽ phải hoàn trả lại Bên B số tiền đặt cọc và phải bồi thường thêm một khoản bằng chính tiền đặt cọc.
    """

    print("📄 Văn bản đầu vào:")
    print(demo_text.strip())

    # NER
    token_results = predict_ner(demo_text)

    print("\n Kết quả NER (token-level):")
    for token, label in token_results:
        print(f"{token:20s} → {label}")

    # GOM ENTITY
    entities = group_entities(token_results)

    # NORMALIZE ENTITY
    entities = normalize_entities(entities)

    print("\n THỰC THỂ TRÍCH XUẤT (ĐÃ CHUẨN HOÁ):")
    for k, v in entities.items():
        print(f"{k}: {v}")
    summary = summarize_clause(entities)

    print("\n TÓM TẮT ĐIỀU KHOẢN:")
    print(summary)