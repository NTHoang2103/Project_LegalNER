'''
   Phần huấn luyện mô hình: Phonert
   sinh viên thực hiện: Nguyễn Thanh Hoàng
'''

import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datasets import Dataset
from seqeval.metrics import classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)


# 1. CẤU HÌNH 

MODEL_NAME = "vinai/phobert-base"
LABEL_LIST = [
    "O",
    "B-PARTY", "I-PARTY",
    "B-OBLIGATION", "I-OBLIGATION",
    "B-AMOUNT", "I-AMOUNT",
    "B-DEADLINE", "I-DEADLINE",
    "B-PENALTY", "I-PENALTY",
]

label2id = {l: i for i, l in enumerate(LABEL_LIST)}
id2label = {i: l for l, i in label2id.items()}

# Thiết lập đường dẫn
BASE_DIR = Path(__file__).resolve().parent.parent
BIO_PATH = BASE_DIR / "data" / "labeled" / "train_1.bio"
MODEL_OUT = BASE_DIR / "models" / "ner_phobert_full"  # Đổi tên folder đầu ra cho bản Full
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
RESULTS_DIR = BASE_DIR / "outputs" / "results"

for d in [PLOTS_DIR, RESULTS_DIR, MODEL_OUT]:
    d.mkdir(parents=True, exist_ok=True)

# 2. TẢI Dữ LIỆU

def load_bio(path):
    sentences, labels = [], []
    tokens, tags = [], []

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file tại: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
            else:
                parts = line.split("\t")
                if len(parts) == 2:
                    tok, tag = parts
                    tokens.append(tok)
                    tags.append(label2id[tag])

    if tokens:
        sentences.append(tokens)
        labels.append(tags)

    return Dataset.from_dict({"tokens": sentences, "labels": labels})

print("Đang tải TOÀN BỘ dữ liệu từ file BIO để huấn luyện")
train_dataset = load_bio(BIO_PATH)

# 3. TOKENIZER & ALIGN 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

def tokenize_and_align_manual(examples):
    all_input_ids, all_labels, all_masks = [], [], []

    for i in range(len(examples["tokens"])):
        word_list = examples["tokens"][i]
        label_list = examples["labels"][i]

        input_ids = [tokenizer.bos_token_id]
        label_ids = [-100] 

        for word, label in zip(word_list, label_list):
            word_tokens = tokenizer.encode(word, add_special_tokens=False)
            if word_tokens:
                input_ids.extend(word_tokens)
                label_ids.append(label)
                label_ids.extend([-100] * (len(word_tokens) - 1))

        input_ids.append(tokenizer.eos_token_id)
        label_ids.append(-100)

        max_len = 128
        input_ids = input_ids[:max_len]
        label_ids = label_ids[:max_len]
        mask = [1] * len(input_ids)
        
        padding_len = max_len - len(input_ids)
        input_ids.extend([tokenizer.pad_token_id] * padding_len)
        label_ids.extend([-100] * padding_len)
        mask.extend([0] * padding_len)

        all_input_ids.append(input_ids)
        all_labels.append(label_ids)
        all_masks.append(mask)

    return {"input_ids": all_input_ids, "attention_mask": all_masks, "labels": all_labels}

print("Đang xử lý tiền dữ liệu")
train_dataset = train_dataset.map(tokenize_and_align_manual, batched=True)

# 4. MODEL

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_LIST),
    id2label=id2label,
    label2id=label2id
)

# 5. TRAINING ARGUMENTS 
training_args = TrainingArguments(
    output_dir=str(MODEL_OUT),
    eval_strategy="no",             
    logging_strategy="epoch",       
    save_strategy="no",          
    save_total_limit=2,          
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # Có thể tăng nếu RAM lớn
    num_train_epochs=40,            
    weight_decay=0.01,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer)
)

# 6. HUẤN LUYỆN 

print("\nBắt đầu huấn luyện PhoBERT trên 100% dữ liệu...")
trainer.train()

trainer.save_model(str(MODEL_OUT))
print(f"Model Full đã được lưu tại: {MODEL_OUT}")

# 7. VẼ BIỂU ĐỒ TRAIN LOSS

history = trainer.state.log_history
train_loss = [x["loss"] for x in history if "loss" in x]
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs, train_loss, 'b-o', label='Training Loss')
plt.title('Training Loss Curve (Full Dataset)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plot_path = PLOTS_DIR / "full_train_loss.png"
plt.savefig(plot_path)
print(f" Biểu đồ Loss đã được lưu tại: {plot_path}")
plt.show()

# 8. ĐÁNH GIÁ TRÊN CHÍNH TẬP TRAIN
print("\nĐang tạo báo cáo đánh giá cuối cùng trên toàn bộ dữ liệu")
predictions, labels, _ = trainer.predict(train_dataset)
predictions = np.argmax(predictions, axis=2)

true_predictions = [[id2label[p] for (p, l) in zip(pr, lab) if l != -100] for pr, lab in zip(predictions, labels)]
true_labels = [[id2label[l] for (p, l) in zip(pr, lab) if l != -100] for pr, lab in zip(predictions, labels)]

report = classification_report(true_labels, true_predictions)
print(report)

with open(RESULTS_DIR / "full_train_evaluation_report.txt", "w", encoding="utf-8") as f:
    f.write(report)


print("HOÀN TẤT HUẤN LUYỆN")
