import numpy as np
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, DataCollatorForTokenClassification
from datasets import Dataset

def convert_to_builtin_type(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def evaluate_and_save(model_path, bio_path):
    print(f"Khởi động đánh giá mô hình tại: {model_path}")
    
    # 1. SETUP
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    id2label = model.config.id2label
    label2id = {v: k for k, v in id2label.items()}
    
    # Xác định thư mục đầu ra
    base_dir = Path(model_path).parent.parent
    output_dir_json = base_dir / "outputs" / "results"
    output_dir_plots = base_dir / "outputs" / "plots"
    output_dir_json.mkdir(parents=True, exist_ok=True)
    output_dir_plots.mkdir(parents=True, exist_ok=True)

    #2. ĐỌC DỮ LIỆU TỪ FILE .BIO 
    sentences, labels = [], []
    tokens, tags = [], []
    with open(bio_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens); labels.append(tags)
                    tokens, tags = [], []
            else:
                parts = line.split("\t")
                if len(parts) == 2:
                    tokens.append(parts[0])
                    tags.append(label2id.get(parts[1], 0))

    # 3. ALIGNMENT 
    def align_labels(token_list, label_list):
        input_ids = [tokenizer.bos_token_id]
        label_ids = [-100]
        for word, label in zip(token_list, label_list):
            word_tokens = tokenizer.encode(word, add_special_tokens=False)
            if word_tokens:
                input_ids.extend(word_tokens)
                label_ids.append(label)
                label_ids.extend([-100] * (len(word_tokens) - 1))
        input_ids.append(tokenizer.eos_token_id)
        label_ids.append(-100)
        return input_ids, label_ids

    processed = [align_labels(s, l) for s, l in zip(sentences, labels)]
    eval_dataset = Dataset.from_dict({
        "input_ids": [x[0] for x in processed],
        "labels": [x[1] for x in processed],
        "attention_mask": [[1]*len(x[0]) for x in processed]
    })

    #4. DỰ ĐOÁN
    trainer = Trainer(
        model=model, 
        processing_class=tokenizer, 
        data_collator=DataCollatorForTokenClassification(tokenizer)
    )
    
    output = trainer.predict(eval_dataset)
    predictions = np.argmax(output.predictions, axis=2)

    # Lọc bỏ các sub-tokens (-100) để tính toán chính xác
    true_pred = [[id2label[p] for (p, l) in zip(pr, lb) if l != -100] 
                 for pr, lb in zip(predictions, output.label_ids)]
    true_lab = [[id2label[l] for (p, l) in zip(pr, lb) if l != -100] 
                for pr, lb in zip(predictions, output.label_ids)]

    # 5. TÍNH TOÁN CHỈ SỐ  
    p = precision_score(true_lab, true_pred)
    r = recall_score(true_lab, true_pred)
    f1 = f1_score(true_lab, true_pred)
    # Xuất dạng dictionary để lưu vào JSON
    report_dict = classification_report(true_lab, true_pred, output_dict=True)

    #  6. LƯU KẾT QUẢ 
    results_json = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {
            "name": "PhoBERT-NER-Legal",
            "task": "Extract Deep Information for Summarization"
        },
        "overall_metrics": {
            "precision": float(p),
            "recall": float(r),
            "f1_score": float(f1)
        },
        "detailed_per_entity": report_dict
    }
    
    json_path = output_dir_json / "ner_evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=4, ensure_ascii=False, default=convert_to_builtin_type)

    # 7. VẼ BIỂU ĐỒ 
    # Chuyển đổi report_dict thành DataFrame để vẽ biểu đồ
    report_df = pd.DataFrame(report_dict).transpose().iloc[:-3] 
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    # Vẽ cột F1-score cho từng thực thể
    ax = report_df['f1-score'].plot(kind='bar', color='forestgreen', alpha=0.8)
    plt.title("Chỉ số F1-Score theo từng loại thực thể Pháp luật", fontsize=14)
    plt.ylabel("Điểm F1")
    plt.xlabel("Thực thể")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    
    # Gắn số lên đầu mỗi cột
    for p_bar in ax.patches:
        ax.annotate(f'{p_bar.get_height():.2f}', (p_bar.get_x() + 0.1, p_bar.get_height() + 0.02))

    plot_path = output_dir_plots / "ner_f1_scores.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    print("-" * 50)
    print(f"Đã lưu kết quả thành công!")
    print(f" JSON: {json_path}")
    print(f" Hình ảnh: {plot_path}")
    print("-" * 50)
    print(classification_report(true_lab, true_pred)) # Hiển thị lại bảng lên console

if __name__ == "__main__":
    MODEL_DIR = "D:/Project_LegalNER/models/ner_phobert_full"
    DATA_BIO = "D:/Project_LegalNER/data/labeled/train_1.bio"
    
    evaluate_and_save(MODEL_DIR, DATA_BIO)