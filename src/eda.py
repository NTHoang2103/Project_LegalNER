# EDA FOR LEGAL NER & SUMMARIZATION (PHOBERT)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

def analyze_bio_data(file_path):
    """Phân tích chi tiết tệp dữ liệu .bio cho NER pháp luật"""
    sentences = []
    all_labels = []
    sentence_lengths = []
    
    current_sentence_labels = []
    current_sentence_len = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence_len > 0:
                    sentences.append(current_sentence_labels)
                    sentence_lengths.append(current_sentence_len)
                    current_sentence_labels = []
                    current_sentence_len = 0
                continue
            
            parts = line.split("\t")
            if len(parts) == 2:
                token, label = parts
                # Chỉ lấy nhãn thực thể (bỏ tiền tố B-, I- để đếm loại thực thể)
                clean_label = label.replace("B-", "").replace("I-", "")
                all_labels.append(clean_label)
                current_sentence_labels.append(clean_label)
                current_sentence_len += 1

    if current_sentence_len > 0:
        sentences.append(current_sentence_labels)
        sentence_lengths.append(current_sentence_len)

    return all_labels, sentence_lengths, sentences

def plot_eda_results(all_labels, sentence_lengths, save_dir):
    """Vẽ biểu đồ phân tích dữ liệu"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Thiết lập font và style
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans' 

    # vẽ biểu đồ 1 
    plt.figure(figsize=(12, 6))
    label_counts = Counter([l for l in all_labels if l != "O"])
    label_df = pd.DataFrame(label_counts.items(), columns=['Entity', 'Count']).sort_values('Count', ascending=False)
    
    sns.barplot(data=label_df, x='Count', y='Entity', hue='Entity', palette='viridis', legend=False)
    plt.title("Phân bổ số lượng các thực thể pháp luật (Trừ nhãn 'O')", fontsize=14)
    plt.xlabel("Số lần xuất hiện")
    plt.ylabel("Loại thực thể")
    plt.savefig(save_dir / "entity_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Vẽ biểu đồ 2  
    plt.figure(figsize=(10, 6))
    sns.histplot(sentence_lengths, bins=20, kde=True, color='skyblue')
    plt.axvline(np.mean(sentence_lengths), color='red', linestyle='--', label=f'Trung bình: {np.mean(sentence_lengths):.1f}')
    plt.title("Phân bổ độ dài câu (Số lượng Tokens)", fontsize=14)
    plt.xlabel("Độ dài câu")
    plt.ylabel("Số lượng câu")
    plt.legend()
    plt.savefig(save_dir / "sentence_length_dist.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("\n" + "="*30)
    print("THỐNG KÊ CHI TIẾT DỮ LIỆU")
    print("="*30)
    print(f"Tổng số câu: {len(sentence_lengths)}")
    print(f"Tổng số tokens: {sum(sentence_lengths)}")
    print(f"Độ dài câu trung bình: {np.mean(sentence_lengths):.2f}")
    print(f"Độ dài câu lớn nhất: {max(sentence_lengths)}")
    print("\nSố lượng từng loại thực thể:")
    for entity, count in label_counts.items():
        print(f"- {entity}: {count}")

if __name__ == "__main__":
    # Cấu hình đường dẫn
    BASE_DIR = Path(__file__).resolve().parent.parent
    BIO_FILE = BASE_DIR / "data" / "labeled" / "train_1.bio"
    OUTPUT_PLOTS = BASE_DIR / "outputs" / "plots"

    print(f"Đang phân tích dữ liệu tại: {BIO_FILE}")
    
    try:
        labels, lengths, sents = analyze_bio_data(BIO_FILE)
        plot_eda_results(labels, lengths, OUTPUT_PLOTS)
        print(f"\nHoàn thành! Biểu đồ đã được lưu tại: {OUTPUT_PLOTS}")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {BIO_FILE}")