# Trích xuất Thông tin Chuyên sâu từ Văn bản Pháp luật tiếng Việt - Nhóm Cá Nhân

**Đề tài**: Áp dụng mô hình PhoBERT kết hợp kỹ thuật Few-shot Learning để trích xuất thông tin (Named Entity Recognition & Information Extraction) từ văn bản pháp luật.

## Mục tiêu dự án

Xây dựng một pipeline hoàn chỉnh để:
- Tiền xử lý văn bản pháp luật (.docx)
- Tách điều khoản & chuẩn bị dữ liệu gán nhãn
- Huấn luyện mô hình PhoBERT cho bài toán trích xuất thông tin
- Đánh giá hiệu suất mô hình
- Triển khai demo trích xuất thông tin và tóm tắt nội dung

## Công nghệ chính

- **Mô hình ngôn ngữ**: PhoBERT (vinai/phobert-base hoặc phobert-large)
- **Kỹ thuật học**: Fine-tuning + Few-shot Learning
- **Bài toán chính**: Named Entity Recognition (NER) trên văn bản pháp luật
- **Công cụ hỗ trợ**: Label Studio, pandas, transformers, seqeval, ...

## Phần 2: Hướng dẫn chạy project

Tiếp theo để ý trước ổ cứng đã có chữ (venv)

Sau đó để chạy các file trong project:

**2.1** Tiền xử lý dữ liệu

-2.1.1 Chuyển dữ liệu thô thành 1 file docx sạch không còn kí tự trống và thùa
python src/preprocessing/read_docx.py

-2.1.2 Sau chuyển thành 1 file docx sạch thì bắt đầu lọc, chọn các điều khoản và tách thành các điều khoản nhỏ riêng biệt
python src/preprocessing/split_clauses.py

-2.1.3 Sau khi tách thành các điều khoản nhỏ riêng biệt thì chuyển thành file json và đưa vào lable studio để gán nhãn thủ công
python src/preprocessing/prepapre_labelstudio.py

-2.1.4 Chuyển đổi file json thành file bio
python src/preprocessing/json_to_bio.py

riêng file __init__.py là file trống không cần chạy vì để chủ yếu gọi model

**3.** Phân tích dữ liệu 
python src/eda.py

**4.** Huấn luyện mô hình 
python src/model_phobert_nguyenthanhhoang.py

**5.** Đánh giá của model 
python src/evaluation.py

**6.** Demo trích xuất thông tin và tóm tắt 
python src/demo_tomtat_trichxuat.py

## Phần 3 Kết quả

kết quả sẽ được lưu trong thư mục outputs

- `plots/:` chứa các biểu đồ và trực quan
- `results/:` chứa đánh giá các thông số
