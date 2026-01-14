from docx import Document
from pathlib import Path

# Xác định thư mục gốc của project
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
TEXT_DIR = BASE_DIR / "data" / "text"

# Tạo thư mục nếu chưa tồn tại
TEXT_DIR.mkdir(parents=True, exist_ok=True)

# Đọc file Word
doc = Document(RAW_DIR / "hop-dong-thue-nha-o.docx")

paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

# Ghi ra file text
with open(TEXT_DIR / "contract_full.txt", "w", encoding="utf-8") as f:
    for p in paragraphs:
        f.write(p + "\n")

print(" Đã trích xuất văn bản từ file Word")
