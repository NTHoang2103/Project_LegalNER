from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent

TEXT_FILE = BASE_DIR / "data" / "text" / "contract_full.txt"
SAMPLE_DIR = BASE_DIR / "data" / "samples"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

with open(TEXT_FILE, "r", encoding="utf-8") as f:
    text = f.read()
# SPLIT ALL ARTICLES (Điều X.)
article_pattern = r"(Điều\s+(\d+)\.[\s\S]*?)(?=\nĐiều\s+\d+\.|\Z)"
articles = re.findall(article_pattern, text)

# Các điều cần xử lý
TARGET_ARTICLES = {"3", "5", "6", "9"}

created_files = 0

for article_text, article_num in articles:
    article_text = article_text.strip()

    if article_num not in TARGET_ARTICLES:
        continue

    clause_pattern = rf"({article_num}\.\d+[\.\s][\s\S]*?)(?=\n{article_num}\.\d+[\.\s]|\Z)"
    clauses = re.findall(clause_pattern, article_text)

    # NẾU CÓ KHOẢN CON → GHI THEO KHOẢN
    if clauses:
        for c in clauses:
            header = re.match(rf"({article_num}\.\d+)", c)
            if header:
                clause_id = header.group(1).replace(".", "_")
                out_file = SAMPLE_DIR / f"{clause_id}.txt"

                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(c.strip())

                created_files += 1
                print(f" Đã tạo: {out_file.name}")
    # NẾU KHÔNG CÓ KHOẢN → GHI NGUYÊN ĐIỀU
   
    else:
        out_file = SAMPLE_DIR / f"dieu_{article_num}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(article_text)
        created_files += 1
        print(f" Đã tạo: {out_file.name}")

print(f"\n HOÀN TẤT – Đã tạo {created_files} file TXT")
print(f" Thư mục: {SAMPLE_DIR}")
