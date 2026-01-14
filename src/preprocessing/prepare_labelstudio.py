import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SAMPLE_DIR = BASE_DIR / "data" / "samples"
LABELED_DIR = BASE_DIR / "data" / "labeled"
LABELED_DIR.mkdir(parents=True, exist_ok=True)

output_file = LABELED_DIR / "labelstudio_import_1.json"

tasks = []

for txt_file in SAMPLE_DIR.glob("*.txt"):
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    tasks.append({
        "data": {
            "text": text
        }
    })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(tasks, f, ensure_ascii=False, indent=2)

print(f"Đã tạo file import cho Label Studio: {output_file}")