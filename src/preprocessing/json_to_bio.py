import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
JSON_PATH = BASE_DIR / "data" / "labeled" / "labelstudio_export_1.json"
BIO_PATH = BASE_DIR / "data" / "labeled" / "train_1.bio"

# 5 NHÃN CHÍNH THỨC
VALID_LABELS = {"PARTY", "OBLIGATION", "AMOUNT", "DEADLINE", "PENALTY"}

def convert_json_to_bio(json_path, bio_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(bio_path, "w", encoding="utf-8") as out:
        for item in data:
            text = item["data"]["text"]

            # Lấy annotations 
            entities = []
            if item.get("annotations"):
                for ann in item["annotations"][0]["result"]:
                    label = ann["value"]["labels"][0]
                    if label in VALID_LABELS:
                        entities.append({
                            "start": ann["value"]["start"],
                            "end": ann["value"]["end"],
                            "label": label
                        })

            # Tokenize đơn giản
            tokens = text.split()
            offset = 0

            for tok in tokens:
                start = text.find(tok, offset)
                end = start + len(tok)
                offset = end

                tag = "O"
                for ent in entities:
                    if start == ent["start"]:
                        tag = "B-" + ent["label"]
                    elif start > ent["start"] and end <= ent["end"]:
                        tag = "I-" + ent["label"]

                out.write(f"{tok}\t{tag}\n")

            out.write("\n")  

    print(f" Đã tạo file BIO tại: {bio_path}")

if __name__ == "__main__":
    convert_json_to_bio(JSON_PATH, BIO_PATH)
