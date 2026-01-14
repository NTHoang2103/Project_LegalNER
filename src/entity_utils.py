def clean_token(token: str) -> str:
    token = token.replace("▁", "")
    token = token.replace("@@", "")
    return token.strip()


def group_entities(tokens_with_labels):
    entities = {
        "PARTY": [],
        "OBLIGATION": [],
        "AMOUNT": [],
        "DEADLINE": [],
        "PENALTY": []
    }

    current_tokens = []
    current_label = None

    for token, label in tokens_with_labels:
        token = clean_token(token)

        if label == "O" or token == "":
            if current_tokens:
                entities[current_label].append(" ".join(current_tokens))
                current_tokens = []
                current_label = None
            continue

        if "-" not in label:
            continue

        prefix, ent_type = label.split("-", 1)

        if prefix == "B":
            if current_tokens:
                entities[current_label].append(" ".join(current_tokens))
            current_tokens = [token]
            current_label = ent_type

        elif prefix == "I" and current_label == ent_type:
            current_tokens.append(token)

        else:
            if current_tokens:
                entities[current_label].append(" ".join(current_tokens))
            current_tokens = []
            current_label = None

    if current_tokens:
        entities[current_label].append(" ".join(current_tokens))

    return entities



# NORMALIZE ENTITY 
def normalize_entities(entities: dict):
    normalized = {}

    for ent_type, values in entities.items():
        cleaned = []
        for v in values:
            v = v.strip()
            v = v.replace("bên", "Bên")   # chuẩn hoá chữ hoa
            if v not in cleaned:
                cleaned.append(v)
        normalized[ent_type] = cleaned

    return normalized
