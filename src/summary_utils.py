'''
Ở phần hàm sẽ chịu trách nhiệm xử lý, logic và suy luận các điều khoản
nhưng ở đây hiệu quả tốt nếu suy luận điều khoản 4.2; còn các điều khoản
vẫn áp dụng được nhưng chưa thực sự ổn định do còn thiếu các logic suy luận

NER → Entity normalization → Rule-based legal reasoning → Summary text
'''
def summarize_clause(entities: dict) -> str:
    obligations = entities.get("OBLIGATION", [])
    penalties = entities.get("PENALTY", [])
    amounts = entities.get("AMOUNT", [])

    sentences = []

    # 1. Nghĩa vụ
    if obligations:
        obligation = obligations[0].strip()
        sentences.append(
            f"Điều khoản quy định {obligation} khi đơn phương chấm dứt hợp đồng."
        )

    # 2. Suy luận chế tài
    ben_a_penalties = set()
    ben_b_penalties = set()

    for p in penalties:
        p_lower = p.lower()

        if "không phải hoàn trả" in p_lower:
            ben_b_penalties.add("mất tiền đặt cọc")

        if "phải hoàn trả" in p_lower:
            ben_a_penalties.add("hoàn trả tiền đặt cọc")

        if "bồi thường" in p_lower:
            ben_a_penalties.add("bồi thường")

    if ben_b_penalties:
        sentences.append(
            "Trường hợp Bên B vi phạm nghĩa vụ báo trước, Bên B "
            + " và ".join(ben_b_penalties) + "."
        )

    if ben_a_penalties:
        sentences.append(
            "Trường hợp Bên A vi phạm nghĩa vụ báo trước, Bên A có trách nhiệm "
            + " và ".join(ben_a_penalties) + "."
        )

    if amounts:
        sentences.append(
            f"Các chế tài nêu trên liên quan đến {amounts[0]}."
        )

    return " ".join(sentences)
