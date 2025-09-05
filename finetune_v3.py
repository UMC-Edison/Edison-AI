from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import re
from kiwipiepy import Kiwi

# -----------------------------
# 토큰화 함수
# -----------------------------
def simple_tokenize(text):
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", str(text).lower())
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]

kiwi = Kiwi()

def tokenize_ko(s: str):
    return [
        t.form for t in kiwi.tokenize(s, normalize_coda=True)
        if t.tag.startswith(("N", "V", "VA", "XR", "SL")) and len(t.form) > 1
    ]

# -----------------------------
# 1. data.csv 불러오기
# -----------------------------
df = pd.read_csv("data/data.csv")  # 학습용 문장
documents = [
    TaggedDocument(words=simple_tokenize(row["content"]), tags=[f"csv_{row['id']}"])
    for _, row in df.iterrows()
]

print(f"✅ data.csv 문서 수: {len(documents)}")

# -----------------------------
# 2. merged_sample.txt 불러오기
# -----------------------------
path = "data/raw/merged_sample.txt"
new_docs = [line.strip() for line in open(path, encoding="utf-8") if line.strip()]
tagged_new = [
    TaggedDocument(words=tokenize_ko(s), tags=[f"sample_{i}"])
    for i, s in enumerate(new_docs)
]

print(f"✅ merged_sample.txt 문서 수: {len(tagged_new)}")

# -----------------------------
# 3. 전체 데이터 합치기
# -----------------------------
all_docs = documents + tagged_new
print(f"총 학습 문서 수: {len(all_docs)}")

# -----------------------------
# 4. Doc2Vec 모델 학습
# -----------------------------
model = Doc2Vec(
    vector_size=50,   # 임베딩 차원 수
    window=5,
    min_count=1,
    workers=4,
    epochs=40
)

model.build_vocab(all_docs)
model.train(all_docs, total_examples=len(all_docs), epochs=model.epochs)

# -----------------------------
# 5. 모델 저장
# -----------------------------
model.save("models/memo_doc2vec_v3.model")
print("🎉 모델 학습 완료! → models/memo_doc2vec_v3.model")