from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import re

def simple_tokenize(text):
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text.lower())
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]

df = pd.read_csv("data/data.csv") #학습용 문장

# TaggedDocument 생성
documents = [
    TaggedDocument(words=simple_tokenize(row["content"]), tags=[str(row["id"])])
    for _, row in df.iterrows()
]

model = Doc2Vec(vector_size=50, window=5, min_count=1, workers=4, epochs=40)
model.build_vocab(documents)
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

model.save("models/memo_doc2vec.model")
print("✅ Doc2Vec 모델 저장 완료!")