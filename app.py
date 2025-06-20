from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.manifold import TSNE
import numpy as np
import re

app = FastAPI()

class Memo(BaseModel):
    id: str
    content: str

class MemoList(BaseModel):
    memos: List[Memo]

# 토큰화 전처리 함수
def simple_tokenize(text):
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text.lower())
    tokens = text.split()
    return [token for token in tokens if len(token) > 1]

@app.post("/ai")
def vectorize(memo_list: MemoList):
    memos = memo_list.memos
    if len(memos) < 2:
        return [{"id": m.id, "x": 0.0, "y": 0.0} for m in memos]

    # TaggedDocument
    documents = [
        TaggedDocument(words=simple_tokenize(m.content), tags=[m.id])
        for m in memos
    ]

    # Doc2Vec 학습
    model = Doc2Vec(vector_size=50, window=5, min_count=2, workers=4, epochs=40)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

    vectors = np.array([model.dv[m.id] for m in memos])

    # TSNE: perplexity 조절
    perplexity = min(30, len(memos) - 1) if len(memos) > 1 else 1

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(vectors)

    # 결과 반환 (ID + 좌표)
    result = [{"id": m.id, "x": float(x), "y": float(y)} for m, (x, y) in zip(memos, coords)]
    return result
