from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from gensim.models.doc2vec import Doc2Vec
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os

app = FastAPI()

MODEL_PATH = "models/memo_doc2vec_finetuned.model"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {MODEL_PATH}")
model = Doc2Vec.load(MODEL_PATH)

class Memo(BaseModel):
    localIdx: str
    content: str

class SimilarityRequest(BaseModel):
    keyword: str
    memos: List[Memo]

class SimilarityResultItem(BaseModel):
    localIdx: str
    similarity: float

class SimilarityResponse(BaseModel):
    keyword: str
    results: List[SimilarityResultItem]

def simple_tokenize(text):
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text.lower())
    tokens = text.split()
    return [token for token in tokens if len(token) > 1]


@app.post("/ai")
def vectorize(memos: List[Memo]):
    if not memos:
        return []

    vectors = []
    valid_memos = []
    for memo in memos:
        tokens = simple_tokenize(memo.content)
        if tokens:
            vectors.append(model.infer_vector(tokens))
            valid_memos.append(memo)

    if not valid_memos:
        return []

    vector_array = np.array(vectors)

    perplexity = min(30, len(valid_memos) - 1) if len(valid_memos) > 1 else 1
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(vector_array)

    scaler = MinMaxScaler(feature_range=(-5, 5))
    scaler_coords = scaler.fit_transform(coords)

    result = [{"localIdx": m.localIdx, "x": float(x), "y": float(y)} for m, (x, y) in zip(valid_memos, scaler_coords)]

    return result

@app.post("/similarity")
def calculate_similarity(req: SimilarityRequest):
    keyword_tokens = simple_tokenize(req.keyword)

    if not keyword_tokens:
        return SimilarityResponse(keyword=req.keyword, results=[])

    keyword_vec = model.infer_vector(keyword_tokens)

    scored = []
    for memo in req.memos:
        tokens = simple_tokenize(memo.content)
        if not tokens:
            continue

        memo_vec = model.infer_vector(tokens)
        sim = float(cosine_similarity([keyword_vec], [memo_vec])[0][0])
        print(f"키워드: {req.keyword}, 메모: {memo.content[:30]}..., 유사도: {sim}")

        if sim >= 0.0:
            scored.append(SimilarityResultItem(localIdx=memo.localIdx, similarity=sim))

    sorted_results = sorted(scored, key=lambda x: x.similarity, reverse=True)

    top_10_results = sorted_results[:10]

    return SimilarityResponse(keyword=req.keyword, results=top_10_results)
