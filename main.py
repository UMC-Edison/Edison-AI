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

MODEL_PATH = "models/memo_doc2vec_v3.model"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {MODEL_PATH}")
model = Doc2Vec.load(MODEL_PATH)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "edison-ai"}

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

class SpaceMapRequestDto(BaseModel):
    memos: List[Memo]

def simple_tokenize(text):
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text.lower())
    tokens = text.split()
    return [token for token in tokens if len(token) > 1]

# 버블 간격 띄우기
def repel_points(xy, min_dist=0.70, n_iter=50, step=0.01): 
    xy = xy.copy()
    for _ in range(n_iter):
        for i in range(len(xy)):
            for j in range(i+1, len(xy)):
                diff = xy[i] - xy[j]
                dist = np.linalg.norm(diff)
                if dist < min_dist and dist > 1e-5:
                    move = (min_dist - dist) * step * diff/dist
                    xy[i] += move
                    xy[j] -= move
    return xy


@app.post("/ai")
def vectorize(memos: List[Memo]):
    print("받은 요청 길이:", len(memos))

    vectors = []
    valid_memos = []
    for memo in memos:
        tokens = simple_tokenize(memo.content)
        if tokens:
            try:
                vec = model.infer_vector(tokens)
                vectors.append(vec)
                valid_memos.append(memo)
            except Exception as e:
                print("infer_vector 실패:", e)

    if not valid_memos:
        return []

    vector_array = np.array(vectors)

    if len(valid_memos) < 3:
        coords = np.zeros((len(valid_memos), 2))
    else:
        perplexity = min(30, len(valid_memos) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        coords = tsne.fit_transform(vector_array)

    scaler = MinMaxScaler(feature_range=(-5, 5))
    scaler_coords = scaler.fit_transform(coords)

    # 간격 띄우기
    scaler_coords = repel_points(scaler_coords, min_dist = 0.15, n_iter=50, step=0.01)

    result = [
        {"localIdx": m.localIdx, "x": float(x), "y": float(y)}
        for m, (x, y) in zip(valid_memos, scaler_coords)
    ]

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
