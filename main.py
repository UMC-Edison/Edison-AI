from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.manifold import TSNE
import numpy as np
import re

app = FastAPI()

# 모델 로드
MODEL_PATH = "models/memo_doc2vec.model"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"모델이 존재하지 않음: {MODEL_PATH}")
model = Doc2Vec.load(MODEL_PATH)

# 요청
class Memo(BaseModel):
    localIdx: str
    content: str

class SimilarityRequest(BaseModel):
    keyword: str
    memos: List[Memo]

# 응답
class SimilarityResponse(BaseModel):
    keyword: str
    top_ids: List[str]

# 토큰화 전처리 함수
def simple_tokenize(text):
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text.lower())
    tokens = text.split()
    return [token for token in tokens if len(token) > 1]

@app.post("/ai")
def vectorize(memo_list: List[Memo]):

    memos = memo_list

    # TaggedDocument
    documents = [
        TaggedDocument(words=simple_tokenize(m.content), tags=[m.localIdx])
        for m in memos
    ]

    vectors = np.array([model.dv[m.localIdx] for m in memos])

    # TSNE: perplexity 조절
    perplexity = min(30, len(memos) - 1) if len(memos) > 1 else 1
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(vectors)

    # 결과 반환 (ID + 좌표)
    result = [{"id": m.localIdx, "x": float(x), "y": float(y)} for m, (x, y) in zip(memos, coords)]
    return result

@app.post("/similarity", response_model = SimilarityResponse)
def calculate_similarity(req:SimilarityRequest):
    keywords_token = simple_tokenize(req.keyword)

    # 키워드 토큰화 불가
    if not keyword_tokens:
            return {"keyword": req.keyword, "top_ids": []}

    keyword_vec = model.infer_vector(keyword_tokens)
    scored = []

    for memo in req.memos:
        tokens = simple_tokenize(memo.content)
        if not tokens:
            continue
        memo_vec = model.infer_vector(tokens)
        sim = cosine_similarity([keyword_vec], [memo_vec])[0][0]

        if sim >= 0.5:
            scored.append((memo.localIdx, sim)

    top_ids = [id for id, _ in sorted(scored, key= lambda x: x[1], reverse = True)[:10]]

    return {
        "keyword": req.keyword,
        "top_ids": top_ids
    }

