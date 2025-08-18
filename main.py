from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os

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
class SimilarityResultItem(BaseModel):
    localIdx: str
    similarity: float

class SimilarityResponse(BaseModel):
    keyword: str
    results: List[SimilarityResultItem]

# 토큰화 전처리 함수
def simple_tokenize(text):
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text.lower())
    tokens = text.split()
    return [token for token in tokens if len(token) > 1]


@app.post("/ai", description= "스페이스 메인 화면 기본 매핑")
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

        # 벡터를 생성할 수 있는 메모가 하나도 없는 경우
    if not valid_memos:
        return []

    vector_array = np.array(vectors)

    # TSNE: perplexity 조절
    perplexity = min(30, len(valid_memos) - 1) if len(valid_memos) > 1 else 1
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(vector_array)

    # 결과 반환 (ID + 좌표)
    result = [{"localIdx": m.localIdx, "x": float(x), "y": float(y)} for m, (x, y) in zip(valid_memos, coords)]
    return result

@app.post("/similarity", response_model = SimilarityResponse)
def calculate_similarity(req:SimilarityRequest):

    keyword_tokens = simple_tokenize(req.keyword)

    # 키워드 토큰화 불가
    if not keyword_tokens:
            return SimilarityResponse(keyword=req.keyword, results=[])

    keyword_vec = model.infer_vector(keyword_tokens)

    scored = []
    for memo in req.memos:
        tokens = simple_tokenize(memo.content)
        if not tokens:
            continue

        memo_vec = model.infer_vector(tokens)
        sim = cosine_similarity([keyword_vec], [memo_vec])[0][0]
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os

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
class SimilarityResultItem(BaseModel):
    localIdx: str
    similarity: float

class SimilarityResponse(BaseModel):
    keyword: str
    results: List[SimilarityResultItem]

# 토큰화 전처리 함수
def simple_tokenize(text):
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text.lower())
    tokens = text.split()
    return [token for token in tokens if len(token) > 1]


@app.post("/ai", description= "스페이스 메인 화면 기본 매핑")
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

        # 벡터를 생성할 수 있는 메모가 하나도 없는 경우
    if not valid_memos:
        return []

    vector_array = np.array(vectors)

    # TSNE: perplexity 조절
    perplexity = min(30, len(valid_memos) - 1) if len(valid_memos) > 1 else 1
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(vector_array)

    # 결과 반환 (ID + 좌표)
    result = [{"localIdx": m.localIdx, "x": float(x), "y": float(y)} for m, (x, y) in zip(valid_memos, coords)]
    return result

@app.post("/similarity", response_model = SimilarityResponse)
def calculate_similarity(req:SimilarityRequest):

    keyword_tokens = simple_tokenize(req.keyword)

    # 키워드 토큰화 불가
    if not keyword_tokens:
            return SimilarityResponse(keyword=req.keyword, results=[])

    keyword_vec = model.infer_vector(keyword_tokens)

    scored = []
    for memo in req.memos:
        tokens = simple_tokenize(memo.content)
        if not tokens:
            continue

        memo_vec = model.infer_vector(tokens)
        sim = cosine_similarity([keyword_vec], [memo_vec])[0][0]

        if sim >= 0.5:
            scored.append(SimilarityResultItem(localIdx=memo.localIdx, similarity=sim))

    sorted_results = sorted(scored, key=lambda x: x.similarity, reverse=True)
    top_10_results = sorted_results[:10]

    return SimilarityResponse(keyword=req.keyword, results=top_10_results)


        if sim >= 0.5:
            scored.append(SimilarityResultItem(localIdx=memo.localIdx, similarity=sim))

    sorted_results = sorted(scored, key=lambda x: x.similarity, reverse=True)
    top_10_results = sorted_results[:10]

    return SimilarityResponse(keyword=req.keyword, results=top_10_results)

