from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from kiwipiepy import Kiwi

kiwi = Kiwi()

def tokenize_ko(s: str):
    return [t.form for t in kiwi.tokenize(s, normalize_coda=True)
            if t.tag.startswith(("N","V","VA","XR","SL")) and len(t.form) > 1]

def main():
    # 1. 기존 모델 로드
    model = Doc2Vec.load("models/memo_doc2vec.model")

    # 2. 새 데이터 로드
    path = "data/raw/merged.txt"
    new_docs = [line.strip() for line in open(path, encoding="utf-8") if line.strip()]
    tagged_new = [TaggedDocument(words=tokenize_ko(s), tags=[f"new_{i}"])
                  for i, s in enumerate(new_docs)]

    print(f"[finetune] new docs: {len(tagged_new)}")

    # 3. 어휘 업데이트
    model.build_vocab(tagged_new, update=True)

    # 4. 추가 학습
    model.train(tagged_new, total_examples=len(tagged_new), epochs=20)

    # 5. 저장
    model.save("models/memo_doc2vec_finetuned.model")
    print("[finetune] saved -> models/memo_doc2vec_finetuned.model")

if __name__ == "__main__":
    main()
