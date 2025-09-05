import os, json, glob, re

SRC_DIR = "data/external/aihub_dialog"
OUT_PATH = "data/raw/aihub_dialog_memos.txt"

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def extract_text_from_record(rec):
    outs = []
    if isinstance(rec, dict):
        for k in ["utterance", "form", "text", "content", "sentence"]:
            if k in rec and isinstance(rec[k], str):
                s = normalize(rec[k])
                if 50 <= len(s) <= 1000:
                    outs.append(s)
        # nested 탐색
        for v in rec.values():
            if isinstance(v, (list, tuple)):
                for it in v:
                    outs.extend(extract_text_from_record(it))
            elif isinstance(v, dict):
                outs.extend(extract_text_from_record(v))
    return outs

def main():
    out = []
    for path in glob.glob(os.path.join(SRC_DIR, "**/*"), recursive=True):
        if os.path.isdir(path):
            continue
        if not path.endswith((".json", ".jsonl")):
            continue
        try:
            if path.endswith(".jsonl"):
                for line in open(path, encoding="utf-8"):
                    if not line.strip(): continue
                    rec = json.loads(line)
                    out.extend(extract_text_from_record(rec))
            else:  # .json
                data = json.load(open(path, encoding="utf-8"))
                if isinstance(data, list):
                    for rec in data:
                        out.extend(extract_text_from_record(rec))
                elif isinstance(data, dict):
                    out.extend(extract_text_from_record(data))
        except Exception as e:
            print("ERR", path, e)
    # 중복 제거
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s); uniq.append(s)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(uniq))
    print(f"[done] saved {len(uniq)} utterances -> {OUT_PATH}")

if __name__ == "__main__":
    main()
