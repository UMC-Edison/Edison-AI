# build_setA_corpus.py
import os, re, json, glob, random, argparse
from typing import List

def normalize(s:str)->str:
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"(무단\s*전재|공유\s*금지|펌\s*금지).*$", "", s)

def keep_memo_length(s:str)->bool:
    return 15 <= len(s) <= 400 and not re.fullmatch(r"[^\w가-힣]+", s)

def uniq(seq:List[str])->List[str]:
    seen = set(); out = []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def load_nsmc(max_take:int=None)->list[str]:
    from Korpora import Korpora
    nsmc = Korpora.load("nsmc")
    texts = [normalize(t) for t in nsmc.get_all_texts() if keep_memo_length(normalize(t))]
    return texts[:max_take] if max_take else texts

def load_spoken_from_path(root:str, max_take:int=None)->list[str]:
    outs = []
    for path in glob.glob(os.path.join(root, "**/*"), recursive=True):
        if os.path.isdir(path): continue
        try:
            if path.endswith(".jsonl"):
                for line in open(path, encoding="utf-8", errors="ignore"):
                    rec = json.loads(line); outs.append(normalize(rec.get("utterance","")))
            elif path.endswith(".txt"):
                for ln in open(path, encoding="utf-8", errors="ignore"):
                    s = normalize(ln)
                    if keep_memo_length(s): outs.append(s)
        except: pass
        if max_take and len(outs) >= max_take: break
    return uniq([o for o in outs if keep_memo_length(o)])

def load_blog_memos(path:str, max_take:int=None)->list[str]:
    if not os.path.exists(path): return []
    memos = [ln.strip() for ln in open(path, encoding="utf-8") if keep_memo_length(ln.strip())]
    return uniq(memos[:max_take] if max_take else memos)

def main(args):
    random.seed(42)
    nsmc = load_nsmc(max_take=args.nsmc)
    spoken = load_spoken_from_path(args.spoken_dir, max_take=args.spoken)
    blogs = load_blog_memos(args.blog_file, max_take=args.blog)

    merged = uniq(nsmc + spoken + blogs)
    random.shuffle(merged)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(merged))
    print(f"[SetA] NSMC={len(nsmc)} Spoken={len(spoken)} Blog={len(blogs)} -> total={len(merged)} written to {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsmc", type=int, default=120000)
    ap.add_argument("--spoken", type=int, default=60000)
    ap.add_argument("--blog", type=int, default=20000)
    ap.add_argument("--spoken_dir", default="data/external/nikl_spoken")
    ap.add_argument("--blog_file", default="data/raw/blog_memos.txt")
    ap.add_argument("--out", default="data/raw/setA_memos.txt")
    args = ap.parse_args()
    main(args)
