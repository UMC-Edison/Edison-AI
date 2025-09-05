import re, feedparser, trafilatura, time, argparse, sys

# ---------------------------
# 유틸 함수
# ---------------------------
def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def is_korean_dominant(s: str, ratio: float = 0.2) -> bool:
    """한글 비율이 ratio 이상일 때 True"""
    if not s:
        return False
    total = len(s)
    ko = len(re.findall(r"[가-힣]", s))
    return (ko / total) >= ratio and ko >= 5

# ---------------------------
# 본문 추출 + fallback
# ---------------------------
def extract_text(url: str) -> str | None:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    txt = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    return normalize(txt) if txt else None

def fetch_entry_text(e):
    # 시도 1: trafilatura
    txt = extract_text(e.link)
    if txt:
        return txt
    # 시도 2: summary + title
    raw = ""
    if hasattr(e, "title"): raw += e.title + ". "
    if hasattr(e, "summary"): raw += e.summary
    return normalize(raw) if raw else None

# ---------------------------
# 문단 분리
# ---------------------------
def split_paragraphs(doc: str, use_lang_filter=False):
    parts = re.split(r"\n{2,}|(?<=[.!?])\s+", doc)
    for p in parts:
        p = p.strip()
        if 15 <= len(p) <= 600:
            if not use_lang_filter or is_korean_dominant(p, 0.2):
                yield p

# ---------------------------
# 메인 함수
# ---------------------------
def main(feeds_file: str, out_path: str, use_lang_filter=False):
    feeds = [l.strip() for l in open(feeds_file, encoding="utf-8")
             if l.strip() and not l.startswith("#")]
    out = []
    for feed in feeds:
        parsed = feedparser.parse(feed)
        for e in parsed.entries[:200]:
            txt = fetch_entry_text(e)
            if txt:
                for p in split_paragraphs(txt, use_lang_filter=use_lang_filter):
                    out.append(p)
            time.sleep(0.3)  # 서버 예의상 딜레이
    # 중복 제거
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p); uniq.append(p)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(uniq))
    print(f"[rss] wrote {out_path} paragraphs={len(uniq)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feeds", default="scripts/rss_feeds.txt")
    ap.add_argument("--out", default="data/raw/blog_memos.txt")
    ap.add_argument("--langfilter", action="store_true", help="한글 비율 필터 적용 여부")
    args = ap.parse_args()
    main(args.feeds, args.out, use_lang_filter=args.langfilter)
