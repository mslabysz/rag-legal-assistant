import re
from collections import Counter
from rag_legal_assistant.ingestion.loader import load_all_documents

ARTICLE_RE = re.compile(r"(?:^|\n)[ \t]*(Art\.[ \t]*(\d+[a-z]*(?:\[\d+\])?))(?![\da-z])")


def split_articles(text: str, min_chars: int = 250) -> list[dict]:
    articles = []
    matches = list(ARTICLE_RE.finditer(text))
    for i, match in enumerate(matches):
        start = match.start(1)
        end = matches[i + 1].start(1) if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if len(body) >= min_chars:
            articles.append({"article": match.group(2), "text": body})
    return articles

def extract_articles(directory: str = "data", min_chars: int = 250) -> list[dict]:
    return [
        {"source": doc["source"], **article}
        for doc in load_all_documents(directory)
        for article in split_articles(doc["text"], min_chars)
    ]


if __name__ == "__main__":
    parsed = extract_articles("data")
    print(f"Articles (>= 250 chars): {len(parsed)}\n")
    for source, count in sorted(Counter(a["source"] for a in parsed).items()):
        print(f"  {source}: {count}")
    for article in parsed[:3]:
        print(f"\n=== {article['source']} / Art. {article['article']} ===")
        print(article["text"][:300])