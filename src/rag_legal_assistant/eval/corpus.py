import re
from collections import Counter
from rag_legal_assistant.ingestion.loader import load_all_documents

ARTICLE_RE = re.compile(r"(?:^|\n)[ \t]*(Art\.[ \t]*(\d+[a-z]*(?:\[\d+\])?))(?![\da-z])")


def extract_articles(directory: str = "data", min_chars: int = 250) -> list[dict]:
    articles = []
    for doc in load_all_documents(directory):
        text = doc["text"]
        matches = list(ARTICLE_RE.finditer(text))
        for i, match in enumerate(matches):
            start = match.start(1)
            end = matches[i + 1].start(1) if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if len(body) >= min_chars:
                articles.append({
                    "source": doc["source"],
                    "article": match.group(2),
                    "text": body,
                })
    return articles


if __name__ == "__main__":
    parsed = extract_articles("data")
    print(f"Artykułów (>= 250 znaków): {len(parsed)}\n")
    for source, count in sorted(Counter(a["source"] for a in parsed).items()):
        print(f"  {source}: {count}")
    for article in parsed[:3]:
        print(f"\n=== {article['source']} / Art. {article['article']} ===")
        print(article["text"][:300])