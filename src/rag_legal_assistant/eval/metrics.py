import re

def hit_rank(docs: list[dict], expected_source: str, expected_article: str) -> int | None:
    pattern = re.compile(rf"(?:^|\n)\s*Art\.\s*{re.escape(expected_article)}(?![\da-z])")
    for position, doc in enumerate(docs, start=1):
        if doc["source"] == expected_source and pattern.search(doc["text"]):
            return position
    return None