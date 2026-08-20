import asyncio
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from rag_legal_assistant.eval.corpus import extract_articles
from rag_legal_assistant.llm import llm

N_QUESTIONS = 100
SEED = 42
OUT = Path("src/rag_legal_assistant/eval/retrieval_eval_set.json")

BOILERPLATE_RE = re.compile(
    r"wchodzi w życie|traci moc|Ustawa nie narusza|Ustawa określa zasady"
    r"|o zmianie ustawy|W ustawie z dnia",
    re.IGNORECASE,
)
MAX_CITATIONS = 2

QUESTION_PROMPT = ChatPromptTemplate.from_template(
    """You are building an evaluation set for a Polish legal search engine.
    Below is the full text of a single article of Polish law.

    Write ONE question in Polish that this article answers.

    Rules:
    - the question MUST be answerable using only this article,
    - phrase it the way a non-lawyer would ask it, in natural language,
    - do NOT mention the article number or the name of the act,
    - do NOT copy phrases from the article - paraphrase the meaning,
    - return only the question itself, nothing else.

    Article:
    {article}

    Question (in Polish):"""
)

VALIDATION_PROMPT = ChatPromptTemplate.from_template(
    """You are validating an evaluation set for a Polish legal search engine.

    Article:
    {article}

    Question:
    {question}

    Can this question be answered COMPLETELY and UNAMBIGUOUSLY using ONLY the article above?
    Answer "no" if answering it would require a different provision, or if the question is so
    general that many other articles would answer it equally well.

    Reply with one word: yes or no."""
)

question_chain = QUESTION_PROMPT | llm
validation_chain = VALIDATION_PROMPT | llm


def is_substantive(article: dict) -> bool:
    text = article["text"]
    if BOILERPLATE_RE.search(text[:400]):
        return False
    if text.count("Dz. U.") > MAX_CITATIONS:
        return False
    return True


def select_articles() -> list[dict]:
    articles = extract_articles("data")
    print(f"Parsed articles: {len(articles)}")

    counts = Counter((a["source"], a["article"]) for a in articles)
    eligible = [
        a for a in articles
        if counts[(a["source"], a["article"])] == 1 and is_substantive(a)
    ]
    print(f"After dropping duplicates and boilerplate provisions: {len(eligible)}\n")

    by_source = defaultdict(list)
    for article in eligible:
        by_source[article["source"]].append(article)

    random.seed(SEED)
    per_source = max(1, N_QUESTIONS // len(by_source))
    sample = []
    for source in sorted(by_source):
        pool = by_source[source]
        taken = random.sample(pool, min(per_source, len(pool)))
        sample.extend(taken)
        print(f"  {source}: {len(pool)} candidates, picked {len(taken)}")
    return sample


async def main():
    sample = select_articles()

    print(f"\nGenerating {len(sample)} questions...")
    generated = await question_chain.abatch(
        [{"article": a["text"][:4000]} for a in sample],
        config={"max_concurrency": 10},
    )
    questions = [response.content.strip() for response in generated]

    print("Validating questions against their articles...")
    verdicts = await validation_chain.abatch(
        [
            {"article": article["text"][:4000], "question": question}
            for article, question in zip(sample, questions)
        ],
        config={"max_concurrency": 10},
    )

    dataset = []
    rejected = []
    for article, question, verdict in zip(sample, questions, verdicts):
        if verdict.content.strip().lower().startswith("yes"):
            dataset.append({
                "question": question,
                "source": article["source"],
                "article": article["article"],
            })
        else:
            rejected.append(f"{article['source']} / Art. {article['article']}: {question}")

    if rejected:
        print(f"\nRejected by validation ({len(rejected)}):")
        for row in rejected:
            print(f"  - {row}")

    OUT.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(dataset)} questions to {OUT}")


if __name__ == "__main__":
    asyncio.run(main())