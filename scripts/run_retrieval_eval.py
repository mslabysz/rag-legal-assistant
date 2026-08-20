import argparse
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from rag_legal_assistant.config import settings
from rag_legal_assistant.llm import llm
from rag_legal_assistant.prompts import MULTI_QUERY_PROMPT
from rag_legal_assistant.vectordb.client import vector_store

EVAL_SET = Path("src/rag_legal_assistant/eval/retrieval_eval_set.json")
BENCHMARKS = Path("benchmarks")
CACHE = BENCHMARKS / "candidates.json"
OUT = BENCHMARKS / "retrieval.md"

K = 5
INCLUDE_ORIGINAL = True

STRATEGIES = {
    "dense k=20": {"k": 20, "use_multi_query": False},
    "dense k=44": {"k": 44, "use_multi_query": False},
    "multi-query k=20x4": {"k": 20, "use_multi_query": True},
}

_MODELS: dict[str, object] = {}

# Dopisywane pod tabelami. Spot-check jest wynikiem ręcznego przeglądu i trzeba go
# powtórzyć po każdej regeneracji zbioru pytań (scripts/build_eval_set.py).
REPORT_FOOTER = """
## Ograniczenia

**Szum etykiet ~5%.** Ręczny przegląd 20 losowych pozycji (seed 7) wykazał 1 pozycję
z błędnym ground truth (pytanie odpowiadające sąsiedniemu przepisowi) oraz 3 pozycje,
w których pytanie jest szersze niż wskazany artykuł. Szum jest identyczny dla wszystkich
wierszy, więc porównania względne pozostają wiarygodne - zaniżone są wartości bezwzględne.

**Definicja trafienia zaniża wszystkie wyniki.** Liczony jest wyłącznie fragment
*rozpoczynający* artykuł. Przy `chunk_size=512` znaków dalsze fragmenty długiego przepisu
są liczone jako pudło, mimo że są trafne. Dotyczy to jednakowo każdej konfiguracji.

**Walidacja nie jest niezależna.** Pytania generował i walidował ten sam model
(`gpt-4o-mini`, `temperature=0`), więc drugi przebieg wyłapuje oczywiste rozjazdy,
a nie subtelne.

**Możliwy wyciek leksykalny.** Pytania powstały z treści artykułu, który mają trafić.
Prompt wymaga parafrazy i zakazuje cytowania, ale nie da się tego wykluczyć w pełni.

**Rozmiar zbioru.** Różnic poniżej ~5 punktów procentowych nie należy traktować jako
istotnych. Przy n=38 `bge-reranker-base` wyglądał na lepszy od TinyBERTa; przy n=90
kolejność się odwróciła.

## Reprodukcja

```bash
docker compose exec api uv run --no-sync python scripts/build_eval_set.py
docker compose exec api uv run --no-sync python scripts/run_retrieval_eval.py
```

Kandydaci są cache'owani w `benchmarks/candidates.json` wraz z odciskiem palca zbioru
pytań - po zmianie zbioru skrypt pobiera ich ponownie automatycznie.

Interpretacja wyników: patrz README, sekcja Performance & Evaluation.
"""


def make_retriever(k: int, use_multi_query: bool):
    base = vector_store.as_retriever(search_kwargs={"k": k})
    if not use_multi_query:
        return base
    return MultiQueryRetriever.from_llm(
        retriever=base,
        llm=llm,
        prompt=MULTI_QUERY_PROMPT,
        include_original=INCLUDE_ORIGINAL,
    )


def fingerprint(dataset: list[dict]) -> str:
    payload = json.dumps(dataset, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def fetch_candidates(dataset: list[dict]) -> dict:
    strategies = {}
    for name, cfg in STRATEGIES.items():
        retriever = make_retriever(**cfg)
        rows = []
        for i, item in enumerate(dataset, start=1):
            started = time.perf_counter()
            docs = retriever.invoke(item["question"])
            rows.append({
                "question": item["question"],
                "source": item["source"],
                "article": item["article"],
                "latency": time.perf_counter() - started,
                "candidates": [
                    {"text": d.page_content, "source": d.metadata.get("source", "")}
                    for d in docs
                ],
            })
            print(f"  [{name}] {i}/{len(dataset)}", end="\r")
        print(f"  [{name}] gotowe ({len(dataset)} pytań)          ")
        strategies[name] = rows
    return {"fingerprint": fingerprint(dataset), "strategies": strategies}



def rerank_none(_model, _query, candidates):
    return candidates


def rerank_flashrank(model_name, query, candidates):
    from flashrank import Ranker, RerankRequest

    if model_name not in _MODELS:
        _MODELS[model_name] = Ranker(model_name=model_name, cache_dir=settings.RERANKER_CACHE_DIR)
    passages = [{"id": i, "text": c["text"]} for i, c in enumerate(candidates)]
    ranked = _MODELS[model_name].rerank(RerankRequest(query=query, passages=passages))
    return [candidates[r["id"]] for r in ranked]


def rerank_cross_encoder(model_name, query, candidates):
    from sentence_transformers import CrossEncoder

    if model_name not in _MODELS:
        print(f"  (ładuję {model_name})")
        _MODELS[model_name] = CrossEncoder(model_name, max_length=512)
    scores = _MODELS[model_name].predict([(query, c["text"]) for c in candidates])
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    return [candidates[i] for i in order]


RERANKERS = {
    "bez rerankera": (rerank_none, None),
    "TinyBERT (EN)": (rerank_flashrank, "ms-marco-TinyBERT-L-2-v2"),
    "MultiBERT": (rerank_flashrank, "ms-marco-MultiBERT-L-12"),
    "bge-reranker-base": (rerank_cross_encoder, "BAAI/bge-reranker-base"),
}



def hit_rank(docs: list[dict], expected_source: str, expected_article: str) -> int | None:
    """Pozycja (1-indeksowana) pierwszego fragmentu rozpoczynającego właściwy artykuł."""
    pattern = re.compile(rf"(?:^|\n)\s*Art\.\s*{re.escape(expected_article)}(?![\da-z])")
    for position, doc in enumerate(docs, start=1):
        if doc["source"] == expected_source and pattern.search(doc["text"]):
            return position
    return None


def score(rows: list[dict], rerank_fn, model_name, cutoff: int) -> dict:
    ranks, elapsed = [], []
    for row in rows:
        started = time.perf_counter()
        ordered = rerank_fn(model_name, row["question"], row["candidates"])
        elapsed.append(time.perf_counter() - started)
        ranks.append(hit_rank(ordered[:cutoff], row["source"], row["article"]))

    found = [rank for rank in ranks if rank is not None]
    return {
        "hit_rate": len(found) / len(ranks),
        "mrr": sum(1 / rank for rank in found) / len(ranks),
        "rerank_p50": statistics.median(elapsed),
        "retrieval_p50": statistics.median(row["latency"] for row in rows),
        "mean_candidates": statistics.mean(len(row["candidates"]) for row in rows),
    }


def load_candidates(dataset: list[dict], refresh: bool) -> dict:
    if CACHE.exists() and not refresh:
        cached = json.loads(CACHE.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint(dataset):
            print(f"Używam kandydatów z {CACHE}\n")
            return cached["strategies"]
        print("Zbiór pytań się zmienił - pobieram kandydatów ponownie.\n")

    print("Pobieram kandydatów...")
    payload = fetch_candidates(dataset)
    BENCHMARKS.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Zapisano do {CACHE}\n")
    return payload["strategies"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="pobierz kandydatów ponownie")
    args = parser.parse_args()

    dataset = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    print(f"{len(dataset)} pytań\n")

    candidates = load_candidates(dataset, args.refresh)

    print("=== Pokrycie zbioru kandydatów ===")
    recall_rows = []
    for name, rows in candidates.items():
        result = score(rows, rerank_none, None, cutoff=10_000)
        recall_rows.append((name, result))
        print(
            f"  {name:<22} Recall={result['hit_rate']:.3f}  "
            f"kandydatów={result['mean_candidates']:.0f}  p50={result['retrieval_p50']:.2f}s"
        )

    print(f"\n=== Jakość rankingu (@{K}) ===")
    ranking_rows = []
    for strategy_name, rows in candidates.items():
        for reranker_name, (fn, model_name) in RERANKERS.items():
            result = score(rows, fn, model_name, cutoff=K)
            label = f"{strategy_name} + {reranker_name}"
            ranking_rows.append((label, result))
            print(
                f"  {label:<46} HitRate@{K}={result['hit_rate']:.3f}  "
                f"MRR@{K}={result['mrr']:.3f}  p50={result['rerank_p50']:.3f}s"
            )

    BENCHMARKS.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        f.write("# Retrieval benchmark\n\n")
        f.write(f"Zbiór: {len(dataset)} pytań wygenerowanych z korpusu (jedno pytanie na losowy ")
        f.write("artykuł, próbkowanie warstwowe po pięciu ustawach, walidacja przez LLM). ")
        f.write("Trafienie = fragment rozpoczynający właściwy artykuł właściwej ustawy. ")
        f.write("Metryki liczone deterministycznie, bez LLM-as-a-judge.\n\n")
        f.write("## Pokrycie zbioru kandydatów\n\n")
        f.write("| Strategia | Recall | Śr. liczba kandydatów | p50 retrievalu |\n|---|---|---|---|\n")
        for name, r in recall_rows:
            f.write(f"| {name} | {r['hit_rate']:.3f} | {r['mean_candidates']:.0f} | {r['retrieval_p50']:.2f}s |\n")
        f.write(f"\n## Jakość rankingu (@{K})\n\n")
        f.write(f"| Konfiguracja | Hit Rate@{K} | MRR@{K} | p50 rerankingu |\n|---|---|---|---|\n")
        for name, r in ranking_rows:
            f.write(f"| {name} | {r['hit_rate']:.3f} | {r['mrr']:.3f} | {r['rerank_p50']:.3f}s |\n")
        f.write(REPORT_FOOTER)

    print(f"\nTabela zapisana do {OUT}")


if __name__ == "__main__":
    main()