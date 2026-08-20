# Retrieval benchmark

Zbiór: 90 pytań wygenerowanych z korpusu (jedno pytanie na losowy artykuł, próbkowanie warstwowe po pięciu ustawach, walidacja przez LLM). Trafienie = fragment rozpoczynający właściwy artykuł właściwej ustawy. Metryki liczone deterministycznie, bez LLM-as-a-judge.

## Pokrycie zbioru kandydatów

| Strategia | Recall | Śr. liczba kandydatów | p50 retrievalu |
|---|---|---|---|
| dense k=20 | 0.689 | 20 | 0.03s |
| dense k=44 | 0.789 | 44 | 0.03s |
| multi-query k=20x4 | 0.844 | 44 | 1.51s |

## Jakość rankingu (@5)

| Konfiguracja | Hit Rate@5 | MRR@5 | p50 rerankingu |
|---|---|---|---|
| dense k=20 + bez rerankera | 0.422 | 0.289 | 0.000s |
| dense k=20 + TinyBERT (EN) | 0.578 | 0.469 | 0.023s |
| dense k=20 + MultiBERT | 0.367 | 0.222 | 0.728s |
| dense k=20 + bge-reranker-base | 0.556 | 0.395 | 1.126s |
| dense k=44 + bez rerankera | 0.422 | 0.289 | 0.000s |
| dense k=44 + TinyBERT (EN) | 0.611 | 0.463 | 0.042s |
| dense k=44 + MultiBERT | 0.333 | 0.172 | 1.651s |
| dense k=44 + bge-reranker-base | 0.511 | 0.298 | 2.244s |
| multi-query k=20x4 + bez rerankera | 0.400 | 0.321 | 0.000s |
| multi-query k=20x4 + TinyBERT (EN) | 0.667 | 0.485 | 0.039s |
| multi-query k=20x4 + MultiBERT | 0.311 | 0.178 | 1.614s |
| multi-query k=20x4 + bge-reranker-base | 0.567 | 0.365 | 2.192s |

## Wnioski

**Reranking zarabia na siebie z dużą przewagą.** Na tym samym zbiorze kandydatów (multi-query) Hit Rate@5 rośnie z 0.400 do 0.667, a MRR@5 z 0.321 do 0.485 — za 39 ms na zapytanie.

**Model multilingual jest gorszy niż brak rerankingu.** `ms-marco-MultiBERT-L-12` przegrywa z konfiguracją bez rerankera we wszystkich trzech strategiach kandydatów (0.367 vs 0.422, 0.333 vs 0.422, 0.311 vs 0.400). Wbrew intuicji, że angielski cross-encoder nie poradzi sobie z polskim tekstem prawniczym, wygrywa 4-megabajtowy `ms-marco-TinyBERT-L-2-v2` — najprawdopodobniej dlatego, że kwantyzowany mBERT jest po prostu słabym modelem, a sygnał leksykalny (numery, terminologia o łacińskich rdzeniach, nazwy własne) przenosi się między językami.

**Duży cross-encoder nie opłaca się na CPU.** `BAAI/bge-reranker-base` przegrywa z TinyBERTem o 10 punktów Hit Rate i 33% MRR, będąc 56× wolniejszym. Przy n=38 wyglądał na minimalnie lepszy — rozszerzenie zbioru do n=90 odwróciło ten wynik, co dobrze pokazuje, jak zwodnicze są małe zbiory testowe.

**Multi-query dodaje realny recall, ale kosztowny.** Przy zbliżonej liczbie kandydatów (44) multi-query daje 0.844 recallu wobec 0.789 dla zwykłego dense, więc przewaga nie wynika wyłącznie z większego zbioru. Kosztuje jednak 1,48 s i jedno dodatkowe wywołanie LLM na zapytanie. Tańsza alternatywa: `dense k=44 + TinyBERT` osiąga 0.611 Hit Rate w ~70 ms, czyli 92% jakości najlepszej konfiguracji przy 22× mniejszej latencji.

**Konfiguracja produkcyjna** odpowiada najlepszemu wierszowi tabeli: multi-query z `include_original=True`, `k=20` na wariant, FlashRank `ms-marco-TinyBERT-L-2-v2`, `top_n=5`.

## Zapas do wykorzystania

Sufit recallu to 0.844, a najlepszy Hit Rate@5 to 0.667 — ranking traci więc 18 punktów z tego, co zostało już znalezione, a 15,6% pytań nie trafia do zbioru kandydatów w ogóle.

## Ograniczenia

**Szum etykiet ~5%.** Ręczny przegląd 20 losowych pozycji (seed 7) wykazał 1 pozycję z błędnym ground truth (pytanie odpowiadające sąsiedniemu przepisowi) oraz 3 pozycje, w których pytanie jest szersze niż wskazany artykuł. Szum jest identyczny dla wszystkich wierszy, więc porównania względne pozostają wiarygodne — zaniżone są wartości bezwzględne.

**Definicja trafienia zaniża wszystkie wyniki.** Liczony jest wyłącznie fragment *rozpoczynający* artykuł. Przy `chunk_size=512` znaków dalsze fragmenty długiego przepisu są liczone jako pudło, mimo że są trafne. Dotyczy to jednakowo każdej konfiguracji.

**Walidacja nie jest niezależna.** Pytania generował i walidował ten sam model (`gpt-4o-mini`, `temperature=0`), więc drugi przebieg wyłapuje oczywiste rozjazdy, a nie subtelne.

**Możliwy wyciek leksykalny.** Pytania powstały z treści artykułu, który mają trafić. Prompt wymaga parafrazy i zakazuje cytowania, ale nie da się tego wykluczyć w pełni.

**Rozmiar zbioru.** Różnic poniżej ~5 punktów procentowych nie należy traktować jako istotnych. Przy n=38 `bge-reranker-base` wyglądał na lepszy od TinyBERTa; przy n=90 kolejność się odwróciła.

## Reprodukcja

```bash
docker compose exec api uv run --no-sync python scripts/build_eval_set.py
docker compose exec api uv run --no-sync python scripts/run_retrieval_eval.py
```

Kandydaci są cache'owani w `benchmarks/candidates.json` wraz z odciskiem palca zbioru pytań — po zmianie zbioru skrypt pobiera ich ponownie automatycznie.

Interpretacja wyników: patrz README, sekcja Performance & Evaluation.
