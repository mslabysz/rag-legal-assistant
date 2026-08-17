import time
from rag_legal_assistant.graph.builder import app

TEST_QUESTIONS = [
    "Jaki jest termin przedawnienia roszczeń?",
    "Czym jest użytkowanie wieczyste?",
    "Kto wydaje pozwolenie na budowę?",
    "Co grozi za naruszenie praw autorskich?",
    "Czym jest czarna dziura?"  # out of domain test
]


def run_benchmark():
    print("Testing agent...\n")

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}/{len(TEST_QUESTIONS)}] Question: {question}")

        start_time = time.time()

        final_state = app.invoke({"query": question, "retry_count": 0})

        elapsed_time = time.time() - start_time

        print(f"⏱️  Time: {elapsed_time:.2f}s")
        print(f"🔄  Retries (Rewrite): {final_state.get('retry_count', 0)}")
        print(f"🤖  Answer: {final_state['answer']}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    run_benchmark()