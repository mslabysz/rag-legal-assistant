from rag_legal_assistant.graph.builder import app


def main():
    print("Agent running. Ask a question (or type 'exit'):")

    while True:
        query = input("\nQ: ")
        if query.lower() == 'exit':
            break

        print("Thinking...")

        final_state = app.invoke({"query": query, "retry_count": 0})

        print("\n============================================================")
        print("A:", final_state["answer"])


if __name__ == "__main__":
    main()