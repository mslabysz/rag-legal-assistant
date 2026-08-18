import sys
import asyncio

# --- HOTFIX: Łata na zepsute importy w Ragas 0.4.x przy użyciu najnowszego LangChaina ---
from unittest.mock import MagicMock
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
sys.modules['langchain_community.embeddings.vertexai'] = MagicMock()
# ---------------------------------------------------------------------------------------

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from rag_legal_assistant.graph.builder import app
from rag_legal_assistant.eval.dataset import EVALUATION_DATA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from rag_legal_assistant.config import settings

async def main():
    print("Starting data collection from Agent for RAGAS evaluation...")

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for item in EVALUATION_DATA:
        question = item["question"]
        print(f"\nAsking question: {question}")

        final_state = await app.ainvoke({"query": question, "retry_count": 0})
        
        answer = final_state["answer"]
        docs = final_state.get("documents", [])

        contexts = [doc.get("text", "") for doc in docs]
        
        data["question"].append(question)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(item["ground_truth"])
        
    print("\nResponses collected. Converting to Dataset format...")
    dataset = Dataset.from_dict(data)
    
    print("\nStarting evaluation via LLM-as-a-Judge (this may take a minute)...")

    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=settings.LLM_MODEL, temperature=0, api_key=settings.OPENAI_API_KEY))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY))

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )
    
    print("\n--- EVALUATION RESULTS ---")
    print(result)

    df = result.to_pandas()
    df.to_csv("benchmark_ragas_results.csv", index=False)
    print("\nDetailed scores have been saved to benchmark_ragas_results.csv")

if __name__ == "__main__":
    asyncio.run(main())
