
from time import perf_counter

from src.answer import AnswerService
from src.retrieve import retrieve


def ask(question: str) -> str:
    start = perf_counter()

    results = retrieve(question)

    retrieved_at = perf_counter()

    for index, result in enumerate(results, start=1):
        print(
            f"\n[Retrieved {index}] "
            f"{result.get('source')} "
            f"(page {result.get('page')}, "
            f"score={result.get('score', 0):.4f})"
        )
        print(result.get("text", ""))


    if not results:
        print(f"[Timing] Retrieval: {retrieved_at - start:.2f}s")
        return "I could not find relevant information in the documents."

    answer_service = AnswerService(model="qwen3:0.6b")
    # answer_service = AnswerService(model="qwen3:latest")
    answer = answer_service.generate(
        question=question,
        context_chunks=results,
    )

    finished_at = perf_counter()

    print(f"[Timing] Retrieval: {retrieved_at - start:.2f}s")
    print(f"[Timing] Generation: {finished_at - retrieved_at:.2f}s")
    print(f"[Timing] Total: {finished_at - start:.2f}s")

    return answer


if __name__ == "__main__":
    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            print(f"\nAssistant: {ask(question)}")
        except Exception as error:
            print(f"\nError: {error}")
