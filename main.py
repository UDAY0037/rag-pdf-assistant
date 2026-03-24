from src.pdf_loader import load_pdf
from src.text_splitter import split_text
from src.embeddings import load_embeddings, create_embeddings
from src.vector_store import create_vector_store
from src.retriever import retrieve
from src.llm import load_llm


def main():

    print("Loading PDF...")
    documents = load_pdf("data/sample.pdf")

    print("Splitting text...")
    chunks = split_text(documents)

    print("Loading embedding model...")
    model = load_embeddings()

    print("Creating embeddings...")
    embeddings, texts = create_embeddings(chunks, model)

    print("Creating FAISS index...")
    index = create_vector_store(embeddings)

    print("Loading LLM...")
    generator = load_llm()

    print("\nSystem Ready!")

    while True:

        query = input("\nAsk a question (type exit to quit): ")

        if query.lower() == "exit":
            break

        results = retrieve(query, model, index, texts)

        print("\nRetrieved Context:\n")

        for r in results:
            print(r)
            print("\n----\n")

        # combine retrieved chunks
        context = "\n".join(results)

        prompt = f"""
You are an AI assistant.

Use the provided context to answer the question clearly and concisely.

Context:
{context}

Question:
{query}

Answer:
"""

        # GENERATE ANSWER
        response = generator(prompt)

        answer = response[0]["generated_text"].replace(prompt, "")

        print("\nGenerated Answer:\n")
        print(answer)


if __name__ == "__main__":
    main()