from src.retriever import retrieve_documents


def generate_answer(query, vector_store, llm):

    docs = retrieve_documents(vector_store, query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    result = llm(prompt)

    return result[0]["generated_text"]