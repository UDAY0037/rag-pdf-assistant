import streamlit as st
import os

from src.pdf_loader import load_pdf
from src.text_splitter import split_text
from src.embeddings import load_embeddings, create_embeddings
from src.vector_store import create_vector_store
from src.retriever import retrieve
from src.llm import load_llm


st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="📚")

st.title("📚 RAG PDF Assistant")
st.write("Upload a PDF and ask questions about it.")


# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")


@st.cache_resource
def load_models():
    embedding_model = load_embeddings()
    generator = load_llm()
    return embedding_model, generator


embedding_model, generator = load_models()


if uploaded_file:

    # Save uploaded file temporarily
    file_path = os.path.join("data", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    with st.spinner("Processing document..."):

        documents = load_pdf(file_path)

        chunks = split_text(documents)

        embeddings, texts = create_embeddings(chunks, embedding_model)

        index = create_vector_store(embeddings)

    st.success("Document processed! You can now ask questions.")

    query = st.text_input("Ask a question about the document")

    if query:

        results = retrieve(query, embedding_model, index, texts)

        context = "\n".join(results)

        prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

        response = generator(
            prompt,
            max_new_tokens=200,
            temperature=0.3
        )[0]["generated_text"]

        # Extract only the answer
        answer = response.split("Answer:")[-1].strip()

        st.subheader("Answer")
        st.success(answer)

else:
    st.info("Please upload a PDF to start.")