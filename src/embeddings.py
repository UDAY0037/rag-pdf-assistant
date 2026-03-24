from sentence_transformers import SentenceTransformer


def load_embeddings():

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    return model


def create_embeddings(chunks, model):

    # chunks are already text strings
    texts = chunks

    embeddings = model.encode(texts)

    return embeddings, texts