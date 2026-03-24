import numpy as np


def retrieve(query, model, index, texts, k=3):

    query_embedding = model.encode([query])

    distances, indices = index.search(np.array(query_embedding), k)

    results = [texts[i] for i in indices[0]]

    return results