import faiss
import numpy as np
from rag.embeddings import get_embeddings, get_text_embedding

class VectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []

    def add_texts(self, texts):
        embeddings = get_embeddings(texts)
        self.index.add(np.array(embeddings))
        self.texts.extend(texts)

    def search(self, query, top_k=3):
        query_vector = np.array([get_text_embedding(query)])
        distances, indices = self.index.search(query_vector, top_k)
        return [self.texts[i] for i in indices[0]]
