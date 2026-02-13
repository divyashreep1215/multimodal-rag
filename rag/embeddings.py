from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)

def get_text_embedding(text):
    return model.encode(text)

def get_embeddings(text_list):
    return model.encode(text_list)
