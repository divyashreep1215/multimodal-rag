from sentence_transformers import SentenceTransformer
import numpy as np
import config

model = SentenceTransformer(config.EMBEDDING_MODEL)

def get_text_embedding(text):
    return model.encode(text)

def get_embeddings(text_list):
    return model.encode(text_list)
