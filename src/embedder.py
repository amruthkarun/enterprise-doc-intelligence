from sentence_transformers import SentenceTransformer
import os
import pickle
import faiss


def load_model():
    return SentenceTransformer('BAAI/bge-base-en-v1.5', device='cpu')  # Force to CPU

def create_embeddings(model, texts):
    if isinstance(texts, str):
        texts = [texts]  # Convert single string to list

    embeddings = model.encode(texts, show_progress_bar=True)

    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)  # Force to 2D

    return embeddings


def save_vector_store(embeddings, metadata, index_path, meta_path):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)

    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

def load_vector_store(index_path, meta_path):
    index = faiss.read_index(index_path)

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata
