from embedder import load_model, load_vector_store
import numpy as np
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INDEX_FILEPATH = os.path.join(BASE_DIR, "..", "embeddings", "faiss_index.index")
METADATA_FILEPATH = os.path.join(BASE_DIR, "..", "embeddings", "metadata.pkl")

model = load_model()
index, metadata_records = load_vector_store(INDEX_FILEPATH, METADATA_FILEPATH)


def search(query, authorized_companies, top_k=5):
    query_embedding = model.encode([query])

    distances, indices = index.search(query_embedding, top_k)

    search_results = []
    for document_index, distance in zip(indices[0], distances[0]):
        if document_index == -1:
            continue

        record = metadata_records[document_index]
        if record["company"] in authorized_companies:
            search_results.append((record["company"], distance))

    return search_results


def retrieve_answers(query, authorized_companies, document_texts, top_k=3):
    query_embedding = model.encode([query])

    distances, indices = index.search(query_embedding, top_k)

    retrieved_results = []
    for document_index, distance in zip(indices[0], distances[0]):
        if document_index == -1:
            continue

        record = metadata_records[document_index]
        if record["company"] in authorized_companies:
            retrieved_results.append({
                "company": record["company"],
                "content": document_texts[document_index],
                "score": float(distance)
            })

    return retrieved_results
