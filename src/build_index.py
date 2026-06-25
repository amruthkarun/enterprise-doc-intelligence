import os
from pdf_parser import extract_text_from_pdf
from embedder import load_model, create_embeddings, save_vector_store

from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIRECTORY = os.path.join(BASE_DIR, "..", "data")
INDEX_DIRECTORY = os.path.join(BASE_DIR, "..", "embeddings")


def prepare_documents():
    document_texts = []
    document_metadata = []

    for filename in os.listdir(DATA_DIRECTORY):
        if filename.lower().endswith(".pdf"):
            company_name = os.path.splitext(filename)[0].strip()
            file_path = os.path.join(DATA_DIRECTORY, filename)

            print(f"Processing {filename}")
            raw_text = extract_text_from_pdf(file_path)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            text_chunks = text_splitter.split_text(raw_text)

            for chunk in text_chunks:
                document_texts.append(chunk)
                document_metadata.append({"company": company_name})

    return document_texts, document_metadata


def main():
    os.makedirs(INDEX_DIRECTORY, exist_ok=True)

    document_texts, document_metadata = prepare_documents()
    model = load_model()
    embeddings = create_embeddings(model, document_texts)

    save_vector_store(
        embeddings,
        document_metadata,
        index_path=os.path.join(INDEX_DIRECTORY, "faiss_index.index"),
        meta_path=os.path.join(INDEX_DIRECTORY, "metadata.pkl")
    )

    print("Vector store built and saved.")


if __name__ == "__main__":
    main()
