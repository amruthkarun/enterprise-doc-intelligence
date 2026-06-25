import os
import streamlit as st
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from user_auth import get_user_access
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ------------------- Load Vector Store -------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CHROMA_DIRECTORY = os.path.join(BASE_DIR, "..", "chroma_db")
DATA_DIRECTORY = os.path.join(BASE_DIR, "..", "data")

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"}
)

document_store = Chroma(
    persist_directory=CHROMA_DIRECTORY,
    embedding_function=embedding_model
)


# ------------------- Load Document Chunks -------------------
company_documents = {}
all_documents = []

for filename in os.listdir(DATA_DIRECTORY):
    if filename.lower().endswith(".pdf"):
        company_name = os.path.splitext(filename)[0].strip()
        file_path = os.path.join(DATA_DIRECTORY, filename)

        loader = PyPDFLoader(file_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        document_chunks = splitter.split_documents(pages)

        company_documents[company_name] = document_chunks
        for chunk in document_chunks:
            all_documents.append(Document(
                page_content=chunk.page_content,
                metadata={"source": file_path, "company": company_name}
            ))

store_is_empty = True
try:
    store_is_empty = document_store._collection.count() == 0
except Exception:
    store_is_empty = not os.path.exists(CHROMA_DIRECTORY) or not os.listdir(CHROMA_DIRECTORY)

if store_is_empty and all_documents:
    document_store.add_documents(all_documents)


# ------------------- Search Function -------------------
def search(query, authorized_companies, k=5):
    search_hits = document_store.similarity_search(query, k=k)

    filtered_results = []
    for hit in search_hits:
        hit_metadata = hit.metadata
        source_path = hit_metadata.get('source', '')

        company_name = os.path.basename(source_path).split(".")[0].strip()

        if company_name in authorized_companies:
            filtered_results.append({
                "company": company_name,
                "content": hit.page_content
            })

    return filtered_results


# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Enterprise Document Intelligence Platform", layout="wide")
st.title("🏢 Enterprise Document Intelligence Platform")

# Simulate Login
st.sidebar.header("User Login")
user_email = st.sidebar.text_input("Enter Email", value="")

if user_email:
    authorized_companies = get_user_access(user_email)

    if not authorized_companies:
        st.sidebar.error("❌ Access Denied. No documents available for this user.")
    else:
        st.sidebar.success(f"✅ Access Granted for: {', '.join(authorized_companies)}")

        # Initialize chat history per user
        if "chat_histories" not in st.session_state:
            st.session_state.chat_histories = {}

        if user_email not in st.session_state.chat_histories:
            st.session_state.chat_histories[user_email] = []

        chat_history = st.session_state.chat_histories[user_email]

        st.subheader(f"🔎 Query Documents ({', '.join(authorized_companies)})")

        query = st.text_input("Enter your query:")

        if st.button("Search"):
            if query:
                chat_history.append({"role": "user", "content": query})

                search_results = search(query, authorized_companies)

                if search_results:
                    for result in search_results:
                        st.markdown(f"**Company:** {result['company']}")
                        st.markdown(f"**Excerpt:** {result['content']}")
                        st.markdown("---")
                        chat_history.append({"role": "assistant", "content": result['content']})
                else:
                    st.warning("No relevant information found.")
                    chat_history.append({"role": "assistant", "content": "No relevant information found."})

        # Reset Chat Button
        if st.sidebar.button("Reset Chat"):
            st.session_state.chat_histories[user_email] = []
            st.sidebar.success("Chat history cleared. You can start fresh.")

        # Display chat history
        st.subheader("🗂️ Conversation History")
        for chat in chat_history:
            role = "👤 You" if chat["role"] == "user" else "🤖 Assistant"
            st.markdown(f"**{role}:** {chat['content']}")
            st.markdown("---")
