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
st.set_page_config(page_title="Enterprise Document Intelligence Platform", page_icon="🏢", layout="wide")
st.title("🏢 Enterprise Document Intelligence Platform")
st.markdown(
    """
    Welcome to the Enterprise Document Intelligence platform. Login with your corporate email to search authorized transcripts,
    find relevant excerpts, and keep a history of your questions.
    """
)

# Sidebar login and status
st.sidebar.header("User Login")
st.sidebar.write("Sign in to access your authorized transcripts and document intelligence tools.")
user_email = st.sidebar.text_input("Email", value="", placeholder="name@example.com")

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}
if "query" not in st.session_state:
    st.session_state.query = ""
if "last_results" not in st.session_state:
    st.session_state.last_results = []

if user_email:
    authorized_companies = get_user_access(user_email)

    if not authorized_companies:
        st.sidebar.error("❌ Access Denied. No documents available for this user.")
    else:
        st.sidebar.success("✅ Access Granted")
        st.sidebar.markdown("**Authorized Documents**")
        for company in authorized_companies:
            st.sidebar.write(f"- {company}")
        st.sidebar.markdown("---")
        if st.sidebar.button("Reset Session"):
            st.session_state.chat_histories[user_email] = []
            st.session_state.last_results = []
            st.session_state.query = ""
            st.sidebar.success("Session reset. You can start fresh.")
        st.sidebar.info("Ask about earnings, strategy, risks, or other topics covered in your authorized documents.")

        if user_email not in st.session_state.chat_histories:
            st.session_state.chat_histories[user_email] = []

        chat_history = st.session_state.chat_histories[user_email]

        left_col, right_col = st.columns((2.5, 1))

        with left_col:
            st.subheader(f"🔎 Query Documents ({', '.join(authorized_companies)})")
            with st.form(key="query_form"):
                query = st.text_input(
                    "Enter your question",
                    value=st.session_state.query,
                    placeholder="What are the latest earnings highlights?"
                )
                search_button = st.form_submit_button("Search")

                if search_button:
                    st.session_state.query = query
                    if query.strip():
                        chat_history.append({"role": "user", "content": query})
                        search_results = search(query, authorized_companies)
                        st.session_state.last_results = search_results

                        if search_results:
                            for result in search_results:
                                chat_history.append({"role": "assistant", "content": result['content']})
                        else:
                            st.warning("No relevant information found.")
                            chat_history.append({"role": "assistant", "content": "No relevant information found."})
                    else:
                        st.warning("Please enter a question before searching.")

            st.markdown("---")
            st.subheader("📄 Search Results")
            if st.session_state.last_results:
                for result in st.session_state.last_results:
                    with st.expander(f"{result['company']}", expanded=False):
                        st.write(result['content'])
            else:
                st.info("Search results will appear here after you submit a query.")

        with right_col:
            st.subheader("📌 Quick Overview")
            st.metric("Authorized Documents", len(authorized_companies))
            st.metric("History Entries", len(chat_history))
            st.markdown(
                """
                **Tips:**
                - Ask one focused question at a time.
                - Use company names or document topics.
                - Reset the session to clear history.
                """
            )
            if st.session_state.query:
                st.markdown(f"**Last query:** {st.session_state.query}")

        st.markdown("---")
        st.subheader("🗂️ Conversation History")
        if chat_history:
            for chat in chat_history:
                if chat["role"] == "user":
                    st.success(f"👤 {chat['content']}")
                else:
                    st.info(f"🤖 {chat['content']}")
                st.markdown("---")
        else:
            st.info("Your chat history is empty. Ask a question to begin.")
