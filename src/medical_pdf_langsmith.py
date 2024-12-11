import os
import streamlit as st
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langsmith import Client

# --- LangSmith Config ---


# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="Medical Assistant LLM with LangSmith",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize LangSmith Client (replace with your API key)
LANGCHAIN_TRACING_V2 = True
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"

# Get API Key from environment
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
if not LANGSMITH_API_KEY:
    raise ValueError("LangSmith API key is missing. Set LANGSMITH_API_KEY in your environment.")

langsmith_client = Client(api_key=LANGSMITH_API_KEY)


# LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT="medical_pdf_qa"
langsmith_client = Client(api_key=LANGSMITH_API_KEY)

# Function: Preprocess the uploaded document
def preprocess_document(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_text(text)

    # Generate embeddings and create vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(split_docs, embeddings)

    return vector_store, text

# Function: Handle user query with LangSmith integration
def handle_query_with_langsmith(vector_store, query):
    start_trace(session_name="PDF_QA_Session")  # Start LangSmith trace
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query)
    answer_snippets = [doc.page_content.strip() for doc in docs]

    # Generate the answer
    if "criteria" in query.lower() or "medicine" in query.lower():
        answer = next((snippet for snippet in answer_snippets if query.lower() in snippet.lower()), "No direct match found.")
    else:
        answer = "\n\n".join(answer_snippets)

    # Log interaction with LangSmith
    langsmith_client.log_message({
        "query": query,
        "answer": answer,
        "relevant_documents": [doc.page_content for doc in docs],
    })
    end_trace()  # End LangSmith trace
    return answer

# Main Function: Streamlit App
def main():
    st.title("Medical Criteria Assistant with LangSmith")
    st.markdown(
        "This application uses a language model to help answer questions about medical criteria and drug usage based on an uploaded PDF, with added tracing and debugging support from LangSmith."
    )

    # Sidebar for file upload
    st.sidebar.header("Upload Document")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF document", type="pdf", help="Upload the PDF containing medical criteria."
    )

    if uploaded_file:
        # Process the uploaded file
        with st.spinner("Processing the document..."):
            vector_store, text = preprocess_document(uploaded_file)

        # Display document preview
        with st.expander("Preview PDF Content", expanded=False):
            st.text_area("PDF Content", text[:2000], height=300)

        st.success("Document processing complete.")

        # Query Section
        st.subheader("Ask Questions")
        query = st.text_input("Enter your question about the document:")
        if query:
            with st.spinner("Finding the answer..."):
                answer = handle_query_with_langsmith(vector_store, query)
            st.markdown("### Answer")
            st.text_area("", answer, height=200)

    else:
        st.info("Please upload a PDF to start.")

if __name__ == "__main__":
    main()