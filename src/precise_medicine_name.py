import streamlit as st
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="Medical Assistant LLM",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar: File Upload
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF document", type="pdf", help="Upload the PDF containing medical criteria."
)

# Main Interface
def main():
    st.title("Medical Criteria Assistant")
    st.markdown(
        "This application uses a language model to help answer questions about medical criteria and drug usage based on an uploaded PDF."
    )

    if uploaded_file:
        # Show Progress
        with st.spinner("Reading and processing the document..."):
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

        with st.expander("Preview PDF Content", expanded=False):
            st.text_area("PDF Content", text[:2000], height=300)

        # Splitting text for better context handling
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        with st.spinner("Indexing document..."):
            vector_store = FAISS.from_texts(split_docs, embeddings)

        with st.spinner("Preparing Question-Answering Model..."):
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        st.success("Document processing complete.")

        # Question Answer Section
        st.subheader("Ask Questions")
        query = st.text_input("Enter your question about the document:")
        if query:
            with st.spinner("Generating answer..."):
                # Retrieve most relevant chunks
                docs = retriever.get_relevant_documents(query)

                # Extract the most relevant text snippets
                answer_snippets = [doc.page_content.strip() for doc in docs]

                # Filter answer to match exact query requirements
                if "criteria" in query.lower() or "which medicine" in query.lower():
                    relevant_snippet = next((snippet for snippet in answer_snippets if query.lower() in snippet.lower()), "No direct match found.")
                    answer = relevant_snippet
                else:
                    answer = "\n\n".join(answer_snippets)

            st.markdown("### Answer")
            st.text_area("", answer, height=200)

            with st.expander("Show Relevant Sections", expanded=False):
                for i, doc in enumerate(docs):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.text_area("", doc.page_content, height=150, key=f"chunk_{i}")
    else:
        st.info("Please upload a PDF to start.")

if __name__ == "__main__":
    main()
