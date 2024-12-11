import streamlit as st
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline

# Set the page configuration
st.set_page_config(page_title="Safqore PDF Agent", layout="wide")

# Initialize the Hugging Face question-answering pipeline
if 'qa_pipeline' not in st.session_state:
    st.session_state.qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Load the pre-trained SentenceTransformer model
if 'embedder' not in st.session_state:
    st.session_state.embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index and chunk storage if not already done
if 'index' not in st.session_state:
    dimension = 384  # Embedding size for 'all-MiniLM-L6-v2'
    st.session_state.index = faiss.IndexFlatL2(dimension)
    st.session_state.chunk_texts = []

st.title("Safqore PDF Agent")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.header("Upload PDF")
    # File upload
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            try:
                # Reset index and chunk_texts for a new PDF
                dimension = 384
                st.session_state.index = faiss.IndexFlatL2(dimension)
                st.session_state.chunk_texts = []

                # Read PDF
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text

                if not text:
                    st.error("No text found in the PDF.")
                else:
                    # Split text into chunks
                    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=4000, chunk_overlap=50)
                    chunks = text_splitter.split_text(text)

                    if not chunks:
                        st.error("Failed to split the text into chunks.")
                    else:
                        # Embed the chunks
                        chunk_vectors = st.session_state.embedder.encode(chunks)

                        # Store vectors in FAISS index
                        faiss_index = np.array(chunk_vectors, dtype=np.float32)
                        st.session_state.index.add(faiss_index)

                        # Store chunks
                        st.session_state.chunk_texts.extend(chunks)

                        # Check FAISS index status
                        faiss_index_count = st.session_state.index.ntotal

                        st.success("PDF uploaded and processed successfully!")
                        st.write(f"The FAISS index has successfully processed and contains {faiss_index_count} vectors.")

                        # Collapsible sections for previews
                        with st.expander("Preview of PDF Chunks"):
                            for i, chunk in enumerate(chunks[:5]):
                                st.text(f"Chunk {i + 1}: {chunk}")

                        with st.expander("Preview of Vectors"):
                            for i, vector in enumerate(chunk_vectors[:5]):
                                st.text(f"Vector {i + 1}: {vector}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

with col2:
    st.header("Ask a Question")
    # Query the PDF content using Hugging Face model
    query = st.text_input("Enter your question:")
    if st.button("Search"):
        if not query.strip():
            st.error("Please enter a valid search query!")
        elif 'index' not in st.session_state or st.session_state.index.ntotal == 0:
            st.error("Please upload and process a PDF first.")
        else:
            with st.spinner("Searching..."):
                try:
                    # Convert the query into a vector
                    query_vector = st.session_state.embedder.encode([query])[0]

                    # Perform a similarity search in FAISS
                    query_vector = np.array([query_vector], dtype=np.float32)
                    k = 5  # Number of closest neighbors to retrieve
                    distances, indices = st.session_state.index.search(query_vector, k)

                    # Gather the most relevant chunks
                    relevant_chunks = []
                    for idx in indices[0]:
                        if idx < len(st.session_state.chunk_texts):
                            relevant_chunks.append(st.session_state.chunk_texts[idx])

                    # Combine relevant chunks into a single context
                    context = " ".join(relevant_chunks)

                    # Use Hugging Face model to generate an answer
                    qa_input = {
                        'question': query,
                        'context': context
                    }
                    answer = st.session_state.qa_pipeline(qa_input)

                    st.subheader("Answer:")
                    st.write(answer['answer'])

                    with st.expander("Relevant Chunks"):
                        for i, chunk in enumerate(relevant_chunks):
                            st.text(f"Chunk {i + 1}: {chunk}")

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")