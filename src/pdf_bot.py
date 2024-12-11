import streamlit as st
from transformers import pipeline
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import hashlib
import pandas as pd
import spacy
from pdf2image import convert_from_path
import pytesseract
import os
from tempfile import NamedTemporaryFile

# Load the spaCy model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.max_length = 10_000_000  # Handle large text
nlp.add_pipe('sentencizer')  # Add sentencizer

# Streamlit app configuration
st.set_page_config(page_title="Enhanced PDF QA App", layout="wide")
st.title("📄 Enhanced PDF Question Answering App with Table Support")

# Sidebar for file upload
st.sidebar.header("Upload your PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

# Initialize session state variables
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = ""
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'file_hash' not in st.session_state:
    st.session_state.file_hash = None
if 'tables' not in st.session_state:
    st.session_state.tables = []

@st.cache_resource
def load_models():
    """Load the QA and embedding models."""
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return qa_model, embedding_model

@st.cache_resource
def load_paraphrasing_model():
    """Load the paraphrasing model."""
    return pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_and_tables_from_pdf(file):
    """Extract text and tables from a PDF using pdfplumber, with OCR fallback."""
    text = ""
    tables = []
    with pdfplumber.open(file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                st.warning(f"Page {page_num + 1} text extraction failed. Using OCR...")
                text += ocr_pdf(file)  # Use OCR for non-extractable pages

            # Extract tables
            page_tables = page.extract_tables()
            for table in page_tables:
                if table:
                    try:
                        header = table[0]
                        data_rows = table[1:]
                        if any(col == '' or col is None for col in header) or len(set(header)) != len(header):
                            header = [f"Column {i+1}" for i in range(len(header))]
                        df = pd.DataFrame(data_rows, columns=header)
                        tables.append(df)
                    except Exception as e:
                        st.warning(f"Failed to process a table on page {page_num + 1}: {e}")
    return text, tables

def ocr_pdf(file):
    """Perform OCR on a PDF file, saving it temporarily if needed."""
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(file.read())  # Save the uploaded file to the temp location
        temp_file_path = temp_pdf.name

    try:
        # Convert PDF pages to images
        images = convert_from_path(temp_file_path)
        ocr_text = ""
        for image in images:
            ocr_text += pytesseract.image_to_string(image) + "\n"
        return ocr_text
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)

def split_text_into_chunks(text, max_chunk_size=300):
    """Split text into chunks using spaCy's sentencizer."""
    if not isinstance(text, str) or not text:
        st.error("The extracted text is not a valid string.")
        return []
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def process_tables(tables):
    """Convert tables to a string format suitable for QA."""
    tables_text = ""
    for idx, table in enumerate(tables):
        tables_text += f"\n\nTable {idx + 1}:\n"
        tables_text += table.to_string(index=False)
    return tables_text

def build_faiss_index(embedding_model, chunks):
    """Build a FAISS index."""
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

def find_answer_in_tables(question, tables):
    """Search for an answer in the extracted tables."""
    for idx, table in enumerate(tables):
        if any(question.lower() in str(cell).lower() for cell in table.values.flatten()):
            return f"Found relevant data in Table {idx + 1}:\n{table.to_string(index=False)}"
    return None

def calculate_file_hash(file):
    """Calculate a hash for the uploaded file to detect changes."""
    file.seek(0)
    file_hash = hashlib.md5(file.read()).hexdigest()
    file.seek(0)
    return file_hash

# Main Logic
if uploaded_file is not None:
    current_file_hash = calculate_file_hash(uploaded_file)
    if st.session_state.file_hash != current_file_hash:
        st.session_state.pdf_text = ""
        st.session_state.chunks = []
        st.session_state.embeddings = None
        st.session_state.index = None
        st.session_state.tables = []
        st.session_state.file_hash = current_file_hash

    if st.session_state.pdf_text == "":
        with st.spinner("Extracting text and tables from PDF..."):
            pdf_text, tables = extract_text_and_tables_from_pdf(uploaded_file)
            st.session_state.pdf_text = pdf_text
            st.session_state.tables = tables
        st.success("PDF loaded successfully!")

        with st.spinner("Processing tables..."):
            tables_text = process_tables(st.session_state.tables)
            st.session_state.pdf_text += tables_text

        with st.spinner("Splitting text into chunks..."):
            chunks = split_text_into_chunks(st.session_state.pdf_text)
            st.session_state.chunks = chunks

        with st.spinner("Generating embeddings and building index..."):
            qa_model, embedding_model = load_models()
            index, embeddings = build_faiss_index(embedding_model, st.session_state.chunks)
            st.session_state.index = index
            st.session_state.embeddings = embeddings

    with st.expander("Show Extracted Text"):
        st.text_area("Extracted Text", st.session_state.pdf_text, height=300)
    if st.session_state.tables:
        with st.expander("Show Extracted Tables"):
            for idx, table in enumerate(st.session_state.tables):
                st.subheader(f"Table {idx + 1}")
                st.dataframe(table)

st.header("Ask a Question")
question = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing your question..."):
            qa_model, embedding_model = load_models()
            paraphrase_model = load_paraphrasing_model()
            question_embedding = embedding_model.encode([question], convert_to_tensor=True).cpu().numpy().astype('float32')
            D, I = st.session_state.index.search(question_embedding, k=5)
            relevant_chunks = [st.session_state.chunks[idx] for idx in I[0] if idx < len(st.session_state.chunks)]
            table_answer = find_answer_in_tables(question, st.session_state.tables)

            if table_answer:
                st.success("Answer from tables:")
                st.write(table_answer)
            else:
                best_answer = None
                best_score = 0
                for chunk in relevant_chunks:
                    try:
                        result = qa_model(question=question, context=chunk)
                        if result['score'] > best_score:
                            best_answer = result['answer']
                            best_score = result['score']
                    except Exception as e:
                        st.error(f"Error: {e}")
                if best_answer:
                    st.success(f"**Answer:** {best_answer}")
                    st.write(f"**Confidence Score:** {best_score:.2f}")
                else:
                    st.error("No relevant answer found.")