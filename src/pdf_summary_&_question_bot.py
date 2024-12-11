import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pdfplumber
import hashlib
import spacy
from nltk.tokenize import sent_tokenize

# Load the spaCy model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.max_length = 10_000_000
nlp.add_pipe('sentencizer')

st.set_page_config(page_title="PDF QA and Summarization App", layout="wide")

@st.cache_resource
def load_models():
    """Load the QA and embedding models."""
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return qa_model, embedding_model

@st.cache_resource
def load_summarization_model():
    """Load a summarization model."""
    return pipeline("summarization", model="facebook/bart-large-cnn")

def calculate_file_hash(file):
    """Calculate a hash for the uploaded file to detect changes."""
    file.seek(0)
    file_bytes = file.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    file.seek(0)
    return file_hash

def extract_text_and_tables_from_pdf(file):
    """Extract text from the PDF."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text, []  # Simplified for focus on text extraction

def split_text_into_chunks(text, max_chunk_size=1000):
    """Split text into manageable chunks."""
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

def build_faiss_index(embedding_model, chunks):
    """Build a FAISS index for the text chunks."""
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

def summarize_text(text, summarizer, max_length=300, min_length=100):
    """Summarize text, handling large inputs by chunking."""
    def chunk_text(text, max_tokens=1000):
        sentences = text.split('. ')
        chunks, current_chunk = [], ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_tokens:
                current_chunk += sentence + ". "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    chunks = chunk_text(text)
    summarized_chunks = [
        summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        for chunk in chunks
    ]
    return " ".join(summarized_chunks)

def find_section(text, query):
    """Find a section in the document based on a query."""
    query_lower = query.lower()
    lines = text.split("\n")
    relevant_section = []
    found = False

    for line in lines:
        if query_lower in line.lower():
            found = True
        if found:
            relevant_section.append(line)
            if line.strip() == "" or line.startswith("Chapter"):
                break

    return "\n".join(relevant_section).strip() if relevant_section else None

def handle_question_answering(uploaded_file):
    st.header("Ask a Question about the PDF")
    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            try:
                pdf_text, _ = extract_text_and_tables_from_pdf(uploaded_file)
                qa_model, embedding_model = load_models()
                chunks = split_text_into_chunks(pdf_text)
                question_embedding = embedding_model.encode([question], convert_to_tensor=True)
                question_embedding = question_embedding.cpu().numpy().astype('float32')
                index, _ = build_faiss_index(embedding_model, chunks)
                D, I = index.search(question_embedding, k=5)
                relevant_chunks = [chunks[idx] for idx in I[0] if idx < len(chunks)]
                best_answer, best_score = None, 0
                for chunk in relevant_chunks:
                    result = qa_model(question=question, context=chunk)
                    if result['score'] > best_score:
                        best_answer = result['answer']
                        best_score = result['score']
                if best_answer:
                    st.success("Here's the answer to your question:")
                    st.write(f"**Answer:** {best_answer}")
                    st.write(f"**Confidence Score:** {best_score:.2f}")
                else:
                    st.error("Sorry, I couldn't find an answer to your question.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

def handle_summarization(uploaded_file):
    st.header("Summarize Text or Sections from the PDF")
    summarize_type = st.radio("Choose what to summarize:", ["Full Document", "Specific Section", "Paste Custom Text"])

    if summarize_type == "Full Document":
        try:
            pdf_text, _ = extract_text_and_tables_from_pdf(uploaded_file)
            summarizer = load_summarization_model()
            with st.spinner("Summarizing the document..."):
                summary = summarize_text(pdf_text, summarizer)
                st.success("Summarization Complete!")
                st.write("**Summary:**")
                st.text(summary)
        except Exception as e:
            st.error(f"An error occurred: {e}")

    elif summarize_type == "Specific Section":
        search_query = st.text_input("Enter the section heading or keyword to summarize:")
        if st.button("Search and Summarize"):
            try:
                pdf_text, _ = extract_text_and_tables_from_pdf(uploaded_file)
                section_text = find_section(pdf_text, search_query)
                if section_text:
                    st.write("**Relevant Section Found:**")
                    st.text(section_text)
                    summarizer = load_summarization_model()
                    with st.spinner("Summarizing the section..."):
                        summary = summarize_text(section_text, summarizer)
                        st.success("Summarization Complete!")
                        st.write("**Summary:**")
                        st.text(summary)
                else:
                    st.warning(f"No section found for the query: {search_query}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    elif summarize_type == "Paste Custom Text":
        user_text = st.text_area("Paste the content to summarize:")
        if st.button("Summarize Custom Text"):
            if user_text.strip():
                try:
                    summarizer = load_summarization_model()
                    with st.spinner("Summarizing..."):
                        summary = summarize_text(user_text, summarizer)
                        st.success("Summarization Complete!")
                        st.write("**Summary:**")
                        st.text(summary)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please paste some text to summarize.")

uploaded_file = st.sidebar.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    st.session_state.file_hash = calculate_file_hash(uploaded_file)

    tab1, tab2 = st.tabs(["Question Answering", "Summarization"])

    with tab1:
        handle_question_answering(uploaded_file)

    with tab2:
        handle_summarization(uploaded_file)
