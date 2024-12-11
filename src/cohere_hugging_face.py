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
def load_cohere_model():
    """Load the Cohere LLM for summarization."""
    # return pipeline("text-generation", model="Cohere/command-nightly")
    return pipeline("text-generation", model="gpt2")
@st.cache_data
def calculate_file_hash(file):
    """Calculate a hash for the uploaded file to detect changes."""
    file.seek(0)
    file_bytes = file.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    file.seek(0)
    return file_hash

@st.cache_data
def extract_text_and_tables_from_pdf(file):
    """Extract text and tables from a PDF file."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages):
            st.write(f"Processing page {i + 1}...")
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text, []

@st.cache_data
def split_text_into_chunks(text, max_chunk_size=1000):
    """Split text into manageable chunks with progress feedback."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    chunks = []
    current_chunk = ""
    total_sentences = len(sentences)
    for i, sentence in enumerate(sentences):
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        if i % 100 == 0:
            st.write(f"Chunking progress: {i}/{total_sentences} sentences processed.")
    if current_chunk:
        chunks.append(current_chunk.strip())
    st.write(f"Total chunks created: {len(chunks)}")
    return chunks

# @st.cache_data
# def build_faiss_index(embedding_model, chunks):
#     """Build a FAISS index for the text chunks."""
#     embeddings = embedding_model.encode(chunks, show_progress_bar=True)
#     embeddings = np.array(embeddings).astype('float32')
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(embeddings)
#     return index, embeddings

@st.cache_data
def build_faiss_index(_embedding_model, chunks):
    """Build a FAISS index for the text chunks."""
    embeddings = _embedding_model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

def cohere_summarize(text, model, max_length=300):
    """Summarize text using Cohere LLM from Hugging Face."""
    prompt = f"Summarize the following text in {max_length} words:\n{text}\n"
    response = model(prompt, max_length=max_length, num_return_sequences=1, do_sample=False)
    return response[0]['generated_text'].strip()

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

    model = load_cohere_model()

    if summarize_type == "Full Document":
        try:
            pdf_text, _ = extract_text_and_tables_from_pdf(uploaded_file)
            with st.spinner("Summarizing the document..."):
                summary = cohere_summarize(pdf_text, model)
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
                    with st.spinner("Summarizing the section..."):
                        summary = cohere_summarize(section_text, model)
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
                    with st.spinner("Summarizing..."):
                        summary = cohere_summarize(user_text, model)
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
