import streamlit as st
from transformers import pipeline
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import hashlib
import pandas as pd
import spacy

# Load the spaCy model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
from nltk.tokenize import sent_tokenize

# Increase the maximum length limit
nlp.max_length = 10_000_000  # Set this to a value greater than your text length

# Add the 'sentencizer' component
nlp.add_pipe('sentencizer')

# Set Streamlit page configuration
st.set_page_config(page_title="Enhanced PDF QA App", layout="wide")
st.title("📄 Enhanced PDF Question Answering App with Table Support")

# Sidebar for PDF upload
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
    """Load the extractive QA and embedding models."""
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # For embeddings
    return qa_model, embedding_model

# @st.cache_resource
# def load_paraphrasing_model():
#     """Load the paraphrasing model."""
#     paraphrase_model = pipeline("text2text-generation", model="t5-base")
#     return paraphrase_model

# @st.cache_resource
# def load_paraphrasing_model():
#     """Load a paraphrasing model fine-tuned for paraphrasing tasks."""
#     paraphrase_model = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")
#     return paraphrase_model

@st.cache_resource
def load_paraphrasing_model():
    """Load the FLAN-T5 model for paraphrasing."""
    paraphrase_model = pipeline("text2text-generation", model="google/flan-t5-large")
    return paraphrase_model

# @st.cache_resource
# def load_paraphrasing_model():
#     """Load a pre-trained summarization model."""
#     return pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_and_tables_from_pdf(file):
    """Extracts text and tables from a PDF file using pdfplumber."""
    text = ""
    tables = []
    with pdfplumber.open(file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            # Extract tables
            page_tables = page.extract_tables()
            for table in page_tables:
                if table:
                    try:
                        header = table[0]
                        data_rows = table[1:]

                        # Check for empty or duplicate column names
                        invalid_header = (
                            any(col == '' or col is None for col in header) or
                            len(set(header)) != len(header)
                        )
                        if invalid_header:
                            num_cols = len(header)
                            header = [f"Column {i+1}" for i in range(num_cols)]
                        df = pd.DataFrame(data_rows, columns=header)
                        tables.append(df)
                    except Exception as e:
                        st.warning(f"Failed to process a table on page {page_num + 1}: {e}")
        return text, tables

def split_text_into_chunks(text, max_chunk_size=300):
    """Splits text into chunks using spaCy's 'sentencizer' for sentence tokenization."""
    if not isinstance(text, str) or not text:
        st.error("The extracted text is not a valid string.")
        return []

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    # Chunk the sentences
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
    """Processes tables into text format suitable for QA."""
    tables_text = ""
    for idx, table in enumerate(tables):
        tables_text += f"\n\nTable {idx + 1}:\n"
        tables_text += table.to_string(index=False)
    return tables_text

def build_faiss_index(embedding_model, chunks):
    """Builds a FAISS index for the text chunks."""
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

def calculate_file_hash(file):
    """Calculates a hash for the uploaded file to detect changes."""
    file.seek(0)  # Ensure we're at the start of the file
    file_bytes = file.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    file.seek(0)  # Reset file pointer after reading
    return file_hash

# Check if a file is uploaded
if uploaded_file is not None:
    current_file_hash = calculate_file_hash(uploaded_file)

    # If the uploaded file is different from the previous one, reset session state
    if st.session_state.file_hash != current_file_hash:
        # Reset session state variables
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

        # Process tables and append to text
        if st.session_state.tables:
            with st.spinner("Processing tables..."):
                tables_text = process_tables(st.session_state.tables)
                st.session_state.pdf_text += tables_text
            st.success(f"Extracted and processed {len(st.session_state.tables)} tables.")

        # Split text into chunks
        with st.spinner("Splitting text into chunks..."):
            chunks = split_text_into_chunks(st.session_state.pdf_text)
            st.session_state.chunks = chunks
        st.success(f"Text split into {len(chunks)} chunks.")

        # Load models and build embeddings
        with st.spinner("Generating embeddings and building search index..."):
            qa_model, embedding_model = load_models()
            index, embeddings = build_faiss_index(embedding_model, st.session_state.chunks)
            st.session_state.index = index
            st.session_state.embeddings = embeddings
        st.success("Models loaded and search index built successfully!")

    # Display extracted text (optional)
    with st.expander("Show Extracted Text"):
        st.text_area("Extracted Text", st.session_state.pdf_text, height=300)

    # Display extracted tables (optional)
    if st.session_state.tables:
        with st.expander("Show Extracted Tables"):
            for idx, table in enumerate(st.session_state.tables):
                st.subheader(f"Table {idx + 1}")
                st.dataframe(table)

   # Question input
st.header("Ask a Question about the PDF")
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing your question..."):
            # Load models
            qa_model, embedding_model = load_models()
            paraphrase_model = load_paraphrasing_model()
            ner = spacy.load('en_core_web_sm')  # For NER checks
            
            # Encode the question
            question_embedding = embedding_model.encode([question], convert_to_tensor=True)
            question_embedding = question_embedding.cpu().numpy().astype('float32')

            # Search for top relevant chunks
            D, I = st.session_state.index.search(question_embedding, k=5)
            relevant_chunks = [
                st.session_state.chunks[idx]
                for idx in I[0] if idx < len(st.session_state.chunks)
            ]

            # Initialize variables to store the best answer
            best_answer = None
            best_score = 0

            # Find the best answer from the relevant chunks
            for chunk in relevant_chunks:
                try:
                    result = qa_model(question=question, context=chunk)
                    if result['score'] > best_score:
                        best_answer = result['answer']
                        best_score = result['score']
                except Exception as e:
                    st.error(f"Error processing chunk: {e}")

            # Check if an answer was found
            if best_answer:
                # Paraphrase the extracted answer with safeguards
                with st.spinner("Generating a detailed, human-like response..."):
                    paraphrase_input = (
                        f"Paraphrase the following answer to make it more detailed and conversational, "
                        f"without adding any new information. Ensure that the meaning remains the same.\n\n"
                        f"Answer: {best_answer}"
                    )
                    paraphrased_output = paraphrase_model(
                        paraphrase_input,
                        max_length=150,
                        num_beams=5,
                        early_stopping=True,
                        temperature=0.7,
                        top_p=0.9,
                        no_repeat_ngram_size=2,
                        do_sample=True
                    )
                    final_answer = paraphrased_output[0]['generated_text']

                # Perform similarity check
                original_embedding = embedding_model.encode(best_answer, convert_to_tensor=True)
                paraphrased_embedding = embedding_model.encode(final_answer, convert_to_tensor=True)

                # Compute similarity score
                similarity_score = util.pytorch_cos_sim(original_embedding, paraphrased_embedding).item()
                st.write(f"Similarity score between original and paraphrased answer: {similarity_score}")

                # Perform NER check
                original_entities = {ent.text for ent in ner(best_answer).ents}
                paraphrased_entities = {ent.text for ent in ner(final_answer).ents}
                st.write(f"Original entities: {original_entities}")
                st.write(f"Paraphrased entities: {paraphrased_entities}")

                similarity_threshold = 0.6
                if similarity_score >= similarity_threshold and paraphrased_entities.issubset(original_entities):
                    st.success("Here's the answer to your question:")
                    st.write(f"**Answer:** {final_answer}")
                    st.write(f"**Confidence Score:** {best_score:.2f}")
                else:
                    st.error("The paraphrased answer deviated from the original meaning. Please try again or rephrase your question.")
            else:
                st.error("Sorry, I couldn't find an answer to your question.")
else:
    st.info("Please upload a PDF file to get started.")