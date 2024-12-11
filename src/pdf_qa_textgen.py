import streamlit as st
import PyPDF2
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

def extract_text_from_pdf(pdf_file):
    """Extract text and organize by sections or paragraphs."""
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def find_relevant_context(text, question, tokenizer, model):
    """Search for the most relevant section of the text."""
    # Split text into paragraphs
    paragraphs = text.split("\n")
    
    # Rank paragraphs by relevance using the model
    scored_paragraphs = []
    for paragraph in paragraphs:
        if len(paragraph.strip()) > 0:
            inputs = tokenizer(
                f"Question: {question}\nContext: {paragraph}", 
                return_tensors="pt", 
                truncation=True
            )
            outputs = model.generate(
                inputs["input_ids"],
                max_length=50,
                pad_token_id=tokenizer.eos_token_id,
            )
            relevance_score = len(outputs[0])  # Approximate score by output length
            scored_paragraphs.append((paragraph, relevance_score))

    # Sort paragraphs by relevance score
    scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
    return " ".join([p[0] for p in scored_paragraphs[:3]])  # Top 3 paragraphs

def generate_answer_with_text_generation_model(context, question, model, tokenizer):
    """Use a text generation model to answer the question based on the context."""
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding="longest"
    )
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # Pass attention mask explicitly
        max_length=300,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,  # Explicitly set pad_token_id
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

@st.cache_resource
def load_model():
    """Load a larger and better model for better QA."""
    model_name = "EleutherAI/gpt-neo-1.3B"  # Use a larger model if system allows
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return model, tokenizer

# Load model
model, tokenizer = load_model()

# Streamlit app layout
st.title("Enhanced PDF Question-Answering App")
st.write("Upload a PDF, and ask questions about its content. The app finds the most relevant sections before answering.")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Extract text from the uploaded PDF
    with st.spinner("Extracting text from the PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
    st.success("Text extracted successfully!")

    # Show extracted text (optional)
    if st.checkbox("Show extracted text"):
        st.write(pdf_text)

    # Question input
    question = st.text_input("Enter your question:")

    if question:
        with st.spinner("Finding the most relevant context..."):
            context = find_relevant_context(pdf_text, question, tokenizer, model)

        if context:
            st.write("Relevant Context Found:")
            st.write(context)

            with st.spinner("Generating an answer..."):
                answer = generate_answer_with_text_generation_model(context, question, model, tokenizer)

            st.write("Answer:")
            st.write(answer)
        else:
            st.write("Sorry, no relevant context found!")