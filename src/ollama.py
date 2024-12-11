import requests
import json  # Use this for JSON parsing
import streamlit as st
import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_file):
    """Extract text from an uploaded PDF file."""
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text


def query_ollama_api_stream(model, prompt):
    """Query the Ollama API and handle streaming responses."""
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "options": {"temperature": 0.7}
    }

    # Make a POST request to stream the response
    response = requests.post(url, headers=headers, json=data, stream=True)

    if response.status_code == 200:
        complete_response = ""  # Collect the full response
        try:
            for line in response.iter_lines():
                if line.strip():  # Skip empty lines
                    json_data = line.decode('utf-8')
                    parsed = json.loads(json_data)  # Safely parse JSON
                    complete_response += parsed.get("response", "")  # Append the "response" part
                    if parsed.get("done", False):  # Check if "done" is True
                        break
        except Exception as e:
            return f"Error processing stream: {e}"
        return complete_response.strip()  # Return the full response
    else:
        return f"Error: API returned status code {response.status_code}, content: {response.text}"


# Streamlit app layout
st.title("Enhanced PDF Question-Answering App with Ollama")
st.write("Upload a PDF, and ask questions about its content. The app processes streaming responses for better handling.")

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
        with st.spinner("Generating an answer..."):
            prompt = f"Context:\n{pdf_text}\n\nQuestion: {question}\nAnswer:"
            answer = query_ollama_api_stream("llama3.2", prompt)
        st.write("Answer:")
        st.write(answer)