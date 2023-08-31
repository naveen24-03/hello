import streamlit as st
from PyPDF2 import PdfFileReader
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2, IndexIVFFlat

# Load the necessary models
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

# Function to read PDF and split into chunks
def read_pdf(file):
    pdf = PdfFileReader(file)
    text = ""
    for page in range(pdf.getNumPages()):
        text += pdf.getPage(page).extractText()
    chunks = text.split('.')  # Splitting into chunks at every period
    return chunks

# Function to embed chunks
def embed_chunks(chunks):
    embeddings = sentence_transformer.encode(chunks)
    return embeddings

# Function to create FAISS index
def create_faiss_index(embeddings):
    dimension = len(embeddings[0])  # Dimension of embeddings
    index = IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Function to search FAISS index
def search_faiss_index(query, index, chunks):
    query_embedding = sentence_transformer.encode([query])[0]
    D, I = index.search(query_embedding.reshape(1, -1), k=1)
    return chunks[I[0][0]]

# Main function
def main():
    st.set_page_config(page_title="Ask your PDF", page_icon=":books:")
    st.title("Ask your PDF")
    
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"], accept_multiple_files=False)
    
    if uploaded_file is not None:
        with st.spinner('Processing...'):
            chunks = read_pdf(uploaded_file)
            embeddings = embed_chunks(chunks)
            index = create_faiss_index(embeddings)

        query = st.text_input("Enter your query")
        if query:
            answer = search_faiss_index(query, index, chunks)
            st.write(answer)

if __name__ == "__main__":
    main()
