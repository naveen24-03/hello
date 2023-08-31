import streamlit as st
from PyPDF2 import PdfFileReader
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2

# Initialize the tokenizer and model with LLaMa-7b model from Hugging Face
model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
model_basename = "model"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)

"""
To download from a specific branch, use the revision parameter, as in this example:

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        revision="gptq-4bit-32g-actorder_True",
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
        quantize_config=None)
"""

# Initialize the sentence transformer with LLaMa-7b model from Hugging Face
sent_trans = SentenceTransformer(model_name_or_path)

# Initialize the FAISS index
index = IndexFlatL2(768)

@st.cache
def process_pdf(pdf_file):
    # Read the PDF file
    with open(pdf_file, "rb") as f:
        pdf = PdfFileReader(f)
        text = ""
        # Extract the text from each page
        for page in pdf.pages:
            text += page.extract_text()
    # Split the text into chunks
    chunks = text.split("\n\n")
    # Embed the chunks using the sentence transformer
    embeddings = sent_trans.encode(chunks)
    # Add the embeddings to the FAISS index
    index.add(embeddings)
    return chunks

def answer_question(question, chunks):
    # Tokenize the question
    input_ids = tokenizer.encode(question, return_tensors="pt")
    # Generate an answer using the model
    answer = model.generate(input_ids)
    answer_text = tokenizer.decode(answer[0], skip_special_tokens=True)
    # Embed the answer using the sentence transformer
    answer_embedding = sent_trans.encode([answer_text])[0]
    # Search for the most similar chunk in the FAISS index
    _, indices = index.search(answer_embedding.reshape(1, -1), 1)
    # Return the most similar chunk as the answer
    return chunks[indices[0][0]]

def main():
    st.set_page_config(page_title="Intelliread", page_icon=":books:", layout="wide")
    
    st.title("Intelliread 💬")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload your PDF")
        pdf_file = st.file_uploader("", type=["pdf"], accept_multiple_files=False, key="pdf")
    
    with col2:
        st.write("")
    
    if pdf_file is not None:
        chunks = process_pdf(pdf_file)
        question = st.text_input("Ask a question")
        if question:
            answer = answer_question(question, chunks)
            st.write(answer)

if __name__ == "__main__":
    main()
