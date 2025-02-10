import streamlit as st
import os
import base64
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import textwrap
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from constants import CHROMA_SETTINGS
from streamlit_chat import message

st.set_page_config(layout="wide")

device = torch.device('cpu')

checkpoint = "MBZUAI/LaMini-T5-738M"
print(f"Checkpoint path: {checkpoint}")  # Add this line for debugging
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(
    checkpoint,
    device_map=device,
    torch_dtype=torch.float32
)

# Initialize ChromaDB and embeddings
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.Client(CHROMA_SETTINGS)

@st.cache_resource
def data_ingestion():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    
    try:
        # Try to get existing collection
        collection = client.get_collection(
            name="pdf_collection",
            embedding_function=sentence_transformer_ef
        )
        # Delete existing collection to refresh the data
        client.delete_collection(name="pdf_collection")
    except Exception as e:
        print(f"Collection not found, creating new one: {str(e)}")
    
    # Create new collection
    collection = client.create_collection(
        name="pdf_collection",
        embedding_function=sentence_transformer_ef,
    )
    
    # Process documents in batches
    documents_texts = [doc.page_content for doc in texts]
    documents_metadata = [{"source": doc.metadata.get("source", ""), "page": doc.metadata.get("page", 0)} for doc in texts]
    
    # Add documents to collection
    collection.add(
        documents=documents_texts,
        metadatas=documents_metadata,
        ids=[f"doc_{i}" for i in range(len(texts))]
    )

def generate_answer(query, context, max_length=512):
    # Prepare the input text
    input_text = f"Question: {query}\nContext: {context}\nAnswer:"
    
    # Tokenize and generate
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = base_model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    
    # Decode and return the answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def get_relevant_context(query, k=2):
    # Get collection
    collection = client.get_collection(
        name="pdf_collection",
        embedding_function=sentence_transformer_ef
    )
    
    # Search for relevant documents
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    
    # Combine the content of relevant documents
    if results and results['documents']:
        context = " ".join(results['documents'][0])
    else:
        context = ""
    return context

def process_answer(instruction):
    if isinstance(instruction, dict):
        query = instruction.get('query', '')
    else:
        query = instruction
        
    context = get_relevant_context(query)
    answer = generate_answer(query, context)
    return answer

def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size

@st.cache_data
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))

def main():
    st.markdown("<h1 style='text-align: center; color: blue;'>Chat with your PDF 🦜📄 </h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: grey;'>Built by <a href='https://github.com/AIAnytime'>AI Anytime with ❤️ </a></h3>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color:red;'>Upload your PDF 👇</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["pdf"])

    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": get_file_size(uploaded_file)
        }
        filepath = "docs/"+uploaded_file.name
        with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

        col1, col2 = st.columns([1,2])
        with col1:
            st.markdown("<h4 style='color:black;'>File details</h4>", unsafe_allow_html=True)
            st.json(file_details)
            st.markdown("<h4 style='color:black;'>File preview</h4>", unsafe_allow_html=True)
            pdf_view = displayPDF(filepath)

        with col2:
            with st.spinner('Embeddings are in process...'):
                ingested_data = data_ingestion()
            st.success('Embeddings are created successfully!')
            st.markdown("<h4 style='color:black;'>Chat Here</h4>", unsafe_allow_html=True)

            user_input = st.text_input("", key="input")

            # Initialize session state for generated responses and past messages
            if "generated" not in st.session_state:
                st.session_state["generated"] = ["I am ready to help you"]
            if "past" not in st.session_state:
                st.session_state["past"] = ["Hey there!"]

            # Search the database for a response based on user input and update session state
            if user_input:
                answer = process_answer({'query': user_input})
                st.session_state["past"].append(user_input)
                response = answer
                st.session_state["generated"].append(response)

            # Display conversation history using Streamlit messages
            if st.session_state["generated"]:
                display_conversation(st.session_state)

if __name__ == "__main__":
    main()