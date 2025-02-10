import streamlit as st
import os
import base64
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from constants import CHROMA_SETTINGS
from streamlit_chat import message
import random

# Set page config
st.set_page_config(layout="wide", page_title="Cosmic PDF Explorer", page_icon="🌌")

# Custom CSS for galaxy theme, glassmorphism, and card effects
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&display=swap');
    
    body {
        background-image: url('https://images.unsplash.com/photo-1534796636912-3b95b3ab5986?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2071&q=80');
        background-size: cover;
        background-attachment: fixed;
        font-family: 'Orbitron', sans-serif;
    }
    
    .stApp {
        background-color: rgba(0, 0, 0, 0.7);
    }
    
    .cosmic-card {
        background-color: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .cosmic-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 7px 14px rgba(0, 0, 0, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
    }
    
    .upload-card {
        background: linear-gradient(45deg, rgba(123, 31, 162, 0.5), rgba(103, 58, 183, 0.5));
    }
    
    .preview-card {
        background: linear-gradient(45deg, rgba(32, 156, 238, 0.5), rgba(0, 176, 155, 0.5));
    }
    
    .chat-card {
        background: linear-gradient(45deg, rgba(255, 87, 34, 0.5), rgba(255, 152, 0, 0.5));
    }
    
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 15px;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #FF4081, #FF9100);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #FF9100, #FF4081);
        transform: scale(1.05);
    }
    
    h1, h2, h3, h4 {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(45deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    
    .cosmic-spinner {
        width: 50px;
        height: 50px;
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top-color: #FFD700;
        animation: cosmic-spin 1s ease-in-out infinite;
    }
    
    @keyframes cosmic-spin {
        to { transform: rotate(360deg); }
    }
    
    .star {
        position: fixed;
        width: 2px;
        height: 2px;
        background: white;
        border-radius: 50%;
        animation: twinkle 5s infinite;
    }
    
    @keyframes twinkle {
        0% { opacity: 0; }
        50% { opacity: 1; }
        100% { opacity: 0; }
    }
</style>
""", unsafe_allow_html=True)

# Add twinkling stars
for i in range(100):
    left = random.randint(0, 100)
    top = random.randint(0, 100)
    delay = random.uniform(0, 5)
    st.markdown(f"""
    <div class="star" style="left: {left}vw; top: {top}vh; animation-delay: {delay}s;"></div>
    """, unsafe_allow_html=True)

# Device setup
device = torch.device('cpu')

# Model setup
checkpoint = "MBZUAI/LaMini-T5-738M"
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
        collection = client.get_collection(name="pdf_collection", embedding_function=sentence_transformer_ef)
        client.delete_collection(name="pdf_collection")
    except Exception as e:
        print(f"Collection not found, creating new one: {str(e)}")
    
    collection = client.create_collection(name="pdf_collection", embedding_function=sentence_transformer_ef)
    
    documents_texts = [doc.page_content for doc in texts]
    documents_metadata = [{"source": doc.metadata.get("source", ""), "page": doc.metadata.get("page", 0)} for doc in texts]
    
    collection.add(
        documents=documents_texts,
        metadatas=documents_metadata,
        ids=[f"doc_{i}" for i in range(len(texts))]
    )

def generate_answer(query, context, max_length=512):
    input_text = f"Question: {query}\nContext: {context}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = base_model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def get_relevant_context(query, k=2):
    collection = client.get_collection(name="pdf_collection", embedding_function=sentence_transformer_ef)
    results = collection.query(query_texts=[query], n_results=k)
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
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user", avatar_style="adventurer")
        message(history["generated"][i], key=str(i), avatar_style="bottts")

def main():
    st.markdown("<h1>🌌 Cosmic PDF Explorer 📚</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Unravel the secrets of the universe, one PDF at a time 🚀</h3>", unsafe_allow_html=True)

    st.markdown("<h2>📤 Upload your Galactic Manuscript 🌠</h2>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="cosmic-card upload-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["pdf"])
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{get_file_size(uploaded_file) / 1024:.2f} KB"
        }
        filepath = os.path.join("docs", uploaded_file.name)
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        col1, col2 = st.columns([1,2])
        
        with col1:
            st.markdown('<div class="cosmic-card preview-card">', unsafe_allow_html=True)
            st.markdown("<h4>📄 Cosmic Manuscript Details</h4>", unsafe_allow_html=True)
            st.json(file_details)
            st.markdown("<h4>🔭 Galactic Lens</h4>", unsafe_allow_html=True)
            pdf_view = displayPDF(filepath)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="cosmic-card chat-card">', unsafe_allow_html=True)
            with st.spinner('Decoding the cosmic signals... 🌌'):
                st.markdown('<div class="cosmic-spinner"></div>', unsafe_allow_html=True)
                ingested_data = data_ingestion()
            st.success('Cosmic knowledge assimilated! 🎉')
            
            st.markdown("<h4>💬 Converse with the Cosmic AI</h4>", unsafe_allow_html=True)

            user_input = st.text_input("Ask the cosmic oracle...", key="input")

            if "generated" not in st.session_state:
                st.session_state["generated"] = ["Greetings, intergalactic explorer! How may I illuminate your cosmic journey?"]
            if "past" not in st.session_state:
                st.session_state["past"] = ["Salutations, Cosmic AI!"]

            if user_input:
                answer = process_answer({'query': user_input})
                st.session_state["past"].append(user_input)
                response = answer
                st.session_state["generated"].append(response)

            if st.session_state["generated"]:
                display_conversation(st.session_state)
            
            st.markdown('</div>', unsafe_allow_html=True)

    # Add a pulsating star effect
    st.markdown("""
    <style>
    @keyframes pulse {
        0% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.2); opacity: 1; }
        100% { transform: scale(1); opacity: 0.5; }
    }
    .pulsating-star {
        position: fixed;
        width: 4px;
        height: 4px;
        background: white;
        border-radius: 50%;
        animation: pulse 3s infinite;
    }
    </style>
    """, unsafe_allow_html=True)

    for i in range(20):
        left = random.randint(0, 100)
        top = random.randint(0, 100)
        delay = random.uniform(0, 3)
        st.markdown(f"""
        <div class="pulsating-star" style="left: {left}vw; top: {top}vh; animation-delay: {delay}s;"></div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()