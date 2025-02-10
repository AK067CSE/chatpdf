from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import chromadb
from chromadb.utils import embedding_functions
import torch
from constants import CHROMA_SETTINGS

# Initialize the LaMini-T5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-T5-738M")
model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-T5-738M")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Initialize ChromaDB and embeddings
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.Client(CHROMA_SETTINGS)
collection = client.get_collection(
    name="pdf_collection",
    embedding_function=sentence_transformer_ef
)

def generate_answer(query, context, max_length=512):
    # Prepare the input text
    input_text = f"Question: {query}\nContext: {context}\nAnswer:"
    
    # Tokenize and generate
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model.generate(
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

def answer_question(query):
    # Get relevant context from ChromaDB
    context = get_relevant_context(query)
    
    # Generate answer using LaMini-T5
    answer = generate_answer(query, context)
    
    return answer

if __name__ == "__main__":
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        try:
            answer = answer_question(query)
            print("\nAnswer:", answer)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
