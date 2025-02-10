from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
import os
from constants import CHROMA_SETTINGS

persist_directory = "db"

def main():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PyPDFLoader(os.path.join(root, file))
    documents = loader.load()
    print("splitting into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    #create embeddings here
    print("Loading sentence transformers model")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    # Initialize ChromaDB client
    client = chromadb.Client(CHROMA_SETTINGS)

    # Check if collection exists
    try:
        collection = client.get_collection(name="pdf_collection")
    except chromadb.errors.CollectionNotFoundError:
        # Create collection if it does not exist
        collection = client.create_collection(
            name="pdf_collection",
            embedding_function=sentence_transformer_ef,
        )

    #create vector store here
    print(f"Creating embeddings. May take some minutes...")

    # Process documents in batches
    documents_texts = [doc.page_content for doc in texts]
    documents_metadata = [{"source": doc.metadata.get("source", ""), "page": doc.metadata.get("page", 0)} for doc in texts]

    # Add documents to collection
    collection.add(
        documents=documents_texts,
        metadatas=documents_metadata,
        ids=[f"doc_{i}" for i in range(len(texts))]
    )

    print(f"Ingestion complete! You can now run query.py to query your documents")

if __name__ == "__main__":
    main()