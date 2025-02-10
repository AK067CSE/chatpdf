import os 
import chromadb
from chromadb.config import Settings

CHROMA_SETTINGS = Settings(
    persist_directory='db',
    is_persistent=True,
    anonymized_telemetry=False
)
