import hashlib
from xmlrpc import client
import httpx
import json
import numpy as np
import os
import time
from pathlib import Path
from semantic_text_splitter import MarkdownSplitter
from tqdm import tqdm
import google.generativeai as genai


import os
from semantic_text_splitter import MarkdownSplitter
import google.generativeai as genai

def get_chunk(text, chunk_size=1000, overlap=100):
    with open(text, 'r', encoding='utf-8') as file:
        content = file.read()
    splitter = MarkdownSplitter(chunk_size, overlap=overlap)
    return splitter.chunks(content)

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def get_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        result = genai.embed_content(
            model="models/embedding-001",  # or your model name
            content=chunk
        )
        embeddings.append(result['embedding'])
    return embeddings

if __name__ == "__main__":
    all_chunks = []
    all_embeddings = []
    total_chunks = 0
    file_chunks = {}
    files = [*Path("discourse").glob("*.md"), *Path("tools-in-data-science-public-main").glob("*.md")]
    for file_path in files:
        chunk = get_chunk(file_path, chunk_size=1000, overlap=100)
        file_chunks[file_path] = chunk
        total_chunks += len(chunk)

    with tqdm(total=total_chunks, desc="Processing Chunks") as pbar:
        for file_path, chunks in file_chunks.items():
            for chunk in chunks:
                try:
                    embedding = get_embeddings([chunk])[0]
                    all_chunks.append(chunk)
                    all_embeddings.append(embedding)
                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing chunk from {file_path}: {e}")
                    pbar.update(1)
                    continue
    np.savez("embeddings.npz", chunks=all_chunks, embeddings=all_embeddings)

            