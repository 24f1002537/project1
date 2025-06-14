import argparse
import base64
import json
import os
import numpy as np
import re
from pathlib import Path
from fastapi import FastAPI,Request
from pydantic import BaseModel
import httpx
import google.generativeai as genai

import uvicorn
from PIL import Image

app = FastAPI()

class questionrequest(BaseModel):
    question: str
    image: str

def get_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        result = genai.embed_content(
            model="models/embedding-001",  # or your model name
            content=chunk
        )
        embeddings.append(result['embedding'])
    return embeddings
client = genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

def get_image_description(image_path):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise Exception("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")  # Use correct model name

    img_file = Image.open(image_path)

    prompt = "Detailed description of the image, including any text in the image, in markdown format."
    response = model.generate_content([img_file, prompt])
    return response.text

def load_embeddings(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    data = np.load(file_path, allow_pickle=True)
    return data['chunks'], data['embeddings']

def generate_llm_response(question: str, context: str = None):
    """Generate response using Gemini"""
    generation_config = {
        "max_output_tokens": 1000,
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 40
    }

    prompt = f"""Answer the question based on the context provided.
    
    Question: {question}
    
    Context: {context or 'No context provided'}.
    
    Provide a concise and accurate answer in Markdown format."""

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(
        prompt,
        generation_config=generation_config
    )
    return response.text




def answer(question: str, image: str):
    # Initialize the Generative AI client
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    # Get image description
    if image:
        image_description = get_image_description(f"data:image/jpeg;base64,{image}")
        question += f"{image_description}"

    question_embedding = get_embeddings(question)
    # Load embeddings
    chunks, embeddings = load_embeddings("embeddings.npz")
    # Find the most relevant chunk
    similarities = np.dot(embeddings, question_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(question_embedding))
    #get the index of 10 most similar chunks
    top_indices = np.argsort(similarities)[-10:][::-1]
    # get the top chunks
    top_chunks = [chunks[i] for i in top_indices]
    response = generate_llm_response(question, context="\n\n".join(top_chunks))

    return {
        "candidates": [
            {
                "content": response,
                "metadata": {
                    "source": "discourse",
                    "chunks": top_chunks
                }
            }
        ]
    }
    

@app.post("/api/")
async def generate_answer(request: Request):
    try:
        data = await request.json()
        # Generate content

        # Return the generated text
        return answer(data.get("question"), data.get("image"))

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)