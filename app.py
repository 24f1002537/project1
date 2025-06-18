from flask import Flask, json, request, render_template, jsonify
import os
import json
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import google.generativeai as genai
from flask_cors import CORS



app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        result = genai.embed_content(
            model="models/embedding-001",
            content=chunk
        )
        embeddings.append(result['embedding'])
    return np.array(embeddings)

def get_image_description(image_b64):
    if not image_b64.startswith("data:image/"):
        image_b64 = "data:image/jpeg;base64," + image_b64
    base64_data = image_b64.split(",", 1)[1]
    image_bytes = base64.b64decode(base64_data)
    img_file = Image.open(BytesIO(image_bytes))
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = "Detailed description of the image, including any text in the image, in markdown format."
    response = model.generate_content([img_file, prompt])
    return response.text

def load_embeddings(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    data = np.load(file_path, allow_pickle=True)
    return data['chunks'], data['embeddings']

def generate_llm_response(question: str, context: str = None):
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
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        prompt,
        generation_config=generation_config
    )
    return response.text

def answer(question: str,image_file=None):
    # If image is uploaded, convert to base64 and get description
    if image_file and image_file.filename:
        image_bytes = image_file.read()
        image_b64 = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("utf-8")
        image_description = get_image_description(image_b64)
        question += f"\n{image_description}"

    question_embedding = get_embeddings([question])[0]
    chunks, embeddings = load_embeddings("embeddings_compressed.npz")
    similarities = np.dot(embeddings, question_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(question_embedding)
    )
    top_indices = np.argsort(similarities)[-10:][::-1]
    top_chunks = [chunks[i] for i in top_indices]

    # Use provided context if available
    
    used_context = "\n\n".join(top_chunks)

    response = generate_llm_response(question, context=used_context)
    return response

@app.route("/api", methods=["GET","POST"])
def index():
    answer_text = None
    data = {}
    if request.method == "POST":
        question = request.form.get("question", "")
        image_file = request.files.get("image", None)
        try:
            answer_text = answer(question,image_file)
        except Exception as e:
            answer_text = f"Error: {e}"
        with open('/tmp/input.md', 'w', encoding='utf-8') as file:
            file.write(answer_text)
        input_filename = '/tmp/input.md'
        output_filename = '/tmp/output.txt'

        with open(input_filename, 'r') as file:
            content = file.read()

        # Remove all occurrences of '`' and 'json'
        content_cleaned = content.replace('`', '').replace('json', '')

        with open(output_filename, 'w', encoding='utf-8') as file:
            file.write(content_cleaned)
        data = json.loads(content_cleaned)
        return jsonify(data)
    return render_template("index.html", answer=answer_text)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
