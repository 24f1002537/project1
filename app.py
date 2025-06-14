from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    image_url = None
    text1 = ''
    text2 = ''
    if request.method == 'POST':
        # Get texts
        text1 = request.form.get('text1')
        text2 = request.form.get('text2')
        # Handle image
        image = request.files.get('image')
        if image and image.filename:
            filename = secure_filename(image.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)
            image_url = url_for('static', filename=f'uploads/{filename}')
        return render_template('index.html', image_url=image_url, text1=text1, text2=text2)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
