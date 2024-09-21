from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

model_name = "sshleifer/distilbart-cnn-12-6"
model_revision = "a4f8f3e"

summarizer = pipeline("summarization", model=model_name, revision=model_revision)

app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'Please enter the text'}), 400
    summary = summarizer(text, max_length=80, min_length=30, do_sample=False)
    return jsonify({'summary': summary[0]['summary_text']})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)