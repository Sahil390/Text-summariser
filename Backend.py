import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
from transformers import pipeline


# Use Hugging Face summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


app = Flask(__name__, template_folder='templates')
@app.route('/')
def home():
    return render_template('frontend.html')
CORS(app)


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'Please enter the text'}), 400

    try:
        summary = summarizer(text, max_length=80, min_length=30, do_sample=False)
        return jsonify({'summary': summary[0]['summary_text']})
    except Exception as e:
        return jsonify({'error': 'Summarization failed', 'details': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8080)