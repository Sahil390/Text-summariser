from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS  # Enable cross-origin requests

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend communication

# Load the text summarizer pipeline from Huggingface
summarizer = pipeline("summarization")

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.json  # Get the JSON data sent from the frontend
    text = data.get('text', '')  # Extract the 'text' field from JSON
    if not text:
        return jsonify({'error': 'No text provided'}), 400  # Handle error if no text is provided

    summary = summarizer(text, max_length=80, min_length=30, do_sample=False)  # Summarize the text
    return jsonify({'summary': summary[0]['summary_text']})  # Return summary to frontend

if __name__ == '__main__':
    app.run(debug=True)
