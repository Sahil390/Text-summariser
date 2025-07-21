import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai


# Configure the Gemini API key
# The API key is read from an environment variable for security
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    # This will stop the application from starting if the key is not set
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=api_key)

# Create the Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')


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
        # Generate the summary using the Gemini API
        response = model.generate_content(f"""Summarize the following text:

{text}""")
        return jsonify({'summary': response.text})
    except Exception as e:
        import traceback
        print('Gemini API error:', e)
        traceback.print_exc()
        return jsonify({'error': 'Summarization failed', 'details': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8080)