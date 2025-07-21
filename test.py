import requests

API_KEY = "AIzaSyAApy_FVLfI1k03gn09gkPbN1rBU1b5rGc"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"

response = requests.get(GEMINI_API_URL)
print("Status code:", response.status_code)
print("Response:", response.text)