import os
import json
import requests
from flask import Flask, request, jsonify
from transformers import pipeline
from deep_translator import GoogleTranslator

app = Flask(__name__)

# Load Hugging Face model (disease detection)
model_repo = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
hf_token = os.getenv("HF_TOKEN")  # stored in Render, not GitHub
image_classifier = pipeline("image-classification", model=model_repo, use_auth_token=hf_token)

# Translator
translator = GoogleTranslator(source="auto", target="en")

# Load remediation data
with open("remediation.json", "r", encoding="utf-8") as f:
    remediation_data = json.load(f)

@app.route("/")
def home():
    return jsonify({"message": "Crop AI API running!"})

# -------------------- PREDICT --------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    lang = data.get("lang", "en")  # target language
    plant = data.get("plant", "")
    symptoms = data.get("symptoms", "")
    image_url = data.get("image_url", "")

    disease_name = None

    # If image provided → classify with HF model
    if image_url:
        try:
            result = image_classifier(image_url)
            disease_name = result[0]["label"]
        except Exception as e:
            return jsonify({"error": f"Image classification failed: {str(e)}"}), 500

    # If manual symptoms provided
    elif symptoms:
        # Simple keyword match (later can be upgraded to NLP model)
        for disease in remediation_data:
            if any(symptom in symptoms.lower() for symptom in remediation_data[disease]["symptoms"]):
                disease_name = disease
                break

    if not disease_name:
        return jsonify({"message": "Disease not recognized"}), 404

    # Get remediation
    remediation = remediation_data.get(disease_name, {}).get("remediation", "No info available")

    # Translate if needed
    if lang != "en":
        disease_name = translator.translate(disease_name, dest=lang).text
        remediation = translator.translate(remediation, dest=lang).text

    return jsonify({
        "plant": plant,
        "disease": disease_name,
        "remediation": remediation
    })

# -------------------- WEATHER --------------------
@app.route("/weather", methods=["GET"])
def weather():
    city = request.args.get("city")
    if not city:
        return jsonify({"error": "City is required"}), 400

    api_key = os.getenv("WEATHER_API_KEY")  # stored in Render
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        res = requests.get(url).json()
        if res.get("cod") != 200:
            return jsonify({"error": res.get("message", "Weather API error")}), 400
        return jsonify({
            "city": city,
            "temperature": res["main"]["temp"],
            "description": res["weather"][0]["description"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------- CHAT --------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query", "")
    # Placeholder response
    return jsonify({"response": f"You said: {query}. Chatbot coming soon!"})

# -------------------- ADMIN --------------------
@app.route("/admin", methods=["GET"])
def admin():
    token = request.headers.get("Authorization")
    if token != os.getenv("ADMIN_TOKEN"):
        return jsonify({"error": "Unauthorized"}), 403
    return jsonify({"message": "Welcome Admin!", "team": ["Yashraj", "Teammate1", "Teammate2"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
