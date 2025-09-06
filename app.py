import os
import json
import requests
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from transformers import pipeline
from deep_translator import GoogleTranslator

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for session management

# Hardcoded login credentials
ADMIN_USERNAME = "Admin"
ADMIN_PASSWORD = "Backspace_Is_Great"

# Load Hugging Face model (disease detection)
model_repo = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
hf_token = os.getenv("HF_TOKEN")  # stored in Render, not GitHub
image_classifier = pipeline("image-classification", model=model_repo)

# Translator
translator = GoogleTranslator(source="auto", target="en")

# Path to remediation file
REMEDIATION_FILE = "remediation.json"

def load_remediation():
    with open(REMEDIATION_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_remediation(data):
    with open(REMEDIATION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# -------------------- ROUTES --------------------

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

    remediation_data = load_remediation()
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
        for disease in remediation_data:
            if any(symptom in symptoms.lower() for symptom in remediation_data[disease]["symptoms"]):
                disease_name = disease
                break

    if not disease_name:
        return jsonify({"message": "Disease not recognized"}), 404

    remediation = remediation_data.get(disease_name, {}).get("remediation", "No info available")

    # Translate if needed
    if lang != "en":
        disease_name = translator.translate(disease_name)
        remediation = translator.translate(remediation)

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
    return jsonify({"response": f"You said: {query}. Chatbot coming soon!"})

# -------------------- LOGIN --------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("admin"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("login"))

# -------------------- ADMIN PANEL --------------------
@app.route("/admin", methods=["GET", "POST"])
def admin():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    remediation_data = load_remediation()

    if request.method == "POST":
        updated_data = request.form.get("remediation_json")
        try:
            parsed = json.loads(updated_data)
            save_remediation(parsed)
            return render_template("dashboard.html", remediation_json=json.dumps(parsed, indent=4), success="File updated successfully")
        except Exception as e:
            return render_template("dashboard.html", remediation_json=json.dumps(remediation_data, indent=4), error=f"Invalid JSON: {e}")

    return render_template("dashboard.html", remediation_json=json.dumps(remediation_data, indent=4))

# -------------------- RUN --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
