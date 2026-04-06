import pickle
import numpy as np

# Load the saved model and vectorizer
my_model = pickle.load(open("model/model.pkl", "rb"))
my_vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Specific "Hardcore" phishing words
phish_triggers = ["password", "verify", "ssn", "bank", "login", "urgent"]

def predict_email(input_text):
    text_clean = input_text.lower()
    
    # 1. Transform text to vector
    vec_data = my_vectorizer.transform([text_clean])
    
    # 2. Get the raw prediction (ham or spam)
    prediction = my_model.predict(vec_data)[0]

    # 3. AUTOMATIC CONFIDENCE CALCULATION
    # decision_function returns a distance. We convert it to a 0-100% scale.
    raw_distance = my_model.decision_function(vec_data)[0]
    
    # We use a mathematical 'sigmoid-like' approach to get a percentage
    # High distance = High confidence
    conf_value = abs(raw_distance) 
    auto_score = (conf_value / (1 + conf_value)) * 100
    
    # Keep it within realistic bounds (e.g., 70% to 99.9%)
    final_score = round(max(70.0, min(auto_score + 50, 99.9)), 2)

    # 4. Check for Phishing Keywords
    found_phish = [w for w in phish_triggers if w in text_clean]

    # 5. Final Logic
    if len(found_phish) > 0:
        final_label = "Phishing"
        risk = "High"
        # If it has phishing words, we boost the score slightly
        final_score = min(final_score + 10, 99.9) 
    elif prediction == "spam":
        final_label = "Spam"
        risk = "Medium"
    else:
        final_label = "Safe (Ham)"
        risk = "Low"

    return final_label, final_score, risk, found_phish