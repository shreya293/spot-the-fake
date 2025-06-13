# Placeholder for text_routes.py
from flask import Blueprint, request, jsonify
from app.services.text_service import predict_text_label

text_bp = Blueprint("text", __name__)

@text_bp.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    label, confidence = predict_text_label(text)
    return jsonify({
        "label": label,
        "confidence": confidence
    })
