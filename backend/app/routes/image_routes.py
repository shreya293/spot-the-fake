# Placeholder for image_routes.py
from flask import Blueprint, request, jsonify
from app.services.image_service import predict_image_label
import tempfile

image_bp = Blueprint("image", __name__)

@image_bp.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No image file uploaded"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        file.save(temp.name)
        label, confidence = predict_image_label(temp.name)

    return jsonify({
        "label": label,
        "confidence": confidence
    })
