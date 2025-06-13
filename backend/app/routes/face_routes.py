# Placeholder for face_routes.py
from flask import Blueprint, request, jsonify
from app.services.face_service import predict_face_label
import tempfile

face_bp = Blueprint("face", __name__)

@face_bp.route("/predict", methods=["POST"])
def predict_face():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No image file provided"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        file.save(temp_file.name)
        label, confidence = predict_face_label(temp_file.name)

    return jsonify({
        "label": label,
        "confidence": confidence
    })
