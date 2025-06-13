# Placeholder for video_routes.py
from flask import Blueprint, request, jsonify
from app.services.video_service import process_video
import tempfile

video_bp = Blueprint("video", __name__)

@video_bp.route("/predict", methods=["POST"])
def predict_video():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No video file provided"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        file.save(temp_file.name)
        result = process_video(temp_file.name)

    return jsonify(result)
