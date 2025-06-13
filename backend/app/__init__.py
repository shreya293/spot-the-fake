# Placeholder for __init__.py
from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)

    from app.routes.face_routes import face_bp
    from app.routes.video_routes import video_bp

    app.register_blueprint(face_bp, url_prefix="/api/face")
    app.register_blueprint(video_bp, url_prefix="/api/video")

    return app
