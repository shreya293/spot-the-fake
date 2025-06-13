# Placeholder for run.py
from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)
    from app.routes.text_routes import text_bp
    from app.routes.image_routes import image_bp
    from app.routes.face_routes import face_bp
    from app.routes.video_routes import video_bp

    app.register_blueprint(text_bp, url_prefix="/api/text")
    app.register_blueprint(image_bp, url_prefix="/api/image")
    app.register_blueprint(face_bp, url_prefix="/api/face")
    app.register_blueprint(video_bp, url_prefix="/api/video")

    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
