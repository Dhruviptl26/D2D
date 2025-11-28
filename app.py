from flask import Flask, request, jsonify
import os
from services.yolo_services import run_yolo, model
from utils.json_formatter import yolo_to_json
from services.gemini_services import generate_code_from_json

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/generate", methods=["POST"])
def generate_ui_code():
    try:
        image = request.files["image"]
        selected_language = request.form["language"]

        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)

        # Step 1: YOLO Detection
        results = run_yolo(image_path)

        # Step 2: Convert YOLO Output → JSON
        detections_json = yolo_to_json(results, model)

        # Step 3: Send JSON to Gemini → Get Code
        generated_code = generate_code_from_json(detections_json, selected_language)

        # Step 4: Return Generated Code
        return jsonify({
            "status": "success",
            "language": selected_language,
            "generated_code": generated_code
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
