from flask import Flask, request, jsonify, send_from_directory
import whisper
import ssl
import os
import cv2
import numpy as np
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

ssl._create_default_https_context = ssl._create_unverified_context

whisper_model = whisper.load_model("base")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

KEYFRAME_DIR = os.path.join(os.getcwd(), "static", "keyframes")
os.makedirs(KEYFRAME_DIR, exist_ok=True) 

@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Handles audio transcription using Whisper."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    audio_file = request.files["file"]
    audio_path = f"/tmp/{audio_file.filename}"
    audio_file.save(audio_path) 

    try:
        result = whisper_model.transcribe(audio_path)
        transcription = result["text"]
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(audio_path)  

    return jsonify({"transcription": transcription})


@app.route("/summarize", methods=["POST"])
def summarize():
    """Handles text summarization."""
    data = request.get_json()
    
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    try:
        summary = summarizer(data["text"], max_length=100, min_length=50, do_sample=False)
        if not summary or "summary_text" not in summary[0]:
            return jsonify({"error": "Summarization failed"}), 500

        summary_text = summary[0]["summary_text"]

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"summary": summary_text})


@app.route("/keyframes", methods=["POST"])
def extract_keyframes():
    """Extracts keyframes from a video and serves them via URLs."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    video_file = request.files["file"]
    video_path = os.path.join(KEYFRAME_DIR, video_file.filename)
    video_file.save(video_path)  

    try:
        cap = cv2.VideoCapture(video_path)
        success, prev_frame = cap.read()
        keyframe_urls = []
        frame_count = 0
        frame_interval = 30 
        threshold = 15_000_000

        while success:
            success, curr_frame = cap.read()
            if not success:
                break

            if frame_count % frame_interval == 0:  
                diff = cv2.absdiff(prev_frame, curr_frame)
                diff_sum = np.sum(diff)

                if diff_sum > threshold:
                    frame_filename = f"keyframe_{frame_count}.jpg"
                    frame_path = os.path.join(KEYFRAME_DIR, frame_filename)
                    cv2.imwrite(frame_path, curr_frame)
                    keyframe_urls.append(f"/static/keyframes/{frame_filename}")

                prev_frame = curr_frame

            frame_count += 1

        cap.release()
        os.remove(video_path) 

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"keyframes": keyframe_urls})


@app.route("/static/keyframes/<filename>")
def get_keyframe(filename):
    """Serves keyframe images."""
    return send_from_directory(KEYFRAME_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))