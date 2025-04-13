from flask import Flask, request, jsonify, send_file, render_template
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
from mha import fuse_sequences
from config import *
from threading import Lock
matplotlib_lock = Lock()


app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": ["http://localhost:3000","https://signolingo.vercel.app"]}})


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/create-videos', methods=['POST'])
def create_video():
    data = request.get_json()
    gloss = data.get('gloss')
    
    if not gloss:
        return jsonify({"error": "Gloss is required"}), 400
    
    try:
        with matplotlib_lock:
            fused_sequence = fuse_sequences(gloss)
        return jsonify({"message": "Videos created successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/cvae-video', methods=['GET'])
def get_cvae_video():
    file_path = CVAE_OUTPUT_VIDEO
    
    if not os.path.exists(file_path):
        logger.error(f"Video not found: {file_path}")
        return jsonify({"error": "Video not found"}), 404
    
    return send_file(file_path, mimetype='video/mp4')

@app.route('/cgan-video', methods=['GET'])
def get_cgan_video():
    file_path = CGAN_OUTPUT_VIDEO
    
    if not os.path.exists(file_path):
        logger.error(f"Video not found: {file_path}")
        return jsonify({"error": "Video not found"}), 404
    
    return send_file(file_path, mimetype='video/mp4')

@app.route('/fuse-video', methods=['GET'])
def get_fused_video():
    file_path = MHA_OUTPUT_VIDEO
    
    if not os.path.exists(file_path):
        logger.error(f"Video not found: {file_path}")
        return jsonify({"error": "Video not found"}), 404
    
    return send_file(file_path, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(debug=True )
