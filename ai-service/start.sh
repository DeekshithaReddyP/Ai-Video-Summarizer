#!/bin/bash
pip install -r requirements.txt
apt-get update && apt-get install -y ffmpeg  # Install FFmpeg
python app.py