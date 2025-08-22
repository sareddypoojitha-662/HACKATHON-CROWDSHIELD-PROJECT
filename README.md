# CrowdShield - AI-Powered People Counting System

Real-time people detection and counting using YOLOv8 and Flask web interface.

## Features

- Live webcam feed with person detection
- Real-time people counting
- Web-based dashboard
- YOLO v8 object detection

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python main.py
```

The application will automatically open your browser to `http://127.0.0.1:5001/evofluxcrowdshield`

## Routes

- `/evofluxcrowdshield` - Landing page
- `/dashboard` - Live dashboard
- `/video_feed` - Video stream
- `/data` - Person count API

## Requirements

- Python 3.7+
- Webcam
- Dependencies listed in requirements.txt