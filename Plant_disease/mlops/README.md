# Plant Disease Detection MLOps System

This repository contains a complete MLOps pipeline for plant disease detection using YOLOv8 with experiment tracking, monitoring, and automated recommendations.

## Overview

The Plant Disease Detection system provides an end-to-end solution for identifying plant diseases in images and video streams using computer vision. The system incorporates MLOps best practices including experiment tracking, model versioning, performance monitoring, and automated analysis.

## Features

- 🌱 **Disease Detection**: Identify plant diseases in images, videos, or webcam streams
- 📊 **MLflow Integration**: Track experiments, model performance, and metrics
- 📈 **Real-time Monitoring**: Track FPS, detection counts, and confidence scores
- 🤖 **Automated Recommendations**: Save detected diseases to recommendation files
- 📱 **Multiple Input Sources**: Support for images, videos, and webcam feeds
- 🔄 **Modular Design**: Easily extendable for different models and use cases

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch
- OpenCV
- MLflow
- Ultralytics YOLOv8

## Usage

### Basic Usage

```bash
# Run with default settings (uses webcam)
python detector.py

# Process a single image
python detector.py --image path/to/image.jpg

# Process a video
python detector.py --video path/to/video.mp4

# Run with webcam (explicitly)
python detector.py --webcam
```

### Advanced Options

```bash
# Use a specific model
python detector.py --model path/to/model.pt

# Use a custom configuration file
python detector.py --config custom_config.yaml

# Don't display results (headless mode)
python detector.py --image path/to/image.jpg --no-display

# Export metrics from all runs
python detector.py --export-metrics

# Disable saving to recommendation files
python detector.py --no-save-recommendation
```

## MLflow Server

The system includes an MLflow server for experiment tracking.

### Starting the MLflow Server

```bash
# Start the MLflow server with default settings
python mlflow_setup.py

# Start with custom host and port
python mlflow_setup.py --host 0.0.0.0 --port 8080

# Check existing experiments
python mlflow_setup.py --check-experiments
```

The MLflow UI will be available at: http://localhost:5000 (or your custom host/port)

## Configuration

The system uses a YAML configuration file (`config.yaml`) with the following structure:

```yaml
model:
  path: best.pt
  confidence_threshold: 0.25
  version: v1.0
output:
  directory: results
  save_detections: true
inference:
  batch_size: 1
  image_size: 640
```

If no configuration file is found, a default one will be created.

## Project Structure

```
.
├── detector.py          # Main detection script
├── mlflow_setup.py      # MLflow server setup utilities
├── config.yaml          # Configuration file
├── models/              # Model files
├── mlruns/              # MLflow experiment tracking
├── mlflow-artifacts/    # MLflow artifacts
├── results/             # Detection results
├── logs/                # Application logs
└── Plant Disease Recommendation/ # Recommendation output directory
```

## Class: PlantDiseaseDetector

The main functionality is provided by the `PlantDiseaseDetector` class:

### Methods

- `__init__(config_path, experiment_name)`: Initialize the detector
- `load_model(model_path)`: Load a YOLOv8 model
- `predict(source, save_results, filename_prefix, run_name)`: Run inference
- `process_image(image_path, display, save_to_recommendation)`: Process a single image
- `process_video(video_path)`: Process a video file
- `run_webcam(save_to_recommendation)`: Run real-time detection using webcam
- `export_metrics(output_file)`: Export metrics from all runs

## Metrics Tracking

The system automatically tracks:

- Inference time
- Detection counts
- Confidence scores
- FPS (for video/webcam)
- Model loading time

## Development

The code follows MLOps best practices:

- Logging with timestamps and levels
- Error handling and validation
- Experiment tracking with MLflow
- Performance monitoring
- Configuration management

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.