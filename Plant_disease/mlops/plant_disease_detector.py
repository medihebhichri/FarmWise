
import os
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import yaml
import logging
import time
import json
from datetime import datetime
import mlflow
import torch
from typing import Dict, List, Optional, Tuple, Union
import shutil
from pathlib import Path


class PlantDiseaseDetector:
    """
    A class for plant disease detection using YOLOv8 with MLOps best practices.

    This class implements model versioning, logging, performance tracking,
    and inference pipeline management.
    """

    def __init__(
            self,
            config_path: str = "config.yaml",
            experiment_name: str = "plant_disease_detection"
    ):
        """
        Initialize the plant disease detector.

        Args:
            config_path (str): Path to the configuration file
            experiment_name (str): Name of the MLflow experiment
        """
        # Setup logging
        self._setup_logging()

        # Load configuration
        self.config = self._load_config(config_path)
        self.logger.info(f"Configuration loaded from {config_path}")

        # Initialize MLflow
        self._setup_mlflow(experiment_name)

        # Initialize model
        self.model = None
        self.class_names = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"plant_disease_detection_{timestamp}.log")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger("PlantDiseaseDetector")

    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.

        Args:
            config_path (str): Path to the configuration file

        Returns:
            Dict: Configuration dictionary
        """
        if not os.path.exists(config_path):
            self.logger.warning(f"Config file {config_path} not found. Using default configuration.")
            config = {
                "model": {
                    "path": "best.pt",
                    "confidence_threshold": 0.25,
                    "version": "v1.0"
                },
                "output": {
                    "directory": "results",
                    "save_detections": True
                },
                "inference": {
                    "batch_size": 1,
                    "image_size": 640
                }
            }

            # Save default config
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
        else:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

        return config

    def _setup_mlflow(self, experiment_name: str) -> None:
        """
        Set up MLflow for experiment tracking.

        Args:
            experiment_name (str): Name of the MLflow experiment
        """
        # Create mlruns directory if it doesn't exist
        os.makedirs("mlruns", exist_ok=True)

        # Set tracking URI to local directory
        mlflow.set_tracking_uri("file:./mlruns")

        # Set or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)

        mlflow.set_experiment(experiment_name)
        self.logger.info(f"MLflow experiment set to: {experiment_name}")

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load the YOLOv8 model.

        Args:
            model_path (str, optional): Path to the model file. If None, uses the path from config.
        """
        if model_path is None:
            model_path = self.config["model"]["path"]

        try:
            # Start tracking model load time
            start_time = time.time()

            # Load model
            self.model = YOLO(model_path)
            self.model.to(self.device)

            # Calculate load time
            load_time = time.time() - start_time

            # Get class names
            self.class_names = self.model.names

            self.logger.info(f"Model loaded successfully from {model_path} in {load_time:.2f} seconds")
            self.logger.info(f"Model version: {self.config['model']['version']}")
            self.logger.info(f"Available classes: {self.class_names}")

            # Log model metadata with MLflow
            with mlflow.start_run(run_name=f"model_load_{self.config['model']['version']}"):
                mlflow.log_param("model_path", model_path)
                mlflow.log_param("model_version", self.config['model']['version'])
                mlflow.log_param("device", str(self.device))
                mlflow.log_metric("model_load_time", load_time)

                # Log model classes
                mlflow.log_dict(self.class_names, "model_classes.json")

                # Log model as artifact
                mlflow.log_artifact(model_path)

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def predict(
            self,
            source: Union[str, np.ndarray],
            save_results: bool = True,
            filename_prefix: str = "",
            run_name: Optional[str] = None
    ) -> List:
        """
        Run inference on images or videos.

        Args:
            source (str or np.ndarray): Path to image/video or numpy array
            save_results (bool): Whether to save detection results
            filename_prefix (str): Prefix for saved files
            run_name (str, optional): Custom name for the MLflow run

        Returns:
            List: Results from the model prediction
        """
        if self.model is None:
            self.logger.error("Model not loaded. Call load_model() first.")
            raise ValueError("Model not loaded")

        conf_threshold = self.config["model"]["confidence_threshold"]
        output_dir = self.config["output"]["directory"]

        # Create results directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"inference_{timestamp}"

        # Start MLflow run for this inference
        with mlflow.start_run(run_name=run_name):
            # Log inference parameters
            mlflow.log_param("confidence_threshold", conf_threshold)

            if isinstance(source, str):
                mlflow.log_param("source_type", "file")
                mlflow.log_param("source_path", source)
            else:
                mlflow.log_param("source_type", "array")
                mlflow.log_param("array_shape", str(source.shape))

            # Measure inference time
            start_time = time.time()

            # Run prediction with tracking
            results = self.model.predict(
                source=source,
                conf=conf_threshold,
                save=save_results,
                project=output_dir,
                name=f"{filename_prefix}_{run_name}" if filename_prefix else run_name
            )

            # Calculate and log inference time
            inference_time = time.time() - start_time
            mlflow.log_metric("inference_time", inference_time)

            # Log inference metrics
            if results:
                detection_count = sum(len(r.boxes) for r in results)
                mlflow.log_metric("detection_count", detection_count)

                # Log class distribution
                class_distribution = {}
                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        class_name = self.class_names[cls]
                        class_distribution[class_name] = class_distribution.get(class_name, 0) + 1

                mlflow.log_dict(class_distribution, "class_distribution.json")

                # Log average confidence
                confidences = [float(box.conf[0]) for result in results for box in result.boxes]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    mlflow.log_metric("average_confidence", avg_confidence)

            # Log results path if saved
            if save_results:
                results_path = os.path.join(output_dir, run_name)
                mlflow.log_param("results_path", results_path)

                # Log result images as artifacts if they exist
                results_dir = Path(results_path)
                if results_dir.exists():
                    for img_path in results_dir.glob("*.jpg"):
                        mlflow.log_artifact(str(img_path))

            self.logger.info(
                f"Inference completed in {inference_time:.2f} seconds. Results saved to {output_dir}/{run_name}")

            return results

    def process_image(self, image_path: str, display: bool = True, save_to_recommendation: bool = True) -> np.ndarray:
        """
        Process a single image and optionally display results.

        Args:
            image_path (str): Path to the input image
            display (bool): Whether to display the image with detections
            save_to_recommendation (bool): Whether to save results to recommendation directory

        Returns:
            np.ndarray: Image with detection annotations
        """
        if not os.path.exists(image_path):
            self.logger.error(f"Image not found at {image_path}")
            raise FileNotFoundError(f"Image not found at {image_path}")

        self.logger.info(f"Processing image: {image_path}")

        # Run prediction
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = self.predict(
            source=image_path,
            filename_prefix="image",
            run_name=f"image_{Path(image_path).stem}_{timestamp}"
        )

        # Create annotated image
        img = cv2.imread(image_path)
        img_annotated = img.copy()

        # List to store detection results
        detection_results = []

        if results:
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Get class and confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Add to detection results list
                    detection_results.append({
                        "class": self.class_names[cls],
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                    })

                    # Draw bounding box
                    cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Add label
                    label = f"{self.class_names[cls]} {conf:.2f}"
                    cv2.putText(img_annotated, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save results to recommendation directory if requested
        if save_to_recommendation and detection_results:
            self._save_to_recommendation_file(detection_results, image_path)

        if display:
            cv2.imshow("Plant Disease Detection", img_annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return img_annotated

    def process_video(self, video_path: str) -> str:
        """
        Process a video file.

        Args:
            video_path (str): Path to the input video

        Returns:
            str: Path to the output video
        """
        if not os.path.exists(video_path):
            self.logger.error(f"Video not found at {video_path}")
            raise FileNotFoundError(f"Video not found at {video_path}")

        self.logger.info(f"Processing video: {video_path}")

        # Run prediction
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem
        run_name = f"video_{video_name}_{timestamp}"

        self.predict(
            source=video_path,
            filename_prefix="video",
            run_name=run_name
        )

        # Get path to result video
        output_dir = self.config["output"]["directory"]
        result_path = os.path.join(output_dir, run_name)

        self.logger.info(f"Video processing completed. Results saved to {result_path}")

        return result_path

    def _save_to_recommendation_file(self, detection_results: List[Dict], image_path: str = None) -> None:
        """
        Save detection results to the recommendation directory.

        Args:
            detection_results: List of detection result dictionaries
            image_path: Original image path (for reference)
        """
        # Only save the highest confidence detection
        if not detection_results:
            return

        # Sort by confidence and get the highest confidence detection
        best_detection = sorted(detection_results, key=lambda x: x['confidence'], reverse=True)[0]
        disease_name = best_detection['class']

        # Use the exact directory name as shown in the image
        # (note the space between "Disease" and "Recommendation")
        recommendation_dir = r"D:\cours\pfa\hardware\code\plantwise\Plant Disease Recommendation"

        # Check if directory exists instead of creating it
        if not os.path.exists(recommendation_dir):
            self.logger.warning(f"Directory does not exist: {recommendation_dir}")
            self.logger.warning("Using current directory instead")
            recommendation_dir = "."

        # Create file with just the disease name
        filepath = os.path.join(recommendation_dir, "disease.txt")

        self.logger.info(f"Saving detected disease to {filepath}")

        # Write only the disease name to the file
        with open(filepath, 'w') as f:
            f.write(f"{disease_name}")

        self.logger.info(f"Disease name saved successfully to file")

    def run_webcam(self, save_to_recommendation: bool = True) -> None:
        """
        Run real-time detection using webcam.

        Args:
            save_to_recommendation (bool): Whether to save results to recommendation directory
        """
        self.logger.info("Starting webcam for real-time detection")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.logger.error("Could not open webcam")
            raise RuntimeError("Could not open webcam")

        # Start MLflow run for webcam session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with mlflow.start_run(run_name=f"webcam_session_{timestamp}"):
            mlflow.log_param("mode", "webcam")
            mlflow.log_param("confidence_threshold", self.config["model"]["confidence_threshold"])

            fps_values = []
            detections_count = 0
            frame_count = 0
            start_time = time.time()
            last_save_time = 0
            save_interval = 5  # Save results every 5 seconds

            try:
                while True:
                    # Measure frame processing time
                    frame_start = time.time()

                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Perform detection
                    results = self.model.predict(
                        source=frame,
                        conf=self.config["model"]["confidence_threshold"],
                        verbose=False
                    )

                    # Process detections and save if needed
                    current_time = time.time()
                    if save_to_recommendation and results and (current_time - last_save_time) >= save_interval:
                        # Format detection results
                        detection_results = []
                        for result in results:
                            for box in result.boxes:
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                detection_results.append({
                                    "class": self.class_names[cls],
                                    "confidence": conf,
                                    "bbox": box.xyxy[0].tolist()
                                })

                        # Only save if we have detections with confidence > threshold
                        significant_detections = [d for d in detection_results if d["confidence"] > 0.5]
                        if significant_detections:
                            self._save_to_recommendation_file(significant_detections)
                            last_save_time = current_time

                    # Count detections
                    if results:
                        frame_detections = sum(len(r.boxes) for r in results)
                        detections_count += frame_detections

                    # Get the rendered frame with detections
                    annotated_frame = results[0].plot()

                    # Calculate FPS
                    frame_time = time.time() - frame_start
                    fps = 1 / max(frame_time, 0.001)  # Avoid division by zero
                    fps_values.append(fps)

                    # Display FPS on frame
                    cv2.putText(
                        annotated_frame,
                        f"FPS: {fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                    # Display the frame
                    cv2.imshow("Plant Disease Detection (Webcam)", annotated_frame)

                    # Count processed frames
                    frame_count += 1

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            finally:
                # Log webcam session metrics
                session_time = time.time() - start_time
                avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0

                mlflow.log_metric("session_duration", session_time)
                mlflow.log_metric("frames_processed", frame_count)
                mlflow.log_metric("average_fps", avg_fps)
                mlflow.log_metric("total_detections", detections_count)

                if frame_count > 0:
                    mlflow.log_metric("detections_per_frame", detections_count / frame_count)

                # Release resources
                cap.release()
                cv2.destroyAllWindows()

                self.logger.info(f"Webcam session ended after {session_time:.2f} seconds")
                self.logger.info(f"Processed {frame_count} frames at avg {avg_fps:.2f} FPS")
                self.logger.info(f"Detected {detections_count} objects")

    def export_metrics(self, output_file: str = "metrics.json") -> Dict:
        """
        Export collected metrics from all runs.

        Args:
            output_file (str): Path to save metrics JSON

        Returns:
            Dict: Dictionary of metrics
        """
        self.logger.info("Exporting metrics from all runs")

        # Get all runs for the current experiment
        experiment = mlflow.get_experiment_by_name("plant_disease_detection")
        if experiment is None:
            self.logger.warning("No experiment found")
            return {}

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        # Collect metrics
        metrics = {
            "total_runs": len(runs),
            "inference_times": runs[
                "metrics.inference_time"].dropna().tolist() if "metrics.inference_time" in runs else [],
            "detection_counts": runs[
                "metrics.detection_count"].dropna().tolist() if "metrics.detection_count" in runs else [],
            "average_confidences": runs[
                "metrics.average_confidence"].dropna().tolist() if "metrics.average_confidence" in runs else [],
            "run_metadata": []
        }

        # Add metadata for each run
        for _, run in runs.iterrows():
            run_data = {
                "run_id": run["run_id"],
                "start_time": run["start_time"],
                "source_type": run["params.source_type"] if "params.source_type" in run else None,
                "confidence_threshold": run[
                    "params.confidence_threshold"] if "params.confidence_threshold" in run else None
            }
            metrics["run_metadata"].append(run_data)

        # Calculate aggregated metrics
        if metrics["inference_times"]:
            metrics["avg_inference_time"] = sum(metrics["inference_times"]) / len(metrics["inference_times"])

        if metrics["detection_counts"]:
            metrics["avg_detection_count"] = sum(metrics["detection_counts"]) / len(metrics["detection_counts"])

        if metrics["average_confidences"]:
            metrics["overall_avg_confidence"] = sum(metrics["average_confidences"]) / len(
                metrics["average_confidences"])

        # Save metrics to file
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        self.logger.info(f"Metrics exported to {output_file}")

        return metrics


def main():
    """Main function to run the plant disease detector."""
    parser = argparse.ArgumentParser(description="MLOps Plant Disease Detection")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--model", type=str, help="Path to YOLOv8 model file (overrides config)")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--webcam", action="store_true", help="Use webcam for real-time detection")
    parser.add_argument("--no-display", action="store_true", help="Don't display results (for headless mode)")
    parser.add_argument("--export-metrics", action="store_true", help="Export metrics from all runs")
    parser.add_argument("--save-recommendation", action="store_true", default=True,
                        help="Save detection results to recommendation files")
    parser.add_argument("--no-save-recommendation", action="store_true",
                        help="Don't save detection results to recommendation files")

    args = parser.parse_args()

    # Determine whether to save to recommendation files
    save_recommendation = args.save_recommendation and not args.no_save_recommendation

    # Initialize detector
    detector = PlantDiseaseDetector(config_path=args.config)

    # Load model
    detector.load_model(args.model)

    # Export metrics if requested
    if args.export_metrics:
        detector.export_metrics()
        return

    # Process image if specified
    if args.image:
        detector.process_image(args.image, display=not args.no_display, save_to_recommendation=save_recommendation)

    # Process video if specified
    elif args.video:
        detector.process_video(args.video)

    # Use webcam if specified or if no input is provided
    elif args.webcam or (not args.image and not args.video):
        detector.run_webcam(save_to_recommendation=save_recommendation)


if __name__ == "__main__":
    main()