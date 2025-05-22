import os
import cv2
import numpy as np
from ultralytics import YOLO
import argparse


def detect_plant_diseases(model_path, image_path=None, video_path=None, output_dir='results'):
    """
    Detect plant diseases using YOLOv8 model.

    Args:
        model_path (str): Path to the YOLOv8 model (.pt file)
        image_path (str, optional): Path to input image
        video_path (str, optional): Path to input video
        output_dir (str, optional): Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the YOLOv8 model
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Print available classes
    class_names = model.names
    print("\nAvailable classes for detection:")
    for idx, name in class_names.items():
        print(f"  - {idx}: {name}")

    # Process image
    if image_path:
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return

        print(f"\nProcessing image: {image_path}")
        results = model.predict(source=image_path, save=True, project=output_dir, name="image_results")

        # Display results
        img = cv2.imread(image_path)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add label
                label = f"{class_names[cls]} {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show image with detections
        cv2.imshow("Plant Disease Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Process video
    if video_path:
        if not os.path.exists(video_path):
            print(f"Error: Video not found at {video_path}")
            return

        print(f"\nProcessing video: {video_path}")
        results = model.predict(source=video_path, save=True, project=output_dir, name="video_results")
        print(f"Video processing completed. Results saved to {output_dir}/video_results")

    # If no input is specified, use webcam
    if not image_path and not video_path:
        print("\nNo input specified. Using webcam for real-time detection.")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform detection
            results = model.predict(source=frame, conf=0.25, verbose=False)

            # Get the rendered frame with detections
            annotated_frame = results[0].plot()

            # Display the frame
            cv2.imshow("Plant Disease Detection (Webcam)", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Plant Disease Detection")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to YOLOv8 model file")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--video", type=str, default=None, help="Path to input video")
    parser.add_argument("--output", type=str, default="results", help="Directory to save results")

    args = parser.parse_args()

    detect_plant_diseases(args.model, args.image, args.video, args.output)